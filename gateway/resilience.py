"""Getting an answer when providers fail: retries, fallback across a route's targets, circuit breakers.

The policy, by what went wrong with one target:

    ProviderUnavailable (5xx, connection refused)   retry the same target, then the next target
    ProviderTimeout                                 next target (a retry would wait just as long again)
    ProviderRateLimited                             next target (a retry would be throttled again)
    ProviderAuthError                               next target (this one is misconfigured)
    ProviderRejected (the request itself is bad)    stop: another provider would most likely reject it too

A circuit breaker per target stops sending traffic to one that keeps failing, so requests stop
paying a timeout each to discover what the last request already found out.

Streams follow the same policy up to their first chunk. After that the client has started
receiving an answer, so a failure can no longer be hidden by switching providers: it ends the
stream with an error.
"""

import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anyio

from gateway.errors import (
    GatewayError,
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.providers.base import Provider
from gateway.routes import Route, Target
from gateway.schemas import ChatRequest, Chunk, Completion

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """closed: traffic flows, consecutive failures are counted.
    open: after `threshold` consecutive failures, the target is skipped for `cooldown_s`.
    half_open: after the cooldown, ONE request is let through as a probe; success closes the
    breaker, failure opens it for another cooldown. Everyone else keeps skipping meanwhile."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._state: State = "closed"
        self.failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> State:
        if self._state == "open" and self._clock() - self._opened_at >= self.cooldown_s:
            self._state = "half_open"
        return self._state

    def allow(self) -> bool:
        """May a request go to this target now? Claims the probe slot when half-open."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def success(self) -> None:
        self._state, self.failures, self._probe_in_flight = "closed", 0, False

    def failure(self) -> None:
        self._probe_in_flight = False
        self.failures += 1
        if self._state == "half_open" or self.failures >= self.threshold:
            self._state, self._opened_at = "open", self._clock()

    def release(self) -> None:
        """The attempt ended without saying anything about the target's health (e.g. a bad request)."""
        self._probe_in_flight = False


class Permit:
    """Permission from a breaker store to call one target once. `probe` marks the single trial call
    a half-open breaker lets through; `token` identifies it in a shared store."""

    __slots__ = ("probe", "token")

    def __init__(self, probe: bool = False, token: str | None = None):
        self.probe, self.token = probe, token


class BreakerStore(Protocol):
    """Where circuit breaker state lives: in this process (LocalBreakers) or shared by every gateway
    process through Redis (gateway.redis_state.RedisBreakers)."""

    async def allow(self, target: Target) -> Permit | None: ...
    async def success(self, target: Target, permit: Permit) -> None: ...
    async def failure(self, target: Target, permit: Permit) -> None: ...
    async def release(self, target: Target, permit: Permit) -> None: ...
    async def states(self) -> list[dict]: ...


class LocalBreakers:
    """One CircuitBreaker per target, in this process."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def get(self, target: Target) -> CircuitBreaker:
        key = (target.provider, target.model)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(self.threshold, self.cooldown_s, clock=self._clock)
        return self._breakers[key]

    async def allow(self, target: Target) -> Permit | None:
        breaker = self.get(target)
        probing = breaker.state == "half_open"
        return Permit(probe=probing) if breaker.allow() else None

    async def success(self, target: Target, permit: Permit) -> None:
        self.get(target).success()

    async def failure(self, target: Target, permit: Permit) -> None:
        self.get(target).failure()

    async def release(self, target: Target, permit: Permit) -> None:
        self.get(target).release()

    async def states(self) -> list[dict]:
        return [{"provider": p, "model": m, "state": b.state, "consecutive_failures": b.failures}
                for (p, m), b in sorted(self._breakers.items())]


@dataclass
class ResiliencePolicy:
    attempt_timeout_s: float = 30.0      # one call to one provider
    total_timeout_s: float = 60.0        # the whole request, across retries and fallbacks
    attempts_per_target: int = 2         # for ProviderUnavailable only
    backoff_s: float = 0.2               # first retry waits up to this; doubles each retry (full jitter)
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 30.0


@dataclass
class Attempt:
    provider: str
    model: str
    outcome: str                         # ok | an error code | circuit_open
    latency_ms: float = 0.0


@dataclass
class Result:
    completion: Completion
    target: Target
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def calls(self) -> int:
        """Provider calls actually made; targets skipped by an open circuit don't count."""
        return sum(a.outcome != "circuit_open" for a in self.attempts)


@dataclass
class StreamResult:
    """A stream that has produced its first chunk. Iterate `chunks()` for the whole answer."""

    target: Target
    attempts: list[Attempt]
    _first: Chunk
    _rest: AsyncIterator[Chunk]
    _breakers: BreakerStore
    _permit: Permit
    _idle_timeout_s: float
    _deadline: float
    _clock: Callable[[], float]

    @property
    def calls(self) -> int:
        return sum(a.outcome != "circuit_open" for a in self.attempts)

    async def chunks(self) -> AsyncIterator[Chunk]:
        """Yields every chunk. Raises a ProviderError if the stream breaks, stalls for longer than the
        per-call timeout, or runs past the request's deadline.

        The circuit breaker hears about the target only now: a stream that starts but always breaks
        halfway is a failing target, so success is counted when the stream finishes."""
        settled = False
        try:
            yield self._first
            if self._first.final:
                settled = True
                await self._breakers.success(self.target, self._permit)
                return
            while True:
                wait = min(self._idle_timeout_s, self._deadline - self._clock())
                if wait <= 0:
                    raise ProviderTimeout("the answer did not finish within the request's time limit")
                try:
                    with anyio.fail_after(wait):
                        chunk = await anext(self._rest)
                except StopAsyncIteration:
                    raise ProviderUnavailable(f"{self.target.provider} ended the stream without finishing") from None
                except TimeoutError:
                    raise ProviderTimeout(f"{self.target.provider} stopped sending for {wait:.1f}s") from None
                yield chunk
                if chunk.final:
                    settled = True
                    await self._breakers.success(self.target, self._permit)
                    return
        except ProviderError:
            settled = True
            await self._breakers.failure(self.target, self._permit)
            raise
        finally:
            with anyio.CancelScope(shield=True):
                if not settled:             # the client left mid-stream: says nothing about the target
                    await self._breakers.release(self.target, self._permit)
                await _close(self._rest)


class Resilience:
    def __init__(self, policy: ResiliencePolicy | None = None, *, clock: Callable[[], float] = time.monotonic,
                 sleep=anyio.sleep, rng: random.Random | None = None, breakers: BreakerStore | None = None):
        self.policy = policy or ResiliencePolicy()
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.breakers: BreakerStore = breakers if breakers is not None else LocalBreakers(
            self.policy.breaker_threshold, self.policy.breaker_cooldown_s, clock=clock)

    def breaker(self, target: Target) -> CircuitBreaker:
        """The in-process breaker for a target (local store only; for tests and inspection)."""
        assert isinstance(self.breakers, LocalBreakers), "breaker state is shared in Redis"
        return self.breakers.get(target)

    async def states(self) -> list[dict]:
        return await self.breakers.states()

    async def complete(self, route: Route, chat: ChatRequest, providers: dict[str, Provider]) -> Result:
        """Try the route's targets in order. Returns the first answer, or raises the last error with
        every attempt attached as `exc.attempts`."""
        async def call(provider: Provider, target: Target, timeout_s: float) -> Completion:
            return await _call(provider, chat, target.model, timeout_s)

        completion, target, attempts, _, _ = await self._run(route, providers, call)
        return Result(completion, target, attempts)

    async def stream(self, route: Route, chat: ChatRequest, providers: dict[str, Provider]) -> StreamResult:
        """Like complete(), but a target counts as answering once it sends its first chunk."""
        async def open_stream(provider: Provider, target: Target, timeout_s: float):
            return await _open_stream(provider, chat, target.model, timeout_s)

        (first, rest), target, attempts, permit, deadline = await self._run(route, providers, open_stream,
                                                                            settle_on_return=False)
        return StreamResult(target, attempts, first, rest, self.breakers, permit, self.policy.attempt_timeout_s,
                            deadline, self._clock)

    async def _run(self, route: Route, providers: dict[str, Provider],
                   attempt: Callable[[Provider, Target, float], Awaitable[Any]], *, settle_on_return: bool = True):
        """The failure policy, shared by complete() and stream(). `attempt` makes one call to one
        target within the timeout it's given. With settle_on_return=False, the caller reports the
        target's success to its breaker later (streams do, when they finish)."""
        p = self.policy
        deadline = self._clock() + p.total_timeout_s
        attempts: list[Attempt] = []
        last_error: GatewayError | None = None

        for target in route.targets:
            permit = await self.breakers.allow(target)
            if permit is None:
                attempts.append(Attempt(target.provider, target.model, "circuit_open"))
                continue
            for n in range(p.attempts_per_target):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    await self.breakers.release(target, permit)
                    raise _fail(ProviderTimeout(f"no answer within {p.total_timeout_s}s"), attempts)
                started = self._clock()
                try:
                    value = await attempt(providers[target.provider], target, min(p.attempt_timeout_s, remaining))
                except ProviderRejected as exc:
                    await self.breakers.release(target, permit)
                    attempts.append(Attempt(target.provider, target.model, exc.code, _ms(self._clock() - started)))
                    raise _fail(exc, attempts) from None
                except ProviderError as exc:
                    attempts.append(Attempt(target.provider, target.model, exc.code, _ms(self._clock() - started)))
                    await self.breakers.failure(target, permit)
                    last_error = exc
                    retry_here = isinstance(exc, ProviderUnavailable) and n + 1 < p.attempts_per_target
                    if not retry_here or (permit := await self.breakers.allow(target)) is None:
                        break
                    await self._sleep(min(self._rng.uniform(0, p.backoff_s * 2 ** n), max(0.0, remaining)))
                    continue
                except BaseException:       # cancelled (the client went away): free a probe slot if we held it
                    with anyio.CancelScope(shield=True):
                        await self.breakers.release(target, permit)
                    raise
                attempts.append(Attempt(target.provider, target.model, "ok", _ms(self._clock() - started)))
                if settle_on_return:
                    await self.breakers.success(target, permit)
                return value, target, attempts, permit, deadline

        if last_error is None:
            raise _fail(ProviderUnavailable("every provider for this model is failing right now (circuit open); "
                                            "try again shortly"), attempts)
        tried = "; ".join(f"{a.provider}/{a.model}: {a.outcome}" for a in attempts)
        raise _fail(type(last_error)(f"all providers failed ({tried}). Last error: {last_error.message}"), attempts)


async def _call(provider: Provider, chat: ChatRequest, model: str, timeout_s: float) -> Completion:
    """Enforce the timeout on the wall clock as well. Adapters pass it to their HTTP client, but an
    HTTP read timeout is per chunk: a provider trickling bytes could otherwise run far past it."""
    try:
        with anyio.fail_after(timeout_s):
            return await provider.complete(chat, model=model, timeout_s=timeout_s)
    except TimeoutError:
        raise ProviderTimeout(f"{provider.name} gave no answer within {timeout_s:.1f}s") from None


async def _open_stream(provider: Provider, chat: ChatRequest, model: str,
                       timeout_s: float) -> tuple[Chunk, AsyncIterator[Chunk]]:
    """Start a stream and wait for its first chunk: until then, failing over is still invisible."""
    stream = aiter(provider.stream(chat, model=model, timeout_s=timeout_s))
    try:
        with anyio.fail_after(timeout_s):
            first = await anext(stream)
    except StopAsyncIteration:
        raise ProviderUnavailable(f"{provider.name} returned an empty stream") from None
    except TimeoutError:
        await _close(stream)
        raise ProviderTimeout(f"{provider.name} sent nothing within {timeout_s:.1f}s") from None
    except BaseException:
        await _close(stream)
        raise
    return first, stream


async def _close(stream: AsyncIterator[Chunk]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        with anyio.CancelScope(shield=True):
            try:
                await aclose()
            except Exception:  # noqa: BLE001  (a stream that fails to close has nothing more to give)
                pass


def _fail(exc: GatewayError, attempts: list[Attempt]) -> GatewayError:
    exc.attempts = attempts
    return exc


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
