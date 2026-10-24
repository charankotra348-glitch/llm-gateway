"""Getting an answer when providers fail: retries, fallback across a route's targets, circuit breakers.

The policy, by what went wrong with one target:

    ProviderUnavailable (5xx, connection refused)   retry the same target, then the next target
    ProviderTimeout                                 next target (a retry would wait just as long again)
    ProviderRateLimited                             next target (a retry would be throttled again)
    ProviderAuthError                               next target (this one is misconfigured)
    ProviderRejected (the request itself is bad)    stop: another provider would most likely reject it too

A circuit breaker per target stops sending traffic to one that keeps failing, so requests stop
paying a timeout each to discover what the last request already found out.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

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
from gateway.schemas import ChatRequest, Completion

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


class Resilience:
    def __init__(self, policy: ResiliencePolicy | None = None, *, clock: Callable[[], float] = time.monotonic,
                 sleep=anyio.sleep, rng: random.Random | None = None):
        self.policy = policy or ResiliencePolicy()
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def breaker(self, target: Target) -> CircuitBreaker:
        key = (target.provider, target.model)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(self.policy.breaker_threshold, self.policy.breaker_cooldown_s,
                                                 clock=self._clock)
        return self._breakers[key]

    def states(self) -> list[dict]:
        return [{"provider": p, "model": m, "state": b.state, "consecutive_failures": b.failures}
                for (p, m), b in sorted(self._breakers.items())]

    async def complete(self, route: Route, chat: ChatRequest, providers: dict[str, Provider]) -> Result:
        """Try the route's targets in order. Returns the first answer, or raises the last error with
        every attempt attached as `exc.attempts`."""
        p = self.policy
        deadline = self._clock() + p.total_timeout_s
        attempts: list[Attempt] = []
        last_error: GatewayError | None = None

        for target in route.targets:
            breaker = self.breaker(target)
            if not breaker.allow():
                attempts.append(Attempt(target.provider, target.model, "circuit_open"))
                continue
            for n in range(p.attempts_per_target):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    breaker.release()
                    raise _fail(ProviderTimeout(f"no answer within {p.total_timeout_s}s"), attempts)
                started = self._clock()
                try:
                    completion = await _call(providers[target.provider], chat, target.model,
                                             min(p.attempt_timeout_s, remaining))
                except ProviderRejected as exc:
                    breaker.release()
                    attempts.append(Attempt(target.provider, target.model, exc.code, _ms(self._clock() - started)))
                    raise _fail(exc, attempts) from None
                except ProviderError as exc:
                    attempts.append(Attempt(target.provider, target.model, exc.code, _ms(self._clock() - started)))
                    breaker.failure()
                    last_error = exc
                    retry_here = (isinstance(exc, ProviderUnavailable) and n + 1 < p.attempts_per_target
                                  and breaker.allow())
                    if not retry_here:
                        break
                    await self._sleep(min(self._rng.uniform(0, p.backoff_s * 2 ** n), max(0.0, remaining)))
                    continue
                attempts.append(Attempt(target.provider, target.model, "ok", _ms(self._clock() - started)))
                breaker.success()
                return Result(completion, target, attempts)

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


def _fail(exc: GatewayError, attempts: list[Attempt]) -> GatewayError:
    exc.attempts = attempts
    return exc


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
