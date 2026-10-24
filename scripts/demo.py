"""A guided tour of the gateway, for recording a demo. Each step pauses until you press Enter.

    python scripts/demo.py --key gw_...                                        # local, http://localhost:8000
    python scripts/demo.py --base-url https://<service>.onrender.com --key gw_... --model fast

--model picks the route for the streaming, redaction and cache steps: `demo` (the fake, instant, no
API key needed) or `fast` (Gemini, on the deployment). --no-pause runs straight through.
"""

import argparse
import collections
import sys
import time

import httpx
import openai

from gateway.redaction import Vault, redact_text


def step(title: str, pause: bool) -> None:
    print(f"\n\033[1m== {title}\033[0m")
    if pause:
        input("   (Enter to run) ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--key", required=True)
    parser.add_argument("--model", default="demo", help="route for the streaming step")
    parser.add_argument("--no-pause", action="store_true")
    args = parser.parse_args()
    pause = not args.no_pause
    client = openai.OpenAI(base_url=f"{args.base_url}/v1", api_key=args.key, max_retries=0)
    http = httpx.Client(base_url=args.base_url, headers={"authorization": f"Bearer {args.key}"}, timeout=120)

    step("1. An unmodified OpenAI client, pointed at the gateway, streaming", pause)
    print(f'   client = OpenAI(base_url="{args.base_url}/v1", api_key="gw_...")\n   ', end="")
    for chunk in client.chat.completions.create(
            model=args.model, stream=True,
            messages=[{"role": "user", "content": "In two sentences, what does an API gateway do?"}]):
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    step("2. A provider failing half its calls: 20 requests to the `resilient` route", pause)
    served, failed = collections.Counter(), 0
    for i in range(20):
        try:
            r = client.chat.completions.with_raw_response.create(
                model="resilient", messages=[{"role": "user", "content": f"request {i}"}])
            served[r.headers["x-gateway-provider"]] += 1
            print(f"   {i + 1:>2}  answered by {r.headers['x-gateway-provider']:<6} after "
                  f"{r.headers['x-gateway-attempts']} call(s)")
        except openai.APIError as exc:
            failed += 1
            print(f"   {i + 1:>2}  failed: {exc}")
    print(f"   -> {20 - failed} of 20 succeeded; served by {dict(served)}")
    states = http.get("/v1/status").json()["targets"]
    print("   circuit breakers:", ", ".join(f"{s['provider']} {s['state']}" for s in states))

    step("3. Personal data never reaches the provider", pause)
    prompt = "Draft a reply to jane.doe@acme.com confirming card 4111 1111 1111 1111 was refunded."
    r = client.chat.completions.with_raw_response.create(model=args.model,
                                                         messages=[{"role": "user", "content": prompt}])
    print(f"   sent:      {prompt}")
    print(f"   provider:  {redact_text(prompt, Vault())}")      # the same function the gateway runs
    print(f"   answer:    {r.parse().choices[0].message.content}")
    print(f"   x-gateway-redactions: {r.headers.get('x-gateway-redactions')}  (the log records kinds and counts only)")

    step("4. The same question twice at temperature 0: the second comes from the cache", pause)
    for attempt in ("first", "second"):
        t = time.perf_counter()
        r = client.chat.completions.with_raw_response.create(
            model=args.model, temperature=0,
            messages=[{"role": "user", "content": "In one sentence, what is a refund window?"}])
        print(f"   {attempt:<6} {r.headers.get('x-gateway-cache', '-'):<5} "
              f"{(time.perf_counter() - t) * 1000:6.1f} ms  provider: {r.headers['x-gateway-provider']}")

    step("5. What this key has used this month", pause)
    time.sleep(0.5)          # ledger rows are written in the background, a few milliseconds behind
    for k, v in http.get("/v1/usage").json().items():
        print(f"   {k:<20} {v}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
