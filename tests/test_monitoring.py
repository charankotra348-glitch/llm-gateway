import sys
import types

from gateway.config import Settings
from gateway.monitoring import init_error_tracking


def test_off_without_a_dsn(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)   # would fail to import if it were touched
    assert init_error_tracking(Settings(sentry_dsn="")) is False


def test_on_with_a_dsn_and_never_sends_prompts(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "sentry_sdk", types.SimpleNamespace(init=lambda **kw: calls.append(kw)))
    assert init_error_tracking(Settings(sentry_dsn="https://key@o0.ingest.sentry.io/1", sentry_environment="demo"))
    (kw,) = calls
    assert kw["dsn"] == "https://key@o0.ingest.sentry.io/1" and kw["environment"] == "demo"
    # prompts and keys travel in bodies, headers and locals: none of them may be attached to a report
    assert kw["send_default_pii"] is False
    assert kw["max_request_body_size"] == "never"
    assert kw["include_local_variables"] is False
    assert kw["traces_sample_rate"] == 0.0
