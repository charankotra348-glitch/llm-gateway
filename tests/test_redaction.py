"""Personal data and secrets never reach a provider, and the client never notices."""

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from gateway.db import make_engine, make_sessions
from gateway.main import Gateway, create_app
from gateway.models import RequestLog
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.redaction import StreamRestorer, Vault, luhn_ok, redact, redact_text
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable
from gateway.schemas import ChatRequest, Message


@pytest.mark.parametrize("text, kind", [
    ("jane.doe@acme.co.uk", "EMAIL"), ("JANE+tag@mail.example.com", "EMAIL"),
    ("4111 1111 1111 1111", "CARD"), ("4111-1111-1111-1111", "CARD"), ("378282246310005", "CARD"),
    ("123-45-6789", "SSN"),
    ("(555) 123-4567", "PHONE"), ("555.123.4567", "PHONE"), ("+1 555-123-4567", "PHONE"), ("+44 20 7946 0958", "PHONE"),
    ("192.168.1.20", "IP"),
    ("sk-proj-abcdefghijklmnopqrstuv1234", "SECRET"), ("AKIAIOSFODNN7EXAMPLE", "SECRET"),
    ("ghp_" + "a" * 36, "SECRET"), ("AIza" + "b" * 35, "SECRET"),
])
def test_finds(text, kind):
    vault = Vault()
    assert redact_text(f"see {text} please", vault) == f"see [{kind}_1] please"
    assert vault.originals[f"[{kind}_1]"] == text


@pytest.mark.parametrize("text", [
    "order 1234567890", "on 2026-10-24 at 10:30", "version 1.2.3", "total $1,234.56", "zip 11794",
    "ISBN 978-3-16-148410-0", "4111-1111-1111-1112", "000-12-3456", "999.1.1.1", "room 555-1234",
    "the year 2027", "sk-short",
])
def test_leaves_alone(text):
    vault = Vault()
    assert redact_text(text, vault) == text and vault.total == 0


def test_luhn():
    assert luhn_ok("4111111111111111") and luhn_ok("5500 0000 0000 0004")
    assert not luhn_ok("4111111111111112") and not luhn_ok("123")


def test_same_value_same_placeholder_across_messages():
    chat = ChatRequest(messages=[Message(role="system", text="Reply to jane@acme.com"),
                                 Message(role="user", text="Cc bob@acme.com and jane@acme.com")])
    out, vault = redact(chat)
    assert out.messages[0].text == "Reply to [EMAIL_1]"
    assert out.messages[1].text == "Cc [EMAIL_2] and [EMAIL_1]"
    assert vault.total == 2 and chat.messages[0].text == "Reply to jane@acme.com"   # original untouched


def test_restore_ignores_unknown_and_lookalike_tags():
    vault = Vault()
    redact_text("jane@acme.com", vault)
    assert vault.restore("to [EMAIL_1], not [EMAIL_7] or [link]") == "to jane@acme.com, not [EMAIL_7] or [link]"


@pytest.mark.parametrize("pieces", [
    ["Write to [EMAIL_1] now"],
    ["Write to [", "EMAIL_1] now"],
    ["Write to [EM", "AIL", "_1", "] now"],
    ["Write to [EMAIL_1", "]", " now"],
])
def test_stream_restorer_handles_split_placeholders(pieces):
    vault = Vault()
    redact_text("jane@acme.com", vault)
    r = StreamRestorer(vault)
    assert "".join(r.feed(p) for p in pieces) + r.flush() == "Write to jane@acme.com now"


def test_stream_restorer_does_not_hold_ordinary_brackets_forever():
    r = StreamRestorer(Vault())
    assert r.feed("a list [1, 2") == "a list [1, 2"        # can't be a placeholder: sent at once
    assert r.feed("x [NOTE") == "x "                         # could be: held...
    assert r.feed("] done") == "[NOTE] done"                 # ...until it clearly isn't
    assert r.feed("tail [A") == "tail " and r.flush() == "[A"


# --------------------------------------------------------------------------- through the API
@pytest.fixture
def gateway_with(database_url, make_key):
    def make(redact_pii=True, chunk_delay_s=0.0):
        fake = FakeProvider(chunk_delay_s=chunk_delay_s)
        engine = make_engine(database_url)
        gw = Gateway({"fake": fake},
                     RouteTable.from_dict({"routes": {"demo": [{"provider": "fake", "model": "fake-echo"}]}}),
                     make_sessions(engine), prices=PriceTable.from_dict({"free_providers": ["fake"]}),
                     resilience=Resilience(ResiliencePolicy(backoff_s=0)), redact_pii=redact_pii, engine=engine)
        client = TestClient(create_app(gw)).__enter__()
        clients.append(client)
        _, key = make_key()
        return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0), fake
    clients = []
    yield make
    for c in clients:
        c.__exit__(None, None, None)


PROMPT = "Email jane@acme.com, card 4111 1111 1111 1111"


def test_provider_sees_placeholders_client_sees_originals(gateway_with, db, caplog):
    sdk, fake = gateway_with()
    raw = sdk.chat.completions.with_raw_response.create(model="demo", messages=[{"role": "user", "content": PROMPT}])
    assert fake.calls[0][1].messages[0].text == "Email [EMAIL_1], card [CARD_1]"
    assert raw.parse().choices[0].message.content == f"echo: {PROMPT}"
    assert raw.headers["x-gateway-redactions"] == "2"

    async def rows(s):
        return list(await s.scalars(select(RequestLog)))
    assert db(rows)[0].redactions == 2
    assert "jane@acme.com" not in caplog.text and '"redacted": {"EMAIL": 1, "CARD": 1}' in caplog.text


def test_streamed_answers_are_restored_too(gateway_with):
    sdk, fake = gateway_with()
    chunks = list(sdk.chat.completions.create(model="demo", stream=True,
                                              messages=[{"role": "user", "content": PROMPT}]))
    assert fake.calls[0][1].messages[0].text == "Email [EMAIL_1], card [CARD_1]"
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == f"echo: {PROMPT}" and "[EMAIL_1]" not in text


def test_redaction_can_be_turned_off(gateway_with):
    sdk, fake = gateway_with(redact_pii=False)
    raw = sdk.chat.completions.with_raw_response.create(model="demo", messages=[{"role": "user", "content": PROMPT}])
    assert fake.calls[0][1].messages[0].text == PROMPT and "x-gateway-redactions" not in raw.headers


def test_clean_prompts_report_zero(gateway_with):
    sdk, _ = gateway_with()
    raw = sdk.chat.completions.with_raw_response.create(model="demo", messages=[{"role": "user", "content": "hi"}])
    assert raw.headers["x-gateway-redactions"] == "0"
