"""The Day 1 bar: an unmodified OpenAI SDK client works against the gateway."""

import openai
import pytest

from gateway.errors import ProviderUnavailable


@pytest.fixture
def sdk(client):
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client,
                         max_retries=0)


def test_chat_completion_through_the_sdk(sdk, fake):
    resp = sdk.chat.completions.create(
        model="demo",
        messages=[{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hello there"}],
        temperature=0.2, max_completion_tokens=50, stop="END",
    )
    assert resp.object == "chat.completion"
    assert resp.id.startswith("chatcmpl-")
    assert resp.model == "fake-echo"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "echo: hello there"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens

    model, sent = fake.calls[0]
    assert model == "fake-echo"
    assert sent.system == "Be brief."          # developer -> system
    assert sent.max_tokens == 50 and sent.stop == ["END"] and sent.temperature == 0.2


def test_raw_response_carries_route_headers(sdk):
    raw = sdk.chat.completions.with_raw_response.create(model="demo", messages=[{"role": "user", "content": "x"}])
    assert raw.headers["x-gateway-route"] == "demo"
    assert raw.headers["x-gateway-provider"] == "fake"


def test_length_finish_reason(sdk, fake):
    fake.reply = "one two three four"
    resp = sdk.chat.completions.create(model="demo", max_tokens=2, messages=[{"role": "user", "content": "x"}])
    assert resp.choices[0].finish_reason == "length"
    assert resp.choices[0].message.content == "one two"


def test_list_models(sdk):
    assert [m.id for m in sdk.models.list()] == ["demo"]


def test_unknown_model_is_404(sdk):
    with pytest.raises(openai.NotFoundError) as exc:
        sdk.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}])
    assert exc.value.code == "model_not_found"


def test_bad_request_is_400_in_openai_shape(sdk):
    with pytest.raises(openai.BadRequestError) as exc:
        sdk.chat.completions.create(model="demo", temperature=5, messages=[{"role": "user", "content": "x"}])
    assert exc.value.body["type"] == "invalid_request_error"
    assert "temperature" in exc.value.body["message"]


@pytest.mark.parametrize("kwargs, needle", [
    ({"n": 2}, "n must be 1"),
    ({"stream": True}, "stream"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}}, "json_schema"),
])
def test_unsupported_options_are_refused_not_ignored(sdk, kwargs, needle):
    with pytest.raises(openai.BadRequestError) as exc:
        sdk.chat.completions.create(model="demo", messages=[{"role": "user", "content": "x"}], **kwargs)
    assert needle in exc.value.message


def test_image_parts_are_refused(sdk):
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.create(model="demo", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}])


def test_text_parts_are_joined(sdk, fake):
    sdk.chat.completions.create(model="demo", messages=[{"role": "user", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert fake.calls[0][1].messages[0].text == "ab"


def test_provider_failure_is_a_server_error(sdk, fake):
    async def boom(*a, **k):
        raise ProviderUnavailable("down")
    fake.complete = boom
    with pytest.raises(openai.APIStatusError) as exc:
        sdk.chat.completions.create(model="demo", messages=[{"role": "user", "content": "x"}])
    assert exc.value.status_code == 502
    assert exc.value.body["code"] == "provider_unavailable"


def test_missing_usage_is_null_not_invented(sdk, fake):
    from gateway.schemas import Completion

    async def no_usage(request, *, model, timeout_s):
        return Completion(text="hi", model=model)
    fake.complete = no_usage
    resp = sdk.chat.completions.create(model="demo", messages=[{"role": "user", "content": "x"}])
    assert resp.usage is None
