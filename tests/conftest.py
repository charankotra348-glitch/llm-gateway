import pytest
from fastapi.testclient import TestClient

from gateway.main import Gateway, create_app
from gateway.providers.fake import FakeProvider
from gateway.routes import RouteTable


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def client(fake):
    routes = RouteTable.from_dict({"routes": {"demo": [{"provider": "fake", "model": "fake-echo"}]}})
    with TestClient(create_app(Gateway({"fake": fake}, routes, timeout_s=1.0))) as c:
        yield c
