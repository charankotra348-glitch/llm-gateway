import pytest

from gateway.errors import ModelNotFound
from gateway.main import Settings, build_gateway
from gateway.routes import RouteTable


def test_route_lookup_and_unknown_model():
    table = RouteTable.from_dict({"routes": {"fast": [{"provider": "fake", "model": "a"},
                                                      {"provider": "fake", "model": "b"}]}})
    assert [t.model for t in table.get("fast").targets] == ["a", "b"]
    with pytest.raises(ModelNotFound, match="fast"):
        table.get("slow")


@pytest.mark.parametrize("data, msg", [
    ({}, "non-empty"),
    ({"routes": {"x": []}}, "route 'x'"),
    ({"routes": {"x": [{"provider": "gemini"}]}}, "route 'x'"),
    ({"routes": {"x": [{"provider": "gemini", "model": "m"}]}}, "not configured"),
])
def test_bad_route_files_fail_at_startup(data, msg):
    with pytest.raises(ValueError, match=msg):
        RouteTable.from_dict(data, known_providers={"fake"})


def test_shipped_routes_file_loads(tmp_path):
    gw = build_gateway(Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path}/x.db"))
    assert gw.routes.names() == ["demo"] and set(gw.providers) == {"fake"}
    assert gw.unpriced_targets() == []
