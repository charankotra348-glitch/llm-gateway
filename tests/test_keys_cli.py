import asyncio

import pytest

from gateway import keys


def test_cli_create_list_revoke(database_url, monkeypatch, capsys):
    from gateway.config import get_settings
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    try:
        asyncio.run(keys._main(["create", "--name", "cli-app", "--budget", "2.5"]))
        out = capsys.readouterr().out
        key = next(w for w in out.split() if w.startswith("gw_") and not w.endswith("..."))
        prefix = key[:9]
        asyncio.run(keys._main(["list"]))
        listing = capsys.readouterr().out
        assert "cli-app" in listing and "$2.50/month" in listing and "active" in listing and key not in listing
        asyncio.run(keys._main(["revoke", prefix]))
        asyncio.run(keys._main(["list"]))
        assert "revoked" in capsys.readouterr().out
        with pytest.raises(SystemExit):
            asyncio.run(keys._main(["revoke", "gw_nope"]))
    finally:
        get_settings.cache_clear()
