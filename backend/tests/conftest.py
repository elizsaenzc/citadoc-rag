import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.container import build_container


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", llm_provider="demo", _env_file=None)


@pytest.fixture
def container(settings):
    return build_container(settings)


@pytest.fixture
def client(settings, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    get_settings.cache_clear()
    with TestClient(main.app) as test_client:
        yield test_client
