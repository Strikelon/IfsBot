import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'health.db'}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("LLM_PROXY", raising=False)
    monkeypatch.chdir(tmp_path)
    app = create_app(start_bot=False)
    return TestClient(app)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
