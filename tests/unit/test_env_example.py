""".env.example перечисляет каждую обязательную настройку бота, админки и деплоя."""

from pathlib import Path

from admin.deps import AdminSettings
from app.config import Settings

_ROOT = Path(__file__).resolve().parents[2]


def _example_keys() -> set[str]:
    text = (_ROOT / ".env.example").read_text(encoding="utf-8")
    return {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }


def test_env_example_lists_every_setting():
    keys = _example_keys()
    fields = {n.upper() for n in Settings.model_fields} | {
        n.upper() for n in AdminSettings.model_fields
    }
    # Тонкая настройка с разумными дефолтами — в пример не выносится.
    optional = {
        "WEBHOOK_ENABLED", "WEBHOOK_BASE_URL", "IMAGE_QUESTIONS_ENABLED",
        "LLM_MAX_CONCURRENCY", "STT_MAX_CONCURRENCY", "TG_SEND_RATE_PER_SEC",
        "SESSION_TTL_DAYS", "ADMIN_MEDIA_DIR",
    }
    missing = fields - optional - keys
    assert not missing, f"нет в .env.example: {sorted(missing)}"


def test_env_example_lists_deploy_variables():
    keys = _example_keys()
    for var in ("SERVER_IP", "SERVER_USER", "SERVER_SSH_KEY", "SERVER_PASSWORD", "DB_PASSWORD"):
        assert var in keys
