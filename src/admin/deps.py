from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from fastapi.templating import Jinja2Templates
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import async_sessionmaker

from data.database import create_engine, create_session_factory

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    bot_token: SecretStr
    bot_dev_token: SecretStr | None = None
    admin_username: str
    admin_password_hash: str
    admin_session_secret: str
    admin_id: int
    cookie_secure: bool = True
    session_ttl_days: int = 7
    admin_tz: str = "Europe/Moscow"
    admin_media_dir: str = "media"
    llm_proxy: SecretStr | None = None
    daily_spend_alert_usd: float = 100.0
    # Пороги тиров — нужны карточке юзера, чтобы показывать реальный лимит,
    # а не хардкод в шаблоне. Значения те же, что читает бот из общего .env.
    user_spend_limit_usd: float = 1.0
    new_user_spend_limit_usd: float = 1.0
    spend_limit_exempt_usernames: str = ""
    sentry_dsn: SecretStr | None = None

    @property
    def effective_bot_token(self) -> SecretStr:
        return self.bot_dev_token or self.bot_token


@lru_cache
def get_settings() -> AdminSettings:
    return AdminSettings()


def make_session_factory(settings: AdminSettings) -> async_sessionmaker:
    engine = create_engine(settings.database_url)
    return create_session_factory(engine)


def make_bot(settings: AdminSettings) -> Bot:
    session = (
        AiohttpSession(proxy=settings.llm_proxy.get_secret_value())
        if settings.llm_proxy
        else None
    )
    return Bot(
        token=settings.effective_bot_token.get_secret_value(),
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
STATIC_DIR = _STATIC_DIR
TEMPLATES_DIR = _TEMPLATES_DIR
