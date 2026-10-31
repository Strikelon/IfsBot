import itertools
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Обязательная настройка без разумного дефолта: тестам достаточно заглушки.
os.environ.setdefault("SUPPORT_CONTACT_URL", "https://t.me/test_support")

from app.config import Settings  # noqa: E402
from data.models import Base, ExperienceLevel, Session, SessionStatus, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import KIND_DREAM


@pytest_asyncio.fixture
async def test_engine(tmp_path) -> AsyncIterator[AsyncEngine]:
    """Per-test async SQLite engine на временном файле."""
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine) -> AsyncIterator[None]:
    """Create all tables before the test and drop them after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def test_session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def test_session(
    schema,  # noqa: ARG001 — ensures schema is set up before session and torn down after
    test_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with test_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()  # release write lock before schema fixture's drop_all


@pytest_asyncio.fixture
async def db_session(test_session: AsyncSession) -> AsyncSession:
    """Алиас test_session для тестов, написанных под имя db_session."""
    return test_session


@pytest_asyncio.fixture
async def make_user(db_session: AsyncSession):
    """Фабрика юзеров для тестов аналитики (уникальный tg_id на вызов)."""
    counter = itertools.count(1)

    async def _make(
        display_name: str = "Катя", onboarded: bool = False, with_dream: bool = False
    ):
        n = next(counter)
        user_repo = UserRepository(db_session)
        user = await user_repo.get_or_create_by_tg_id(
            telegram_user_id=77_000 + n, display_name=display_name
        )
        if onboarded:
            await user_repo.mark_onboarded(user_id=user.id)
        if with_dream:
            dream = await DreamRepository(db_session).create(
                user_id=user.id,
                recorded_at=datetime.now(UTC),
                transcription="мне снился тигр",
            )
            # Тестовый атрибут-удобство (НЕ колонка модели): id записанного сна.
            user.last_dream_id = dream.id
        await db_session.commit()
        return user

    return _make


@pytest_asyncio.fixture
async def make_quick_session(db_session: AsyncSession):
    """Фабрика «онбордженый юзер + сон + Quick-сессия» (по образцу test_quick_handlers).

    Принимает status= (default — активная in_progress); "paused"/"completed" —
    для display-тестов, где нужна inactive-сессия. user_id= — использовать
    существующего юзера (из make_user) вместо создания нового. kind= — тип
    обращения (default dream: все тесты до типов обращения ждут именно его).
    """
    counter = itertools.count(1)

    async def _make(
        status: str = "in_progress",
        user_id: int | None = None,
        kind: str = KIND_DREAM,
    ) -> Session:
        n = next(counter)
        if user_id is None:
            user_repo = UserRepository(db_session)
            user = await user_repo.get_or_create_by_tg_id(
                telegram_user_id=42_000 + n, display_name="Катя"
            )
            await user_repo.update_profile(
                user_id=user.id,
                profile_life_context="меняю работу",
                profile_experience_level=ExperienceLevel.NEWBIE,
            )
            await user_repo.mark_onboarded(user_id=user.id)
            user_id = user.id
        dream = await DreamRepository(db_session).create(
            user_id=user_id,
            recorded_at=datetime.now(UTC),
            transcription="мне снился тигр",
            kind=kind,
        )
        sess = await SessionRepository(db_session).create(
            dream_id=dream.id, user_id=user_id, type_=SessionType.QUICK
        )
        if status != SessionStatus.IN_PROGRESS.value:
            sess.status = SessionStatus(status)
            await db_session.flush()
        await db_session.commit()
        return sess

    return _make


@pytest.fixture
def fake_message():
    """Фабрика Message-моков (по образцу _text из test_quick_handlers):
    from_user.id = telegram_user_id переданного юзера, answer=AsyncMock()."""

    def _make(user):
        m = MagicMock()
        m.from_user = SimpleNamespace(
            id=user.telegram_user_id, is_bot=False, first_name="Катя"
        )
        m.chat = SimpleNamespace(id=user.telegram_user_id, type="private")
        m.text = "без контекста"
        m.voice = None
        m.answer = AsyncMock()
        return m

    return _make


@pytest.fixture
def fsm_state() -> FSMContext:
    """Реальный FSMContext на MemoryStorage (по образцу fsm из test_quick_handlers)."""
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


@pytest.fixture
def settings() -> Settings:
    """Инстанс Settings с тестовыми значениями (hermetic: репозиторный .env не читаем)."""
    return Settings(
        _env_file=None,
        bot_token="42:TEST",
        database_url="sqlite+aiosqlite://",
        openai_api_key="sk-test",
        openrouter_api_key="sk-or-test",
    )
