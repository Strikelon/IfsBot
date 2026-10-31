"""Тесты для SqlAlchemyFSMStorage — персистентный FSM-storage aiogram3."""

from __future__ import annotations

import pytest
import pytest_asyncio
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.base import StorageKey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from adapters.telegram.fsm_storage import SqlAlchemyFSMStorage, _stringify
from data.models import FsmState


class _SampleStates(StatesGroup):
    FIRST = State()
    SECOND = State()


@pytest_asyncio.fixture
async def storage(test_session, test_session_factory: async_sessionmaker):
    """Возвращает SqlAlchemyFSMStorage поверх тестовой sessionmaker."""
    return SqlAlchemyFSMStorage(test_session_factory)


def _key(chat_id: int = 1, user_id: int = 1) -> StorageKey:
    return StorageKey(bot_id=42, chat_id=chat_id, user_id=user_id)


def test_stringify_is_deterministic_and_unique() -> None:
    """StorageKey → stable строка; разные ключи → разные строки."""
    k1 = StorageKey(bot_id=1, chat_id=10, user_id=100)
    k2 = StorageKey(bot_id=1, chat_id=10, user_id=100)
    k3 = StorageKey(bot_id=1, chat_id=11, user_id=100)
    assert _stringify(k1) == _stringify(k2)
    assert _stringify(k1) != _stringify(k3)


@pytest.mark.asyncio
async def test_get_state_returns_none_for_unknown_key(storage, test_session) -> None:
    result = await storage.get_state(_key(chat_id=999))
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get_state_roundtrip(storage, test_session) -> None:
    key = _key()
    await storage.set_state(key, _SampleStates.FIRST)
    state = await storage.get_state(key)
    assert state == _SampleStates.FIRST.state


@pytest.mark.asyncio
async def test_set_state_to_none_clears(storage, test_session) -> None:
    """set_state(key, None) обнуляет state, но не data — это контракт aiogram."""
    key = _key()
    await storage.set_state(key, _SampleStates.FIRST)
    await storage.set_data(key, {"x": 1})
    await storage.set_state(key, None)

    assert await storage.get_state(key) is None
    # data сохранилась
    assert await storage.get_data(key) == {"x": 1}


@pytest.mark.asyncio
async def test_get_data_returns_empty_dict_for_unknown_key(storage) -> None:
    """Контракт aiogram BaseStorage: get_data() для неизвестного key → {}."""
    assert await storage.get_data(_key(chat_id=888)) == {}


@pytest.mark.asyncio
async def test_set_and_get_data_roundtrip(storage) -> None:
    key = _key()
    payload = {"dream_id": 5, "session_id": 42, "nested": [1, 2, 3]}
    await storage.set_data(key, payload)
    result = await storage.get_data(key)
    assert result == payload


@pytest.mark.asyncio
async def test_set_data_does_not_clobber_state(storage) -> None:
    """set_data(...) НЕ должен затирать state — у aiogram это два независимых поля."""
    key = _key()
    await storage.set_state(key, _SampleStates.FIRST)
    await storage.set_data(key, {"k": "v"})
    state = await storage.get_state(key)
    assert state == _SampleStates.FIRST.state


@pytest.mark.asyncio
async def test_set_state_does_not_clobber_data(storage) -> None:
    """Обратное: set_state(...) НЕ затирает data."""
    key = _key()
    await storage.set_data(key, {"k": "v"})
    await storage.set_state(key, _SampleStates.SECOND)
    data = await storage.get_data(key)
    assert data == {"k": "v"}


@pytest.mark.asyncio
async def test_multiple_users_isolated(storage) -> None:
    """Ключи разных юзеров не пересекаются."""
    k1 = _key(chat_id=10, user_id=10)
    k2 = _key(chat_id=20, user_id=20)
    await storage.set_state(k1, _SampleStates.FIRST)
    await storage.set_state(k2, _SampleStates.SECOND)
    await storage.set_data(k1, {"who": "first"})
    await storage.set_data(k2, {"who": "second"})

    assert (await storage.get_state(k1)) == _SampleStates.FIRST.state
    assert (await storage.get_state(k2)) == _SampleStates.SECOND.state
    assert (await storage.get_data(k1)) == {"who": "first"}
    assert (await storage.get_data(k2)) == {"who": "second"}


@pytest.mark.asyncio
async def test_state_persists_across_storage_instances(
    test_session, test_session_factory
) -> None:
    """Главный контракт: рестарт бота (= новый storage instance с тем же factory)
    НЕ теряет state. Раньше с MemoryStorage всё стиралось.

    test_session здесь только чтобы поднять схему (миграция/create_all) до того
    как storage'ы начнут с ней общаться.
    """
    _ = test_session  # noqa: F841 — фикстура нужна только для schema setup
    key = _key()

    storage1 = SqlAlchemyFSMStorage(test_session_factory)
    await storage1.set_state(key, _SampleStates.FIRST)
    await storage1.set_data(key, {"important": "context"})
    await storage1.close()

    # Эмулируем рестарт: новый storage instance, тот же session_factory (т.е. БД).
    storage2 = SqlAlchemyFSMStorage(test_session_factory)
    assert await storage2.get_state(key) == _SampleStates.FIRST.state
    assert await storage2.get_data(key) == {"important": "context"}


@pytest.mark.asyncio
async def test_fsm_storage_writes_to_fsm_state_table(storage, test_session) -> None:
    """Sanity check: данные реально оседают в таблице fsm_state."""
    key = _key()
    await storage.set_state(key, _SampleStates.FIRST)

    # test_session — отдельный AsyncSession, увидит коммиты из storage
    rows = (await test_session.execute(select(FsmState))).scalars().all()
    assert len(rows) == 1
    assert rows[0].state == _SampleStates.FIRST.state
