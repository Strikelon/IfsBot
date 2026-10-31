import pytest

from data.repositories import UserRepository


@pytest.mark.asyncio
async def test_get_or_create_idempotent(test_session):
    repo = UserRepository(test_session)
    u1 = await repo.get_or_create_by_tg_id(telegram_user_id=555, display_name="A")
    u2 = await repo.get_or_create_by_tg_id(telegram_user_id=555, display_name="B")
    assert u1.id == u2.id


@pytest.mark.asyncio
async def test_get_or_create_recovers_from_concurrent_insert(test_session_factory, schema):  # noqa: ARG001
    """Гонка двух апдейтов: сессия A создаёт и коммитит юзера; сессия B в окне
    гонки «не видит» его (get_by_tg_id мокнут на miss), пытается вставить → ловит
    UniqueViolation и берёт уже существующего, без краша."""
    # Сессия A — «победитель гонки», коммитит юзера.
    async with test_session_factory() as sess_a:
        u1 = await UserRepository(sess_a).get_or_create_by_tg_id(
            telegram_user_id=777, display_name="A"
        )
        await sess_a.commit()
        u1_id = u1.id

    # Сессия B — INSERT конфликтует с уже закоммиченной строкой.
    async with test_session_factory() as sess_b:
        repo_b = UserRepository(sess_b)
        real_get = repo_b.get_by_tg_id
        calls = {"n": 0}

        async def flaky_get(tg_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # окно гонки: строку ещё «не видим»
            return await real_get(tg_id)

        repo_b.get_by_tg_id = flaky_get
        u2 = await repo_b.get_or_create_by_tg_id(telegram_user_id=777, display_name="B")
        assert u2.id == u1_id  # восстановились на существующего, без IntegrityError
