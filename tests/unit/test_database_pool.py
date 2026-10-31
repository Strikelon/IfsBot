from sqlalchemy.pool import StaticPool

from data.database import create_engine


def test_postgres_url_gets_tuned_pool() -> None:
    engine = create_engine("postgresql+asyncpg://u:p@localhost/db")
    pool = engine.pool
    # asyncpg-движок получает расширенный пул, а не дефолт (5)
    assert pool.size() == 20
    assert engine.pool._pre_ping is True


def test_sqlite_url_keeps_default_pool() -> None:
    # sqlite-ветка не должна получать asyncpg-тюнинг пула (pool_size/pre_ping).
    # Страж от регрессии: если кто-то применит asyncpg-ветку к sqlite, движок
    # получит QueuePool вместо StaticPool — этот тест упадёт.
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    assert isinstance(engine.pool, StaticPool)
