import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from data.database import create_engine, create_session_factory


@pytest.mark.asyncio
async def test_create_engine_returns_async_engine(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'x.db'}"
    engine = create_engine(url)
    try:
        assert isinstance(engine, AsyncEngine)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_session_factory_yields_async_session(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'x.db'}"
    engine = create_engine(url)
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_writers_dont_lock(tmp_path) -> None:
    """Регрессия: две AsyncSession одновременно пишут — не должно быть `database is locked`.

    Воспроизводит ситуацию handler-session + SqlAlchemyFSMStorage-session на
    один update. До фикса (WAL + busy_timeout) второй commit падал с
    sqlite3.OperationalError.
    """
    url = f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}"
    engine = create_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT)"))
        factory = create_session_factory(engine)

        async def writer(k: str, v: str) -> None:
            async with factory() as session:
                await session.execute(
                    text("INSERT INTO t (k, v) VALUES (:k, :v)"), {"k": k, "v": v}
                )
                await asyncio.sleep(0.05)  # удерживаем write-lock
                await session.commit()

        await asyncio.gather(writer("a", "1"), writer("b", "2"))

        async with factory() as session:
            rows = (await session.execute(text("SELECT COUNT(*) FROM t"))).scalar()
            assert rows == 2
    finally:
        await engine.dispose()
