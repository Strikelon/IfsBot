from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Создать async engine. Для asyncpg — расширенный пул; для SQLite — WAL-прагмы.

    Для SQLite-URL включаются WAL + busy_timeout: без них две конкурентные
    AsyncSession (handler-session + SqlAlchemyFSMStorage-session) на один
    update получают `database is locked` мгновенно. WAL разрешает читателей
    параллельно писателю, busy_timeout заставляет второго писателя ждать
    освобождения лока вместо немедленного отказа.

    Для asyncpg (и прочего не-SQLite) применяется расширенный пул под спайк трафика:
    pool_size=20 + max_overflow=20 + pool_pre_ping=True + pool_recycle=1800.
    """
    if database_url.startswith("sqlite"):
        engine = create_async_engine(database_url, echo=echo, future=True)
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        return engine
    # asyncpg / прочее: под спайк нужен запас коннектов (на update — 2 сессии).
    return create_async_engine(
        database_url,
        echo=echo,
        future=True,
        pool_size=20,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Фабрика AsyncSession с `expire_on_commit=False` — стандартная практика для FastAPI."""
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
