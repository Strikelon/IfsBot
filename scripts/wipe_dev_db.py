"""Очистка ВСЕХ данных в dev-БД (для прохождения онбординга заново).

Защита: выполняется только если DATABASE_URL указывает на локальный хост
(127.0.0.1 / localhost). TRUNCATE всех таблиц, кроме alembic_version.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import Settings  # noqa: E402


async def main() -> None:
    s = Settings()
    url = s.database_url
    # На сервере база тоже на 127.0.0.1, поэтому одного адреса мало: dev-токен
    # задаётся только на своём компьютере и на сервер не уезжает.
    if s.bot_dev_token is None:
        print("[ОТКАЗ] BOT_DEV_TOKEN не задан — похоже, это не локальный стенд. Не трогаю.")
        sys.exit(2)
    if not ("127.0.0.1" in url or "localhost" in url or "@localhost" in url):
        print(f"[ОТКАЗ] DATABASE_URL не локальный — не трогаю: {url}")
        sys.exit(2)
    print(f"[ok] Локальная БД: {url}")

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            rows = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename <> 'alembic_version'"
                )
            )
            tables = [r[0] for r in rows]
            if not tables:
                print("[ok] Таблиц для очистки нет.")
                return
            joined = ", ".join(f'"{t}"' for t in tables)
            await conn.execute(text(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE"))
            print(f"[ok] Очищено таблиц: {len(tables)} -- {tables}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
