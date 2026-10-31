#!/usr/bin/env bash
# Daily PG backup для dream-bot. Запускается из cron под юзером dream-bot.
# Сохраняет dump в /opt/dream-bot/backups/, ротация — 7 дней.
set -euo pipefail

BACKUP_DIR="/opt/dream-bot/backups"
RETENTION_DAYS=7
TIMESTAMP=$(date +%F)
DUMP_FILE="$BACKUP_DIR/dream_bot_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

# Читаем DATABASE_URL из .env — там вся PG-конфигурация (host/port/user/db/password)
# Используем pg_dump через ~/.pgpass или через PGPASSWORD env (последнее проще).
# .env лежит в /opt/dream-bot/.env, формат: DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db
ENV_FILE="/opt/dream-bot/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found" >&2
    exit 1
fi

# shellcheck disable=SC1090
DB_URL=$(grep -E '^DATABASE_URL=' "$ENV_FILE" | head -1 | cut -d'=' -f2-)
# Убираем sqlalchemy-префикс +asyncpg для pg_dump
DB_URL_PG=${DB_URL/+asyncpg/}

pg_dump --no-owner --no-acl "$DB_URL_PG" | gzip > "$DUMP_FILE"

# Ротация
find "$BACKUP_DIR" -name 'dream_bot_*.sql.gz' -mtime +${RETENTION_DAYS} -delete

# Краткий лог
echo "$(date -Iseconds) backup done: $DUMP_FILE ($(du -h "$DUMP_FILE" | cut -f1))"
