#!/usr/bin/env bash
# Восстановить PG-БД dream_bot из бэкапа (созданного backup_db.sh).
# Запуск на VPS под root (или любым юзером с доступом к sudo -u postgres).
#
# Usage:
#     ./deploy/restore_db.sh /opt/dream-bot/backups/dream_bot_2026-05-26.sql.gz
#
# ВНИМАНИЕ: DROP DATABASE — все текущие данные dream_bot будут затёрты.
# Перед restore делается финальный pg_dump текущего состояния в /tmp на
# случай, если новый dump окажется битым.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup.sql.gz>" >&2
    echo "Available backups:" >&2
    ls -la /opt/dream-bot/backups/ 2>&1 | tail -n +2 >&2
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

# Сначала останавливаем бота, чтобы он не держал коннекты к старой БД
echo ">>> stopping dream-bot service"
sudo systemctl stop dream-bot

# Safety: дамп текущего состояния перед перезаписью
SAFETY_DUMP="/tmp/dream_bot_pre_restore_$(date +%F_%H%M%S).sql.gz"
echo ">>> safety dump current state -> $SAFETY_DUMP"
sudo -u postgres pg_dump --no-owner --no-acl dream_bot | gzip > "$SAFETY_DUMP"

# Restore
echo ">>> drop+recreate database dream_bot"
sudo -u postgres psql -c "DROP DATABASE IF EXISTS dream_bot"
sudo -u postgres psql -c "CREATE DATABASE dream_bot OWNER dream_bot"
sudo -u postgres psql -d dream_bot -c "GRANT ALL ON SCHEMA public TO dream_bot"

echo ">>> restoring from $BACKUP_FILE"
gunzip -c "$BACKUP_FILE" | sudo -u postgres psql dream_bot

echo ">>> starting dream-bot service"
sudo systemctl start dream-bot

sleep 3
sudo systemctl --no-pager status dream-bot | head -10

echo ""
echo "Restore done. Safety dump (pre-restore state): $SAFETY_DUMP"
echo "Удалить вручную после проверки: rm $SAFETY_DUMP"
