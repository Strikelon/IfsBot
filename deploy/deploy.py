#!/usr/bin/env python3
"""Идемпотентная установка и деплой бота на сервер Ubuntu.

Запуск с локальной машины из корня проекта:
    .venv/bin/python deploy/deploy.py          (Linux/macOS)
    .venv/Scripts/python.exe deploy/deploy.py  (Windows)

Что делает:
    1. apt: postgresql, curl, build-essential
    2. ставит uv в /usr/local/bin/uv
    3. создаёт системного юзера dream-bot и каталог /opt/dream-bot
    4. поднимает Postgres, создаёт роль и БД dream_bot
    5. заливает код по SFTP (папки с кодом перед этим очищаются)
    6. пишет /opt/dream-bot/.env из локального .env
    7. uv sync --frozen (сам ставит Python 3.12) + alembic upgrade head
    8. systemd-сервисы dream-bot и dream-admin, ротация журнала, ежедневный бэкап
    9. перезапускает сервисы и показывает статус

Все настройки — в локальном .env (см. .env.example). Запускать можно на каждом деплое.
"""

from __future__ import annotations

import os
import posixpath
import sys
from pathlib import Path

# Windows-консоль может быть в cp1251 — для удалённых apt-логов и
# спец-символов в наших print'ах форсим UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import paramiko
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: установите зависимости — uv sync (paramiko и python-dotenv в группе dev)")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent

# --- паттерны исключений при SFTP-аплоаде ---
# Правило: на сервер едет только то, что нужно боту в рантайме. Всё остальное —
# инструментарий разработки, материалы ресёрча, артефакты агентов — остаётся локально.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "tests",
    "docs",
    "assets",
    "node_modules",
    "backups",
    "analytics_out",
    ".idea",
    ".vscode",
}
SKIP_FILE_SUFFIXES = (".pyc", ".pyo", ".db", ".db-wal", ".db-shm", ".log")
SKIP_FILE_NAMES = {
    ".env",
    ".env.local",
    "dev.db",
    ".DS_Store",
}

PROJECT_DIR = "/opt/dream-bot"
# Папки с кодом: перед заливкой удаляются на сервере целиком, иначе удалённые
# локально файлы (например, старые миграции) остались бы там и ломали запуск.
CODE_DIRS = ("src", "alembic", "deploy", "scripts")


def load_env() -> dict[str, str]:
    load_dotenv(ROOT / ".env")
    required = [
        "SERVER_IP",
        "SERVER_USER",
        "DB_PASSWORD",
        "BOT_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "SUPPORT_CONTACT_URL",
        # Админка (отдельный сервис dream-admin) — обязательны, иначе admin.main не стартует
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD_HASH",
        "ADMIN_SESSION_SECRET",
        "ADMIN_ID",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if not (os.getenv("SERVER_SSH_KEY") or os.getenv("SERVER_PASSWORD")):
        missing.append("SERVER_SSH_KEY или SERVER_PASSWORD")
    if missing:
        print(f"ERROR: пустые/отсутствующие переменные в .env: {missing}")
        sys.exit(1)
    return {
        "host": os.environ["SERVER_IP"],
        "user": os.environ["SERVER_USER"],
        "password": os.getenv("SERVER_PASSWORD", ""),
        "ssh_key": os.getenv("SERVER_SSH_KEY", ""),
        "port": int(os.getenv("SERVER_SSH_PORT", "22")),
        "project_dir": PROJECT_DIR,
        "db_password": os.environ["DB_PASSWORD"],
        "bot_token": os.environ["BOT_TOKEN"],
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "openrouter_api_key": os.environ["OPENROUTER_API_KEY"],
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "openrouter_generator_model": os.getenv("OPENROUTER_GENERATOR_MODEL", "google/gemini-2.5-pro"),
        "openrouter_classifier_model": os.getenv("OPENROUTER_CLASSIFIER_MODEL", "google/gemini-2.5-flash"),
        # Дефолты совпадают с app/config.py.
        "openrouter_intent_model": os.getenv("OPENROUTER_INTENT_MODEL", "openai/gpt-4o-mini"),
        "openrouter_context_model": os.getenv(
            "OPENROUTER_CONTEXT_MODEL", "google/gemini-3.1-flash-lite"
        ),
        # --- Долговременная память ---
        "memory_write_enabled": os.getenv("MEMORY_WRITE_ENABLED", "true"),
        "memory_inject_enabled": os.getenv("MEMORY_INJECT_ENABLED", "true"),
        "memory_compact_enabled": os.getenv("MEMORY_COMPACT_ENABLED", "true"),
        "memory_compact_budget_chars": os.getenv("MEMORY_COMPACT_BUDGET_CHARS", "2500"),
        "llm_proxy": os.getenv("LLM_PROXY", ""),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        # --- Админка ---
        # BOT_DEV_TOKEN сознательно НЕ пробрасывается на сервер: и бот, и админка
        # предпочитают dev-токен, если он задан, и работали бы с тестовым ботом.
        "admin_username": os.environ["ADMIN_USERNAME"],
        "admin_password_hash": os.environ["ADMIN_PASSWORD_HASH"],
        "admin_session_secret": os.environ["ADMIN_SESSION_SECRET"].strip(),
        "admin_id": os.environ["ADMIN_ID"],
        # Доступ к админке только по SSH-туннелю (127.0.0.1) → HTTPS нет → cookie без Secure.
        "cookie_secure": os.getenv("COOKIE_SECURE", "false"),
        "admin_tz": os.getenv("ADMIN_TZ", "Europe/Moscow"),
        # --- Лимит трат ---
        "user_spend_limit_usd": os.getenv("USER_SPEND_LIMIT_USD", "1.0"),
        "new_user_spend_limit_usd": os.getenv("NEW_USER_SPEND_LIMIT_USD", "1.0"),
        "spend_limit_exempt_usernames": os.getenv("SPEND_LIMIT_EXEMPT_USERNAMES", ""),
        "support_contact_url": os.environ["SUPPORT_CONTACT_URL"],
        "daily_spend_alert_usd": os.getenv("DAILY_SPEND_ALERT_USD", "100"),
        "sentry_dsn": os.getenv("SENTRY_DSN", ""),
    }


def run(ssh: paramiko.SSHClient, command: str, check: bool = True) -> tuple[str, str, int]:
    """Run remote command and STREAM stdout in real time (line-by-line)."""
    short = command if len(command) <= 100 else command[:100] + "..."
    print(f">>> {short}", flush=True)
    _, stdout, stderr = ssh.exec_command(command)
    # Стримим stdout пока команда работает (важно для долгих apt/uv sync).
    out_lines: list[str] = []
    for raw in iter(stdout.readline, ""):
        line = raw.rstrip("\r\n")
        if line:
            print(f"  {line}", flush=True)
            out_lines.append(line)
    exit_code = stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if exit_code != 0 and err:
        print(f"STDERR: {err}", flush=True)
    if check and exit_code != 0:
        print(f"!!! exit_code={exit_code}", flush=True)
        if not input("Continue? [y/N]: ").lower().startswith("y"):
            sys.exit(1)
    return "\n".join(out_lines), err, exit_code


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    """mkdir -p через SFTP."""
    parts = [p for p in remote_path.split("/") if p]
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def should_skip(path: Path) -> bool:
    if path.name in SKIP_FILE_NAMES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix in SKIP_FILE_SUFFIXES:
        return True
    return False


def upload_tree(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> int:
    """Рекурсивно копируем local_root -> remote_root с прогрессом."""
    # Собираем список файлов для аплоада (нужно знать total для прогресса)
    files: list[Path] = []
    for item in sorted(local_root.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(local_root)
        if should_skip(rel):
            continue
        files.append(item)
    total = len(files)
    print(f"  uploading {total} files...", flush=True)
    for i, item in enumerate(files, start=1):
        rel = item.relative_to(local_root)
        remote_path = posixpath.join(remote_root, str(rel).replace("\\", "/"))
        sftp_mkdirs(sftp, posixpath.dirname(remote_path))
        sftp.put(str(item), remote_path)
        if item.suffix in {".sh", ".py"} and rel.parts and rel.parts[0] == "deploy":
            sftp.chmod(remote_path, 0o755)
        # Печатаем каждые 10 файлов + первый и последний
        if i == 1 or i == total or i % 10 == 0:
            print(f"  [{i}/{total}] {rel}", flush=True)
    return total


def build_prod_env(env: dict[str, str]) -> str:
    """Содержимое /opt/dream-bot/.env на VPS."""
    lines = [
        "# Auto-generated by deploy/deploy.py — не редактируй вручную.",
        "",
        "# Telegram",
        f"BOT_TOKEN={env['bot_token']}",
        "",
        "# Postgres (локальный на VPS)",
        f"DATABASE_URL=postgresql+asyncpg://dream_bot:{env['db_password']}@127.0.0.1:5432/dream_bot",
        "",
        f"LOG_LEVEL={env['log_level']}",
        "",
        "# Webhook (выключен — polling)",
        "WEBHOOK_ENABLED=false",
        "WEBHOOK_BASE_URL=",
        "",
        "# OpenAI (Whisper)",
        f"OPENAI_API_KEY={env['openai_api_key']}",
        "",
        f"LLM_PROXY={env['llm_proxy']}",
        "",
        "# OpenRouter",
        f"OPENROUTER_API_KEY={env['openrouter_api_key']}",
        f"OPENROUTER_BASE_URL={env['openrouter_base_url']}",
        f"OPENROUTER_GENERATOR_MODEL={env['openrouter_generator_model']}",
        f"OPENROUTER_CLASSIFIER_MODEL={env['openrouter_classifier_model']}",
        f"OPENROUTER_INTENT_MODEL={env['openrouter_intent_model']}",
        f"OPENROUTER_CONTEXT_MODEL={env['openrouter_context_model']}",
        "",
        "# Долговременная память о юзере: запись единиц, подстановка в промпты,",
        "# уплотнение («сухой остаток») и лимит блока, по которому оно запускается.",
        f"MEMORY_WRITE_ENABLED={env['memory_write_enabled']}",
        f"MEMORY_INJECT_ENABLED={env['memory_inject_enabled']}",
        f"MEMORY_COMPACT_ENABLED={env['memory_compact_enabled']}",
        f"MEMORY_COMPACT_BUDGET_CHARS={env['memory_compact_budget_chars']}",
        "",
        "# Admin panel (сервис dream-admin, доступ по SSH-туннелю 127.0.0.1:8081)",
        f"ADMIN_USERNAME={env['admin_username']}",
        f"ADMIN_PASSWORD_HASH={env['admin_password_hash']}",
        f"ADMIN_SESSION_SECRET={env['admin_session_secret']}",
        f"ADMIN_ID={env['admin_id']}",
        f"COOKIE_SECURE={env['cookie_secure']}",
        f"ADMIN_TZ={env['admin_tz']}",
        "ADMIN_MEDIA_DIR=/opt/dream-bot/media",
        "",
        "# Spend limit (per-user neural-cost cap)",
        f"USER_SPEND_LIMIT_USD={env['user_spend_limit_usd']}",
        f"NEW_USER_SPEND_LIMIT_USD={env['new_user_spend_limit_usd']}",
        f"SPEND_LIMIT_EXEMPT_USERNAMES={env['spend_limit_exempt_usernames']}",
        f"DAILY_SPEND_ALERT_USD={env['daily_spend_alert_usd']}",
        f"SUPPORT_CONTACT_URL={env['support_contact_url']}",
        "",
    ]
    # Sentry — только если задан: пустая строка включила бы инициализацию с пустым DSN.
    if env.get("sentry_dsn"):
        lines += [f"SENTRY_DSN={env['sentry_dsn']}", ""]
    return "\n".join(lines)


# ====== шаги ======


def step_apt_install(ssh: paramiko.SSHClient) -> None:
    print("\n[1/11] apt update + базовые пакеты")
    run(ssh, "DEBIAN_FRONTEND=noninteractive apt-get update -y", check=False)
    run(
        ssh,
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "postgresql postgresql-contrib postgresql-client "
        "curl ca-certificates git build-essential "
        "libffi-dev libssl-dev",
    )


def step_install_uv(ssh: paramiko.SSHClient) -> None:
    print("\n[2/11] uv -> /usr/local/bin/uv")
    _, _, code = run(ssh, "test -x /usr/local/bin/uv && /usr/local/bin/uv --version", check=False)
    if code == 0:
        print("uv уже установлен")
        return
    # standalone tarball — без зависимости от pip
    run(
        ssh,
        "curl -L https://github.com/astral-sh/uv/releases/latest/download/"
        "uv-x86_64-unknown-linux-gnu.tar.gz -o /tmp/uv.tar.gz",
    )
    run(ssh, "tar -xzf /tmp/uv.tar.gz -C /tmp")
    run(ssh, "install -m 755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv")
    run(ssh, "install -m 755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx")
    run(ssh, "rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu")
    run(ssh, "/usr/local/bin/uv --version")


def step_create_user(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[3/11] юзер dream-bot + /opt/dream-bot")
    run(ssh, f"id dream-bot >/dev/null 2>&1 || useradd -r -m -d {project_dir} -s /bin/bash dream-bot")
    run(ssh, f"mkdir -p {project_dir}")
    run(ssh, f"chown -R dream-bot:dream-bot {project_dir}")


def step_setup_pg(ssh: paramiko.SSHClient, db_password: str) -> None:
    print("\n[4/11] PostgreSQL: юзер dream_bot + БД dream_bot")
    run(ssh, "systemctl enable --now postgresql")
    # Создаём роль если её нет; пароль обновляем каждый раз — на случай ротации
    run(
        ssh,
        f"sudo -u postgres psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='dream_bot'\" "
        f"| grep -q 1 || sudo -u postgres psql -c \"CREATE ROLE dream_bot LOGIN PASSWORD '{db_password}'\"",
    )
    run(ssh, f"sudo -u postgres psql -c \"ALTER ROLE dream_bot WITH PASSWORD '{db_password}'\"")
    run(
        ssh,
        "sudo -u postgres psql -tAc \"SELECT 1 FROM pg_database WHERE datname='dream_bot'\" "
        "| grep -q 1 || sudo -u postgres psql -c \"CREATE DATABASE dream_bot OWNER dream_bot\"",
    )
    run(ssh, "sudo -u postgres psql -d dream_bot -c \"GRANT ALL ON SCHEMA public TO dream_bot\"")


def step_upload_code(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[5/11] SFTP upload кода")
    for d in CODE_DIRS:
        run(ssh, f"rm -rf {project_dir}/{d}")
    sftp = ssh.open_sftp()
    try:
        sftp_mkdirs(sftp, project_dir)
        n = upload_tree(sftp, ROOT, project_dir)
        print(f"загружено {n} файлов в {project_dir}")
    finally:
        sftp.close()
    run(ssh, f"chown -R dream-bot:dream-bot {project_dir}")


def step_write_env(ssh: paramiko.SSHClient, env: dict[str, str], project_dir: str) -> None:
    print("\n[6/11] /opt/dream-bot/.env")
    content = build_prod_env(env)
    # Пишем через cat <<EOF — heredoc сохранит multiline и спецсимволы
    # Экранируем одинарные кавычки на случай (но heredoc с 'EOF' без $-expansion безопасен)
    safe = content.replace("'", "'\"'\"'")
    run(
        ssh,
        f"cat > {project_dir}/.env << 'DEPLOY_ENV_EOF'\n{safe}\nDEPLOY_ENV_EOF",
    )
    run(ssh, f"chown dream-bot:dream-bot {project_dir}/.env")
    run(ssh, f"chmod 600 {project_dir}/.env")


def step_uv_sync_and_migrate(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[7/11] uv sync (Python 3.12 + deps) + alembic upgrade")
    # HOME=$project_dir — чтобы uv писал кеш в /opt/dream-bot/.local
    run(
        ssh,
        f"cd {project_dir} && sudo -u dream-bot HOME={project_dir} "
        f"/usr/local/bin/uv sync --frozen",
    )
    run(
        ssh,
        f"cd {project_dir} && sudo -u dream-bot HOME={project_dir} "
        f"/usr/local/bin/uv run alembic upgrade head",
    )


def step_systemd(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[8/11] systemd unit")
    run(ssh, f"cp {project_dir}/deploy/dream-bot.service /etc/systemd/system/dream-bot.service")
    run(ssh, "systemctl daemon-reload")
    run(ssh, "systemctl enable dream-bot")


def step_systemd_admin(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[8b] systemd unit — dream-admin (:8081, SSH-туннель)")
    # media-каталог админки (кэш аватаров, ADMIN_MEDIA_DIR)
    run(ssh, f"mkdir -p {project_dir}/media && chown dream-bot:dream-bot {project_dir}/media")
    run(ssh, f"cp {project_dir}/deploy/dream-admin.service /etc/systemd/system/dream-admin.service")
    run(ssh, "systemctl daemon-reload")
    run(ssh, "systemctl enable dream-admin")


def step_journald_rotation(ssh: paramiko.SSHClient) -> None:
    """Лимит на размер persistent journald (иначе сожрёт ГБ за месяцы)."""
    print("\n[9/11] journald rotation (SystemMaxUse=500M)")
    drop_in = (
        "[Journal]\n"
        "SystemMaxUse=500M\n"
        "SystemKeepFree=200M\n"
        "MaxRetentionSec=2month\n"
    )
    run(ssh, "mkdir -p /etc/systemd/journald.conf.d")
    run(
        ssh,
        f"cat > /etc/systemd/journald.conf.d/dream-bot.conf << 'JOURNALD_EOF'\n{drop_in}JOURNALD_EOF",
    )
    run(ssh, "systemctl restart systemd-journald")


def step_cron_backup(ssh: paramiko.SSHClient, project_dir: str) -> None:
    print("\n[10/11] cron daily backup (04:00)")
    run(ssh, f"chmod +x {project_dir}/deploy/backup_db.sh")
    run(ssh, f"mkdir -p {project_dir}/backups && chown dream-bot:dream-bot {project_dir}/backups")
    cron_line = f"0 4 * * * {project_dir}/deploy/backup_db.sh >> /var/log/dream-bot-backup.log 2>&1"
    # idempotent: убираем старую строку и добавляем новую
    run(
        ssh,
        f"(sudo -u dream-bot crontab -l 2>/dev/null | grep -v backup_db.sh; echo '{cron_line}') "
        f"| sudo -u dream-bot crontab -",
    )
    # touch log file with rw perms
    run(ssh, "touch /var/log/dream-bot-backup.log && chown dream-bot:dream-bot /var/log/dream-bot-backup.log")


def step_restart(ssh: paramiko.SSHClient) -> None:
    print("\n[11/11] systemctl restart + status (bot + admin)")
    run(ssh, "systemctl restart dream-bot")
    run(ssh, "systemctl restart dream-admin")
    run(ssh, "sleep 3 && systemctl --no-pager status dream-bot", check=False)
    run(ssh, "systemctl --no-pager status dream-admin", check=False)


def main() -> None:
    print("=" * 60)
    print("IfsBot — deploy")
    print("=" * 60)
    env = load_env()
    print(f"\nConnecting to {env['user']}@{env['host']}:{env['port']}...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    auth = (
        {"key_filename": os.path.expanduser(env["ssh_key"])}
        if env["ssh_key"]
        else {"password": env["password"]}
    )
    ssh.connect(
        hostname=env["host"],
        port=env["port"],
        username=env["user"],
        timeout=30,
        look_for_keys=False,
        allow_agent=False,
        **auth,
    )
    print("Connected.")

    try:
        step_apt_install(ssh)
        step_install_uv(ssh)
        step_create_user(ssh, env["project_dir"])
        step_setup_pg(ssh, env["db_password"])
        step_upload_code(ssh, env["project_dir"])
        step_write_env(ssh, env, env["project_dir"])
        step_uv_sync_and_migrate(ssh, env["project_dir"])
        step_systemd(ssh, env["project_dir"])
        step_systemd_admin(ssh, env["project_dir"])
        step_journald_rotation(ssh)
        step_cron_backup(ssh, env["project_dir"])
        step_restart(ssh)
    finally:
        ssh.close()

    print("\n" + "=" * 60)
    print("DEPLOY DONE")
    print("=" * 60)
    print(f"  ssh {env['user']}@{env['host']}")
    print("  journalctl -u dream-bot -f")
    print("  systemctl status dream-bot")


if __name__ == "__main__":
    main()
