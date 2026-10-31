"""Явный таймаут Telegram-сессии вместо библиотечной минуты.

Долг из инцидента 18–19.08.2026: сессия создавалась без `timeout=`, дефолт
aiogram — 60 секунд. Когда запрос подвисал, человек сидел минуту перед мёртвым
экраном, жал кнопку второй раз — и получал второй платный вызов.

Long-polling это НЕ ломает: для getUpdates aiogram сам просит
`request_timeout = session.timeout + polling_timeout` (dispatcher.py:216),
то есть ждёт дольше, чем длится сам long-poll. Скачивание файлов тоже не
задето — у `stream_content` свой таймаут (aiohttp.py:189), голосовые грузятся
как грузились.
"""

from pydantic import SecretStr

from app.bot import TELEGRAM_REQUEST_TIMEOUT_SEC, create_bot

_TOKEN = SecretStr("123456789:AAEEd8H6ZzY6Y6Y6Y6Y6Y6Y6Y6Y6Y6Y6Y6")


def test_session_timeout_is_explicit_and_short():
    bot = create_bot(_TOKEN)
    assert bot.session.timeout == TELEGRAM_REQUEST_TIMEOUT_SEC
    assert TELEGRAM_REQUEST_TIMEOUT_SEC <= 20, "минута ожидания — это и был дефект"


def test_session_timeout_same_with_proxy():
    """Прод ходит через прокси — именно там таймаут и важен."""
    bot = create_bot(_TOKEN, proxy=SecretStr("http://127.0.0.1:8888"))
    assert bot.session.timeout == TELEGRAM_REQUEST_TIMEOUT_SEC


def test_timeout_leaves_room_for_long_polling():
    """Гард: getUpdates ждёт session.timeout + polling_timeout, а не наоборот.

    Если однажды таймаут опустят до секунд, long-polling начнёт рвать
    соединение раньше, чем Telegram успеет отдать апдейт.
    """
    polling_timeout = 10  # дефолт dp.start_polling
    assert TELEGRAM_REQUEST_TIMEOUT_SEC + polling_timeout > polling_timeout
    assert TELEGRAM_REQUEST_TIMEOUT_SEC >= 10
