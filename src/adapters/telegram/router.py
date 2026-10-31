from aiogram import Router

from adapters.telegram.handlers import chat_member as chat_member_handler
from adapters.telegram.handlers import dreams as dreams_handler
from adapters.telegram.handlers import entry_choice as entry_choice_handler
from adapters.telegram.handlers import entry_flow as entry_flow_handler
from adapters.telegram.handlers import idle as idle_handler
from adapters.telegram.handlers import image_questions as image_questions_handler
from adapters.telegram.handlers import interview as interview_handler
from adapters.telegram.handlers import media_guard as media_guard_handler
from adapters.telegram.handlers import newdream as newdream_handler
from adapters.telegram.handlers import onboarding as onboarding_handler
from adapters.telegram.handlers import profile as profile_handler
from adapters.telegram.handlers import quick as quick_handler
from adapters.telegram.handlers import start as start_handler
from adapters.telegram.handlers import thinking_guard as thinking_guard_handler


def get_router() -> Router:
    """Собирает все Telegram-хэндлеры в один Router.

    Порядок:
    1. thinking_guard — FIRST, чтобы перехватить text/voice во время LLM-вызовов
       до того, как другие state-handler'ы попытаются обработать сообщение.
    1.6. interview — команда владельца /interview: тоже в любом состоянии.
    2. start, onboarding, newdream, quick, dreams, profile — обычные state-handlers.
    3. media_guard — ловит неподдерживаемые типы (фото/файлы/стикеры/видео) в
       ЛЮБОМ состоянии: подсказка «умею только текст/голос», state не трогаем.
    4. idle — LAST, catch-all для text/voice, что не подошло.

    Новый сон записывается только через /newdream.
    """
    router = Router(name="telegram_root")
    router.include_router(thinking_guard_handler.create_router())
    router.include_router(interview_handler.create_router())
    router.include_router(start_handler.create_router())
    router.include_router(entry_choice_handler.create_router())
    router.include_router(chat_member_handler.create_router())
    router.include_router(onboarding_handler.create_router())
    router.include_router(entry_flow_handler.create_router())
    router.include_router(image_questions_handler.create_router())
    router.include_router(newdream_handler.create_router())
    router.include_router(quick_handler.create_router())
    router.include_router(dreams_handler.create_router())
    router.include_router(profile_handler.create_router())
    router.include_router(media_guard_handler.create_router())
    router.include_router(idle_handler.create_router())
    return router
