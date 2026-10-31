"""/profile — карточка профиля + редактирование (life_context, experience).

Имя read-only — синхронизируется с Telegram first_name на каждом /start.
"""

from __future__ import annotations

from html import escape as html_escape

import structlog
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    auto_pause_active_quick,
    discard_pending_newdream,
)
from adapters.telegram.keyboards import (
    LIFE_CONTEXT_APPEND_CALLBACK,
    life_context_edit_mode_keyboard,
    profile_edit_keyboard,
    remove_keyboard,
)
from adapters.telegram.states import ProfileEditStates
from adapters.telegram.texts import (
    ASK_LIFE_CONTEXT_EDIT_MODE,
    ASK_NEW_LIFE_CONTEXT,
    LIFE_CONTEXT_MODE_NUDGE,
    NOT_ONBOARDED_PROMPT,
    ONBOARDING_IN_PROGRESS_NOTICE,
    PROFILE_LIFE_CONTEXT_UPDATED,
    PROFILE_VIEW_TEMPLATE,
)
from adapters.telegram.voice_helper import transcribe_with_placeholder
from app.config import Settings
from data.repositories import UsageEventRepository, UserRepository
from domain.entry_types import entry_config
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)


async def handle_profile(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    """/profile: показать карточку с inline-кнопками редактирования."""
    if message.from_user is None:
        return

    # Переключение из режима рассказа сна → выкидываем недосказанный сон.
    await discard_pending_newdream(
        state=state, session=session, telegram_user_id=message.from_user.id
    )
    # …или из открытой карточки сна — сбрасываем её state.
    if (await state.get_state() or "").startswith("DreamCardStates"):
        await state.clear()

    repo = UserRepository(session)
    user = await repo.get_by_tg_id(message.from_user.id)
    if user is None:
        await message.answer(NOT_ONBOARDED_PROMPT)
        return
    if not UserRepository.is_onboarded(user):
        # Если юзер ПОСРЕДИ онбординга (есть активный FSM-state) — не советуем /start
        # (он сбросит прогресс); мягко возвращаем к знакомству. Совсем новый (state=None)
        # — стандартный NOT_ONBOARDED с /start.
        if await state.get_state() is not None:
            await message.answer(ONBOARDING_IN_PROGRESS_NOTICE)
        else:
            await message.answer(NOT_ONBOARDED_PROMPT)
        return

    paused = await auto_pause_active_quick(
        session=session, state=state, user_id=user.id,
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )
    if paused is not None:
        await message.answer(
            entry_config(paused.kind).auto_paused_notice, reply_markup=remove_keyboard()
        )

    await message.answer(
        PROFILE_VIEW_TEMPLATE.format(
            name=html_escape(user.display_name or "–"),
            life_context=html_escape(user.profile_life_context or "–"),
        ),
        reply_markup=profile_edit_keyboard(),
    )


async def handle_profile_edit_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    """profile:edit:<field> — переход в соответствующий мини-FSM.

    Spec §4.2: навигационные callback'и profile:edit:* паузят активный Quick.
    """
    if callback.data is None or callback.message is None:
        return
    await safe_answer(callback)

    if callback.from_user is not None:
        repo = UserRepository(session)
        user = await repo.get_by_tg_id(callback.from_user.id)
        if user is not None:
            paused = await auto_pause_active_quick(
                session=session, state=state, user_id=user.id,
                settings=settings,
                openrouter_classifier_client=openrouter_classifier_client,
            )
            if paused is not None:
                await callback.message.answer(
                    entry_config(paused.kind).auto_paused_notice,
                    reply_markup=remove_keyboard(),
                )

    parts = callback.data.split(":", 2)
    if len(parts) != 3:
        return
    field = parts[2]
    if field == "life_context":
        # L6: сперва выбор режима (заменить/дописать), ввод текста — после выбора.
        # Sub-state обязателен: без него стрей-текст до тапа по кнопке провалился бы
        # в idle catch-all и молча потерялся.
        await state.set_state(ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE)
        await callback.message.answer(
            ASK_LIFE_CONTEXT_EDIT_MODE, reply_markup=life_context_edit_mode_keyboard()
        )


async def handle_life_context_mode_callback(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    """profile:life:<mode> — режим правки выбран, переходим к вводу текста."""
    if callback.data is None or callback.message is None:
        return
    await safe_answer(callback)

    mode = "append" if callback.data == LIFE_CONTEXT_APPEND_CALLBACK else "replace"
    await state.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)
    await state.update_data(life_context_edit_mode=mode)
    await callback.message.answer(ASK_NEW_LIFE_CONTEXT, reply_markup=remove_keyboard())


async def handle_life_context_mode_nudge(
    message: Message,
    state: FSMContext,
) -> None:
    """AWAITING_LIFE_CONTEXT_MODE (text/voice): режим ещё не выбран.

    Переспрашиваем с той же клавиатурой; присланный текст НЕ применяем
    (неизвестно, заменить или дописать) — юзер повторит его после выбора.
    """
    await message.answer(
        LIFE_CONTEXT_MODE_NUDGE, reply_markup=life_context_edit_mode_keyboard()
    )


async def _save_life_context(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    text: str,
) -> None:
    """Применить новое значение life_context с учётом выбранного режима.

    append при непустом существующем — дописываем через \\n\\n (как в онбординге
    LIFE_CONTEXT_LOOP); иначе (replace или пустой existing) — обычная запись.
    """
    if message.from_user is None:
        return
    repo = UserRepository(session)
    user = await repo.get_by_tg_id(message.from_user.id)
    if user is None:
        return
    data = await state.get_data()
    mode = data.get("life_context_edit_mode", "replace")
    existing = (user.profile_life_context or "").strip()
    if mode == "append" and existing:
        new_value = f"{existing}\n\n{text.strip()}"
    else:
        new_value = text.strip()
    await repo.update_profile(user_id=user.id, profile_life_context=new_value)
    await state.clear()
    await message.answer(PROFILE_LIFE_CONTEXT_UPDATED, reply_markup=remove_keyboard())


async def _transcribe_profile_voice(
    message: Message,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> str | None:
    """Помощник: расшифровать voice + записать usage. None при ошибке."""
    if message.from_user is None or message.voice is None:
        return None
    # stt_user_lock захватывается на ВЕСЬ handler в BusyGuardMiddleware.
    result = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client, log_prefix="profile",
        session=session, settings=settings,
    )
    if result is None:
        return None
    repo = UserRepository(session)
    user = await repo.get_by_tg_id(message.from_user.id)
    if user is None:
        return result.text
    recorder = UsageRecorder(UsageEventRepository(session))
    await recorder.record_whisper(user_id=user.id, audio_seconds=result.duration_seconds)
    return result.text


async def handle_edit_life_context(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
) -> None:
    """EDITING_LIFE_CONTEXT (text): применяем текст с учётом режима."""
    if message.from_user is None or message.text is None:
        return
    await _save_life_context(message, state, session, message.text)


async def handle_edit_life_context_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """EDITING_LIFE_CONTEXT (voice): расшифровать и применить с учётом режима."""
    text = await _transcribe_profile_voice(message, session, bot, openai_client, settings)
    if text is None:
        return
    await _save_life_context(message, state, session, text)


def create_router() -> Router:
    """Создать роутер с зарегистрированными хэндлерами /profile."""
    router = Router(name="profile_handler")
    router.message.register(handle_profile, Command("profile"))
    router.callback_query.register(
        handle_profile_edit_callback,
        F.data.startswith("profile:edit:"),
    )
    router.callback_query.register(
        handle_life_context_mode_callback,
        F.data.startswith("profile:life:"),
    )
    router.message.register(
        handle_life_context_mode_nudge,
        ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE,
        F.text | F.voice,
    )
    router.message.register(
        handle_edit_life_context, ProfileEditStates.EDITING_LIFE_CONTEXT, F.text
    )
    router.message.register(
        handle_edit_life_context_voice, ProfileEditStates.EDITING_LIFE_CONTEXT, F.voice
    )
    return router
