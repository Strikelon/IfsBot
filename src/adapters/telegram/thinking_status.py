"""Ротирующийся «думаю…»-плейсхолдер для долгих LLM-вызовов.

Pro-генерация занимает 10-30 секунд. Статичное «Думаю…» оставляет юзера в
неведении — живой ли бот вообще. Этот хелпер:
- Шлёт одно сообщение с одной из 10 живых фраз (THINKING_STATUSES).
- Запускает фоновую задачу, которая каждые 5 секунд редактирует это сообщение
  на следующую фразу из списка (циклично; стартовая позиция случайная, чтобы
  два соседних запроса не выглядели одинаково).
- При выходе из контекста (успех или исключение) задача отменяется, плейсхолдер
  удаляется. Финальный ответ бота отправляется отдельным сообщением caller'ом.

Любые ошибки при edit_text/delete тихо проглатываются — это UX-сигнал, а не
часть бизнес-логики, и не должен ломать поток.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State
from aiogram.types import (
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from adapters.telegram.texts import THINKING_STATUSES

log = structlog.get_logger(__name__)

_ROTATION_INTERVAL_SECONDS = 5.0


@asynccontextmanager
async def thinking_status(
    message: Message,
    *,
    reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup | ReplyKeyboardRemove | None = None,
    state: FSMContext | None = None,
    thinking_state: State | None = None,
    statuses: list[str] | None = None,
) -> AsyncIterator[Message]:
    """Async-context: показать ротирующийся плейсхолдер, по выходу — удалить.

    Использовать вокруг ЛЮБОГО долгого LLM-вызова:

        async with thinking_status(message):
            result = await long_llm_call()
        await message.answer(result.text)

    Если LLM упал — контекст всё равно очистится. Плейсхолдер удаляется на
    выходе; ответ бота шлёт caller отдельным `message.answer(...)`.

    `reply_markup` — передаётся при первой отправке (например, чтобы убрать
    предыдущую reply-клаву через ReplyKeyboardRemove). edit_text её не
    переустановит, и это нормально: ReplyKeyboardRemove однажды отправленный
    действует «глобально» в чате до следующего ReplyKeyboardMarkup.

    `state` + `thinking_state` — опциональная пара. Если оба переданы, на входе
    FSM-state swap'ится на thinking_state, на выходе — восстанавливается prev_state
    (даже при exception в LLM). Если только один — оба игнорируются.

    `statuses` — кастомный список фраз вместо общего (dream-themed) THINKING_STATUSES.
    Нужен для шага знакомства в онбординге, где снов ещё нет (PROFILE_THINKING_STATUSES).
    """
    rotation = statuses if statuses else THINKING_STATUSES
    prev_state: str | None = None
    state_swap_active = False  # True только если swap-in реально удался
    if state is not None and thinking_state is not None:
        prev_state = await state.get_state()
        if prev_state == thinking_state.state:
            # Уже в нужном состоянии — другой concurrent task сделал set_state раньше.
            # Не пишем и не restore'им — оставляем как есть.
            log.info(
                "thinking_status.state_already_active",
                current_state=prev_state,
            )
        else:
            await state.set_state(thinking_state)
            state_swap_active = True
            log.info(
                "thinking_status.state_swapped_in",
                prev_state=prev_state,
                thinking_state=thinking_state.state,
            )

    try:
        start_idx = random.randrange(len(rotation))
        try:
            placeholder = await message.answer(
                rotation[start_idx], reply_markup=reply_markup
            )
        except Exception as exc:  # noqa: BLE001
            # Если даже первоначальный answer не прошёл — просто отдаём контекст
            # дальше без ротации; caller всё равно пришлёт результат.
            log.info(
                "thinking_status.initial_send_failed",
                exc_type=type(exc).__name__,
                exc=str(exc),
            )
            yield message  # type: ignore[misc]
            return  # внешний finally восстановит prev_state

        async def _rotate() -> None:
            idx = start_idx
            while True:
                try:
                    await asyncio.sleep(_ROTATION_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
                idx = (idx + 1) % len(rotation)
                try:
                    await placeholder.edit_text(rotation[idx])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # Сообщение могли удалить, Telegram мог отдать «not modified»,
                    # сеть могла моргнуть — всё ок, ждём дальше до отмены. Лог на
                    # INFO (не DEBUG) — чтобы в будущем не диагностировать слепо.
                    log.info(
                        "thinking_status.edit_failed",
                        idx=idx,
                        exc_type=type(exc).__name__,
                        exc=str(exc),
                    )

        task = asyncio.create_task(_rotate())
        try:
            yield placeholder
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            try:
                await placeholder.delete()
            except Exception as exc:  # noqa: BLE001
                log.info(
                    "thinking_status.delete_failed",
                    exc_type=type(exc).__name__,
                    exc=str(exc),
                )
    finally:
        # ВНЕШНИЙ finally: восстанавливает prev_state только если swap-in реально
        # удался. Это защищает от ложного set_state(None) если get_state() ранее
        # бросил исключение и prev_state остался None — иначе мы бы случайно
        # очистили FSM.
        if state_swap_active and state is not None:
            try:
                await state.set_state(prev_state)
                log.info(
                    "thinking_status.state_swapped_out",
                    restored_state=prev_state,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "thinking_status.state_restore_failed",
                    exc_type=type(exc).__name__,
                    exc=str(exc),
                )
