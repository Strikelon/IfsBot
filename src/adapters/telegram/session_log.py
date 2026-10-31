"""Persistence реплик Quick-сессии в таблице session_messages.

Deep с самого начала пишет каждую реплику в session_messages через локальные
_log_bot_message/_log_user_message в deep.py. До quick.v2.1 Quick этого не делал —
qa_history жил только в FSM (RAM), и пропадал при /newdream → state.clear() или
рестарте бота. Эти функции — публичный аналог для Quick.

phase_label-конвенция Quick:
- "initial"  — первый IFS-разбор (одна запись на сессию)
- "qa"       — каждый follow-up ответ модели
- "image_q"  — пре-initial вопрос по образу сна (фича image questions)
- "pre_q"    — пре-initial уточняющий вопрос новых типов (ситуация/конфликт)
Для USER-сообщений phase_label не пишем (direction уже различает).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data.constants import (
    QUICK_PHASE_IMAGE_Q,
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_PRE_Q,
    QUICK_PHASE_QA,
)
from data.models import (
    LlmRole,
    Session,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
    User,
)
from data.models import (
    Session as SessionModel,
)
from data.repositories import SessionMessageRepository

# Re-export so existing callers (quick.py, tests, etc.) keep working unchanged.
__all__ = [
    "QUICK_PHASE_IMAGE_Q",
    "QUICK_PHASE_INITIAL",
    "QUICK_PHASE_PRE_Q",
    "QUICK_PHASE_QA",
    "log_bot_message",
    "log_user_message",
    "load_quick_qa_history",
    "load_dream_qa_for_display",
]


async def log_bot_message(
    session_db: AsyncSession,
    *,
    session_id: int,
    content: str,
    phase_label: str | None = None,
    llm_role: LlmRole | None = LlmRole.GENERATOR,
) -> None:
    """Записать реплику бота в session_messages.

    Используется и Quick, и Deep — в Deep `phase_label` обычно строка типа
    "P2_FOCUS"/"P3_FEEL"/..., в Quick — "initial"/"qa".
    """
    repo = SessionMessageRepository(session_db)
    await repo.append(
        session_id=session_id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content=content,
        llm_role_used=llm_role,
        llm_phase_label=phase_label,
    )
    if phase_label == QUICK_PHASE_QA:
        sess = await session_db.get(Session, session_id)
        if sess is not None:
            user = await session_db.get(User, sess.user_id)
            if user is not None:
                user.qa_answers_total = (user.qa_answers_total or 0) + 1


async def log_user_message(
    session_db: AsyncSession,
    *,
    session_id: int,
    content: str,
    content_type: SessionMessageContentType = SessionMessageContentType.TEXT,
    audio_file_id: str | None = None,
) -> None:
    """Записать реплику юзера в session_messages."""
    repo = SessionMessageRepository(session_db)
    await repo.append(
        session_id=session_id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=content_type,
        content=content,
        audio_file_id=audio_file_id,
    )


async def load_quick_qa_history(
    session_db: AsyncSession, *, session_id: int
) -> tuple[str | None, list[tuple[str, str]]]:
    """Прочитать историю Quick-сессии из session_messages.

    Returns:
        (initial_text, qa_pairs) где:
        - initial_text — первый IFS-разбор (phase_label="initial"), None если не сделан.
        - qa_pairs — список (user_question, bot_answer) follow-up'ов в хронологическом
          порядке. Если у последнего пользовательского вопроса ещё нет ответа бота —
          он в результат не попадает (значит turn ещё в процессе).

    Сообщения до первого initial игнорируются (пре-initial фаза image_q уходит
    в память другим путём — user_replies_with_prompts).
    """
    result = await session_db.execute(
        select(SessionMessage)
        .where(SessionMessage.session_id == session_id)
        .order_by(SessionMessage.created_at.asc())
    )
    msgs = list(result.scalars().all())

    initial_text: str | None = None
    qa_pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    initial_seen = False

    for msg in msgs:
        if msg.direction == SessionMessageDirection.BOT_TO_USER:
            # Пре-initial вопросы по образам — не Q&A; фильтруем всегда
            # (защита и от будущих пре-initial фаз с этим лейблом).
            if msg.llm_phase_label in (QUICK_PHASE_IMAGE_Q, QUICK_PHASE_PRE_Q):
                continue
            if not initial_seen:
                if msg.llm_phase_label == QUICK_PHASE_INITIAL:
                    initial_text = msg.content
                    initial_seen = True
                continue
            if pending_user is not None:
                qa_pairs.append((pending_user, msg.content))
                pending_user = None
        else:  # USER_TO_BOT
            if not initial_seen:
                # Ответы юзера ДО initial (вопросы по образам) — не Q&A-пары.
                continue
            pending_user = msg.content

    return initial_text, qa_pairs


async def load_dream_qa_for_display(
    session_db: AsyncSession, *, dream_id: int
) -> tuple[str, list[tuple[str, str]]] | None:
    """Подгрузить последний initial-разбор + Q&A пары для показа на /dream_<id>.

    Возвращает (initial_text, qa_pairs) из latest PAUSED-или-COMPLETED Quick
    session для этого сна. None если у сна нет ни одной inactive Quick-сессии
    или ни у одной нет initial-сообщения.

    qa_pairs — список (user_question, bot_answer) в хронологическом порядке.
    "Висячий" user без bot-ответа в результат не попадает (как в load_quick_qa_history).
    """
    stmt = (
        select(SessionModel.id)
        .where(
            SessionModel.dream_id == dream_id,
            SessionModel.type == SessionType.QUICK,
            SessionModel.status.in_([SessionStatus.PAUSED, SessionStatus.COMPLETED]),
        )
        .order_by(SessionModel.last_active_at.desc(), SessionModel.id.desc())
        .limit(1)
    )
    result = await session_db.execute(stmt)
    latest_session_id = result.scalar_one_or_none()
    if latest_session_id is None:
        return None

    initial_text, pairs = await load_quick_qa_history(
        session_db, session_id=latest_session_id
    )
    if initial_text is None:
        return None
    return initial_text, pairs
