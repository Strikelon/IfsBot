"""Узкие CRUD-репозитории. Каждый принимает AsyncSession в конструкторе,
не управляет транзакциями — это ответственность вызывающего (handler или middleware).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from data.constants import (
    GROUNDING_QUESTION_MARKER,
    PRE_QUESTION_MARKER,
    QUICK_PHASE_IMAGE_Q,
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_PRE_Q,
)
from data.models import (
    DISPLAY_NAME_MAX_LEN,
    Dream,
    DreamImage,
    DreamStatus,
    ExperienceLevel,
    LlmRole,
    LlmService,
    LlmUsageEvent,
    P4Substep,
    RequestBranch,
    Session,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionPhase,
    SessionState,
    SessionStatus,
    SessionType,
    UsageMetric,
    User,
    UserMemoryItem,
)


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_tg_id(self, telegram_user_id: int) -> User | None:
        result = await self._session.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        )
        return result.scalar_one_or_none()

    async def search(self, query: str, *, limit: int = 8) -> list[User]:
        """Найти юзеров по имени, нику или telegram id — для команды /interview.

        Ник искать недостаточно: у части людей его нет вовсе (6 из 50 на проде
        2026-08-24), и как раз им чаще всего и нужен ручной доступ.

        Регистр сворачивается в Python, а не через ILIKE: на кириллице ILIKE
        зависит от collation базы (в SQLite тестов не работает вовсе, в Postgres
        под `C` тоже), и «мария» молча не находила бы «Марию». Выборка целиком —
        осознанно: команда админская, редкая, а юзеров десятки.
        """
        needle = query.strip().lstrip("@").casefold()
        if not needle:
            return []
        result = await self._session.execute(
            select(User).order_by(
                User.last_seen_at.desc().nullslast(), User.id.desc()
            )
        )
        matched = [
            u
            for u in result.scalars().all()
            if needle in (u.display_name or "").casefold()
            or needle in (u.username or "").casefold()
            or needle == str(u.telegram_user_id)
        ]
        return matched[:limit]

    async def get_or_create_by_tg_id(
        self, telegram_user_id: int, display_name: str | None = None
    ) -> User:
        existing = await self.get_by_tg_id(telegram_user_id)
        if existing is not None:
            return existing
        user = User(telegram_user_id=telegram_user_id, display_name=display_name)
        try:
            # SAVEPOINT: если параллельный апдейт создаёт того же юзера одновременно,
            # INSERT упадёт UniqueViolation'ом — откатываем только этот вложенный блок
            # (не всю транзакцию) и берём уже созданного юзера.
            async with self._session.begin_nested():
                self._session.add(user)
                await self._session.flush()
        except IntegrityError:
            existing = await self.get_by_tg_id(telegram_user_id)
            if existing is None:
                raise
            return existing
        return user

    async def update_profile(
        self,
        *,
        user_id: int,
        display_name: str | None = None,
        profile_life_context: str | None = None,
        profile_experience_level: ExperienceLevel | None = None,
    ) -> None:
        """Частичное обновление — None означает «не трогать».

        NOTE: передача None не очищает поле. Если в будущем понадобится обнуление,
        добавь явный sentinel-объект (например `UNSET`) или отдельный метод.
        """
        values: dict[str, object] = {}
        if display_name is not None:
            values["display_name"] = display_name
        if profile_life_context is not None:
            values["profile_life_context"] = profile_life_context
        if profile_experience_level is not None:
            values["profile_experience_level"] = profile_experience_level

        if not values:
            return
        await self._session.execute(update(User).where(User.id == user_id).values(**values))

    @staticmethod
    def is_onboarded(user: User) -> bool:
        """Юзер прошёл онбординг, если зафиксирован timestamp завершения.

        Старый критерий (наличие profile_life_context + profile_experience_level)
        больше не работает: с новым flow юзер может иметь профиль, но не
        дойти до конца (Quick force + опц. Deep). Маркер — onboarding_completed_at.
        """
        return user.onboarding_completed_at is not None

    async def update_locale(self, *, user_id: int, locale: str) -> None:
        """Обновить локаль пользователя."""
        user = await self._session.get(User, user_id)
        if user is None:
            return
        user.locale = locale

    async def mark_onboarded(self, *, user_id: int) -> None:
        """Set users.onboarding_completed_at = now() if not already set.

        Called from handle_finish_quick_reply on user's first «Завершить» tap
        when sess.is_onboarding_session is True. Idempotent: re-calls on
        already-onboarded user are no-ops.
        """
        user = await self._session.get(User, user_id)
        if user is None:
            return
        user.onboarding_completed_at = datetime.now(UTC)

    async def touch_activity(
        self, *, user_id: int, username: str | None, now: datetime
    ) -> None:
        """On every incoming update: bump last_seen_at, sync username, ensure active."""
        user = await self._session.get(User, user_id)
        if user is None:
            return
        user.last_seen_at = now
        if username is not None and user.username != username[:64]:
            user.username = username[:64]
        if not user.is_active:
            user.is_active = True
            user.blocked_at = None

    async def set_blocked(self, *, user_id: int, blocked: bool, now: datetime) -> None:
        """my_chat_member / 403 handling: toggle active flag."""
        user = await self._session.get(User, user_id)
        if user is None:
            return
        user.is_active = not blocked
        user.blocked_at = now if blocked else None

    async def update_display_name_if_changed(
        self, *, user_id: int, display_name: str
    ) -> bool:
        """Идемпотентно обновить display_name, если отличается от текущего.

        Используется для auto-sync на /start: берём message.from_user.first_name
        и пишем в БД только при реальном изменении. Возвращает True если был UPDATE,
        чтобы handler мог решить, нужен ли flush.
        """
        user = await self._session.get(User, user_id)
        if user is None:
            return False
        new_value = display_name[:DISPLAY_NAME_MAX_LEN]  # truncate to column constraint
        if user.display_name == new_value:
            return False
        user.display_name = new_value
        return True


class DreamRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: int,
        recorded_at: datetime,
        transcription: str,
        raw_audio_file_id: str | None = None,
        title: str | None = None,
        kind: str = "dream",
    ) -> Dream:
        dream = Dream(
            user_id=user_id,
            recorded_at=recorded_at,
            transcription=transcription,
            raw_audio_file_id=raw_audio_file_id,
            title=title,
            status=DreamStatus.RECORDED,
            kind=kind,
        )
        self._session.add(dream)
        user = await self._session.get(User, user_id)
        if user is not None:
            user.dreams_total = (user.dreams_total or 0) + 1
        await self._session.flush()
        return dream

    async def get_by_id(self, dream_id: int, *, user_id: int) -> Dream | None:
        """Возвращает сон только если он принадлежит запрошенному юзеру (изоляция)."""
        result = await self._session.execute(
            select(Dream).where(Dream.id == dream_id, Dream.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self, *, user_id: int, limit: int = 20, kind: str | None = None
    ) -> list[Dream]:
        stmt = (
            select(Dream)
            .where(Dream.user_id == user_id)
            .order_by(Dream.recorded_at.desc())
            .limit(limit)
        )
        if kind is not None:
            stmt = stmt.where(Dream.kind == kind)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def dream_has_analysis(self, *, dream_id: int) -> bool:
        """True если хоть в одной сессии этого сна есть initial-сообщение бота.

        Используется на карточке /dream_<id>: разбор был → показываем сохранённый
        результат и вход в Q&A; разбор не был → показываем кнопку «Разобрать сейчас».
        """
        stmt = (
            select(SessionMessage.id)
            .join(Session, Session.id == SessionMessage.session_id)
            .where(
                Session.dream_id == dream_id,
                SessionMessage.direction == SessionMessageDirection.BOT_TO_USER,
                SessionMessage.llm_phase_label == QUICK_PHASE_INITIAL,
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def hard_delete_with_cascade(
        self, *, dream_id: int, user_id: int
    ) -> bool:
        """Удаляет сон со всеми связанными сущностями.

        Возвращает True если сон был удалён, False если сон не найден или
        принадлежит другому юзеру.

        Порядок (важен, т.к. SQLite не cascade'ит автоматически):
        1. UPDATE llm_usage_events SET session_id=NULL WHERE session_id IN (sessions of this dream)
        2. DELETE FROM session_messages WHERE session_id IN (...)
        3. DELETE FROM session_state WHERE session_id IN (...)
        4. DELETE FROM dream_images WHERE dream_id=?
        5. DELETE FROM sessions WHERE dream_id=?
        6. DELETE FROM dreams WHERE id=? AND user_id=?
        """
        from sqlalchemy import delete

        # Проверяем принадлежность сна юзеру
        dream = await self.get_by_id(dream_id=dream_id, user_id=user_id)
        if dream is None:
            return False

        # Получаем ID всех сессий этого сна
        session_ids_q = select(Session.id).where(Session.dream_id == dream_id)
        session_ids = list((await self._session.execute(session_ids_q)).scalars().all())

        if session_ids:
            # 1. Зануляем session_id в llm_usage_events (billing-аудит сохраняется)
            await self._session.execute(
                update(LlmUsageEvent)
                .where(LlmUsageEvent.session_id.in_(session_ids))
                .values(session_id=None)
            )
            # 2. Удаляем session_messages
            await self._session.execute(
                delete(SessionMessage).where(SessionMessage.session_id.in_(session_ids))
            )
            # 3. Удаляем session_state
            await self._session.execute(
                delete(SessionState).where(SessionState.session_id.in_(session_ids))
            )

        # 4. Удаляем dream_images
        await self._session.execute(
            delete(DreamImage).where(DreamImage.dream_id == dream_id)
        )
        # 5. Удаляем sessions
        await self._session.execute(
            delete(Session).where(Session.dream_id == dream_id)
        )
        # 6. Удаляем dream
        await self._session.execute(
            delete(Dream).where(Dream.id == dream_id, Dream.user_id == user_id)
        )

        return True


class UsageEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        user_id: int,
        service: LlmService,
        model: str,
        usage_metric: UsageMetric,
        usage_amount: int,
        cost_usd: Decimal,
        session_id: int | None = None,
        prompt_version: str | None = None,
    ) -> LlmUsageEvent:
        event = LlmUsageEvent(
            user_id=user_id,
            session_id=session_id,
            service=service,
            model=model,
            usage_metric=usage_metric,
            usage_amount=usage_amount,
            cost_usd=cost_usd,
            prompt_version=prompt_version,
        )
        self._session.add(event)
        await self._session.flush()
        return event

    async def total_cost_for_user(
        self, user_id: int, *, since: datetime | None = None
    ) -> Decimal:
        """Траты пользователя на нейросеть (USD). 0, если событий нет.

        `since` — точка отсчёта (users.spend_reset_at): для лимита считаем только
        траты после сброса, а админка зовёт без since и видит всю историю.
        """
        query = select(func.coalesce(func.sum(LlmUsageEvent.cost_usd), 0)).where(
            LlmUsageEvent.user_id == user_id
        )
        if since is not None:
            query = query.where(LlmUsageEvent.created_at >= since)
        result = await self._session.execute(query)
        return Decimal(str(result.scalar_one()))

    async def sum_cost_since(self, since: datetime) -> Decimal:
        """Суммарные траты всех юзеров начиная с `since` (UTC). 0, если событий нет.

        Используется в алёрте дневных трат (admin.spend_alert) — ТОЛЬКО уведомление,
        не блокирует юзеров (memory feedback-no-rate-limit).
        """
        result = await self._session.execute(
            select(func.coalesce(func.sum(LlmUsageEvent.cost_usd), 0)).where(
                LlmUsageEvent.created_at >= since
            )
        )
        return Decimal(str(result.scalar_one()))


class SessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        dream_id: int,
        user_id: int,
        type_: SessionType,
        is_onboarding_session: bool = False,
    ) -> Session:
        sess = Session(
            dream_id=dream_id,
            user_id=user_id,
            type=type_,
            status=SessionStatus.IN_PROGRESS,
            is_onboarding_session=is_onboarding_session,
        )
        self._session.add(sess)
        await self._session.flush()
        return sess

    async def get_by_id(self, session_id: int, *, user_id: int) -> Session | None:
        """Возвращает сессию только если она принадлежит запрошенному юзеру."""
        result = await self._session.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_active_for_user_dream(self, *, user_id: int, dream_id: int) -> Session | None:
        """Активная (in_progress или paused) сессия по этому сну, если есть."""
        result = await self._session.execute(
            select(Session)
            .where(
                Session.user_id == user_id,
                Session.dream_id == dream_id,
                Session.status.in_([SessionStatus.IN_PROGRESS, SessionStatus.PAUSED]),
            )
            .order_by(Session.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def set_per_session_context(self, *, session_id: int, context: str) -> None:
        await self._session.execute(
            update(Session).where(Session.id == session_id).values(per_session_context=context)
        )

    async def set_request_branch(self, *, session_id: int, branch: RequestBranch) -> None:
        await self._session.execute(
            update(Session).where(Session.id == session_id).values(request_branch=branch)
        )

    async def complete(self, *, session_id: int) -> None:
        await self._session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=SessionStatus.COMPLETED, finished_at=datetime.now(UTC))
        )

    async def reopen(self, *, session_id: int) -> None:
        """Переоткрыть completed-сессию обратно в IN_PROGRESS.

        Используется при «Продолжить прошлый разбор» (quick.v2.1): юзер
        возвращается к ранее завершённому Quick, чтобы добавить ещё вопрос —
        в той же сессии, с накопленной session_messages.
        """
        await self._session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=SessionStatus.IN_PROGRESS, finished_at=None)
        )

    async def latest_inactive_quick_for_dream(
        self, *, dream_id: int, user_id: int
    ) -> Session | None:
        """Latest PAUSED or COMPLETED Quick session for the dream.

        Used to resume Q&A via /dreams. Takes the most recent by last_active_at,
        whether it's paused (new style) or completed (pre-pivot legacy).
        resume_session() promotes it back to IN_PROGRESS.
        """
        stmt = (
            select(Session)
            .where(
                Session.dream_id == dream_id,
                Session.user_id == user_id,
                Session.type == SessionType.QUICK,
                Session.status.in_([SessionStatus.PAUSED, SessionStatus.COMPLETED]),
            )
            .order_by(Session.last_active_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def resume_session(self, *, session_id: int) -> None:
        """Возобновить PAUSED/COMPLETED-сессию обратно в IN_PROGRESS."""
        await self._session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=SessionStatus.IN_PROGRESS, finished_at=None)
        )

    async def set_target_image_id(self, *, session_id: int, image_id: int) -> None:
        await self._session.execute(
            update(Session).where(Session.id == session_id).values(target_image_id=image_id)
        )

    async def set_current_phase(self, *, session_id: int, phase: SessionPhase) -> None:
        await self._session.execute(
            update(Session).where(Session.id == session_id).values(current_phase=phase)
        )

    async def set_p4_substep(self, *, session_id: int, substep: P4Substep) -> None:
        await self._session.execute(
            update(Session).where(Session.id == session_id).values(p4_substep=substep)
        )

    async def increment_pivot_count(self, *, session_id: int) -> None:
        await self._session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(pivot_count=Session.pivot_count + 1)
        )

    async def pause(self, *, session_id: int) -> None:
        """Помечаем сессию paused — используется в resume-сценарии."""
        await self._session.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(status=SessionStatus.PAUSED)
        )

    async def latest_in_progress_for_user(self, *, user_id: int) -> Session | None:
        """Latest IN_PROGRESS session for the user (any dream/type).

        NOTE: Only returns the most recent — does not surface pre-existing
        legacy duplicates (multiple IN_PROGRESS for one user). Normal flow
        cannot produce duplicates, but if they exist (e.g. from a crash
        mid-create), only one will be cleaned up per call.
        """
        stmt = (
            select(Session)
            .where(
                Session.user_id == user_id,
                Session.status == SessionStatus.IN_PROGRESS,
            )
            .order_by(Session.last_active_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists_in_progress_for_user(self, *, user_id: int) -> bool:
        """True if user has any IN_PROGRESS session (any dream, any type)."""
        stmt = (
            select(Session.id)
            .where(
                Session.user_id == user_id,
                Session.status == SessionStatus.IN_PROGRESS,
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None


class DreamImageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(
        self,
        *,
        dream_id: int,
        images: list[dict[str, object]],
    ) -> list[DreamImage]:
        """Создать набор образов одним батчем."""
        objs = [
            DreamImage(
                dream_id=dream_id,
                name=img["name"],
                ifs_category=img["ifs_category"],
                shape=img["shape"],
            )
            for img in images
        ]
        self._session.add_all(objs)
        await self._session.flush()
        return objs

    async def list_by_dream(self, *, dream_id: int) -> list[DreamImage]:
        result = await self._session.execute(
            select(DreamImage).where(DreamImage.dream_id == dream_id).order_by(DreamImage.id)
        )
        return list(result.scalars().all())

    async def get_by_id_for_user(
        self, *, image_id: int, user_id: int
    ) -> DreamImage | None:
        """Изоляция: образ принадлежит только если родительский Dream — юзера."""
        result = await self._session.execute(
            select(DreamImage)
            .join(Dream, Dream.id == DreamImage.dream_id)
            .where(DreamImage.id == image_id, Dream.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def mark_analyzed(self, *, image_id: int, session_id: int) -> None:
        await self._session.execute(
            update(DreamImage)
            .where(DreamImage.id == image_id)
            .values(analyzed_in_session_id=session_id)
        )


class SessionStateRepository:
    """Upsert-репозиторий для 1:1 session_state.

    SQLite не имеет нативного upsert через ORM; делаем «get → update or insert».
    На Postgres в Phase 6 заменим на ON CONFLICT.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, *, session_id: int, data: dict[str, object]) -> None:
        existing = await self._session.get(SessionState, session_id)
        if existing is None:
            self._session.add(SessionState(session_id=session_id, data=data))
        else:
            existing.data = data
        await self._session.flush()

    async def get_data(self, *, session_id: int) -> dict[str, object] | None:
        row = await self._session.get(SessionState, session_id)
        return None if row is None else dict(row.data)


class SessionMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(
        self,
        *,
        session_id: int,
        direction: SessionMessageDirection,
        content_type: SessionMessageContentType,
        content: str,
        audio_file_id: str | None = None,
        llm_role_used: LlmRole | None = None,
        llm_phase_label: str | None = None,
    ) -> SessionMessage:
        msg = SessionMessage(
            session_id=session_id,
            direction=direction,
            content_type=content_type,
            content=content,
            audio_file_id=audio_file_id,
            llm_role_used=llm_role_used,
            llm_phase_label=llm_phase_label,
        )
        self._session.add(msg)
        await self._session.flush()
        return msg

    async def list_for_resume(
        self, *, session_id: int, limit: int = 8
    ) -> list[SessionMessage]:
        """Последние `limit` реплик в хронологическом порядке."""
        # First get all messages for this session, ordered by time
        all_result = await self._session.execute(
            select(SessionMessage)
            .where(SessionMessage.session_id == session_id)
            .order_by(SessionMessage.created_at.asc())
        )
        all_msgs = list(all_result.scalars().all())
        # Return the last `limit` messages in chronological order
        return all_msgs[-limit:] if len(all_msgs) > 0 else []

    async def user_replies_with_prompts(
        self, *, session_id: int
    ) -> list[tuple[str | None, str]]:
        """Пары (последняя проза бота ПЕРЕД репликой юзера, текст реплики юзера).

        Проза бота — только контекст для понимания вопроса; извлекать из неё нельзя.

        Вопросы фазы image_q (заземляющие вопросы по образам сна) получают префикс
        GROUNDING_QUESTION_MARKER: юзер в них отвечает на ПРЯМОЙ вопрос о своей
        реальной жизни, и экстрактор памяти (memory.extract.v3) не должен принимать
        такой ответ за разговор о сне. Остальные пары не помечаются.
        """
        result = await self._session.execute(
            select(SessionMessage)
            .where(SessionMessage.session_id == session_id)
            .order_by(SessionMessage.id.asc())
        )
        pairs: list[tuple[str | None, str]] = []
        last_bot: str | None = None
        last_bot_label: str | None = None
        for msg in result.scalars().all():
            if msg.direction == SessionMessageDirection.BOT_TO_USER:
                last_bot = msg.content
                last_bot_label = msg.llm_phase_label
            else:
                prompt = last_bot
                if prompt is not None:
                    if last_bot_label == QUICK_PHASE_IMAGE_Q:
                        prompt = GROUNDING_QUESTION_MARKER + prompt
                    elif last_bot_label == QUICK_PHASE_PRE_Q:
                        prompt = PRE_QUESTION_MARKER + prompt
                pairs.append((prompt, msg.content))
        return pairs

    async def last_bot_message(self, *, session_id: int) -> str | None:
        """Текст последней реплики бота в сессии (вопрос, на который отвечает юзер)."""
        result = await self._session.execute(
            select(SessionMessage.content)
            .where(
                SessionMessage.session_id == session_id,
                SessionMessage.direction == SessionMessageDirection.BOT_TO_USER,
            )
            .order_by(SessionMessage.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


class UserMemoryItemRepository:
    """CRUD долговременной памяти о юзере (user_memory_items).

    Пишет экстрактор (memory.extract) на паузе сессии; читают промпты Q&A/разбора.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        user_id: int,
        kind: str,
        text: str,
        realm: str,
        source: str,
        entity_key: str | None = None,
        source_session_id: int | None = None,
        dream_id: int | None = None,
        occurred_at: date | None = None,
    ) -> UserMemoryItem:
        item = UserMemoryItem(
            user_id=user_id,
            kind=kind,
            text=text,
            realm=realm,
            source=source,
            entity_key=entity_key,
            source_session_id=source_session_id,
            dream_id=dream_id,
            occurred_at=occurred_at,
        )
        self._session.add(item)
        await self._session.flush()
        return item

    async def list_active(self, *, user_id: int) -> list[UserMemoryItem]:
        stmt = (
            select(UserMemoryItem)
            .where(
                UserMemoryItem.user_id == user_id,
                UserMemoryItem.status == "active",
            )
            .order_by(UserMemoryItem.last_confirmed_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def confirm(self, item_id: int) -> None:
        await self._session.execute(
            update(UserMemoryItem)
            .where(UserMemoryItem.id == item_id)
            .values(
                last_confirmed_at=datetime.now(UTC),
                confidence=UserMemoryItem.confidence + 1,
            )
        )

    async def supersede(self, old_id: int, new_id: int) -> None:
        await self._session.execute(
            update(UserMemoryItem)
            .where(UserMemoryItem.id == old_id)
            .values(status="superseded", superseded_by_id=new_id)
        )

    async def find_by_entity(
        self, user_id: int, entity_key: str, *, kind: str | None = None
    ) -> UserMemoryItem | None:
        """Active-юнит по entity_key; kind сужает поиск (part матчится только с part)."""
        if entity_key is None:  # guard: callers must pass a real key
            return None
        stmt = (
            select(UserMemoryItem)
            .where(
                UserMemoryItem.user_id == user_id,
                UserMemoryItem.entity_key == entity_key,
                UserMemoryItem.status == "active",
            )
            .limit(1)
        )
        if kind is not None:
            stmt = stmt.where(UserMemoryItem.kind == kind)
        return (await self._session.execute(stmt)).scalar_one_or_none()
