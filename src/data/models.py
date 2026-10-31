from __future__ import annotations

import enum
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    false,
    func,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from domain.access import TIER_NEW


class Base(DeclarativeBase):
    pass


# --- enums (отражают spec §7) ---


class DreamStatus(str, enum.Enum):
    RECORDED = "recorded"
    IN_ANALYSIS = "in_analysis"
    ANALYZED = "analyzed"
    FULLY_EXPLORED = "fully_explored"


class SessionType(str, enum.Enum):
    QUICK = "quick"
    DEEP = "deep"


class SessionStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    SAFETY_PAUSED = "safety_paused"


class RequestBranch(str, enum.Enum):
    """6 вариантов запроса разбора (spec §3 stage 2) + custom.

    CUSTOM — юзер вместо тапа по кнопке написал/наговорил свой запрос в свободной
    форме. Сам текст запроса хранится в session_state (JSON), а в request_branch
    проставляется CUSTOM — чтобы в аналитике такие сессии были видны отдельно.
    """

    FEEL_STATE = "feel_state"
    DECISION = "decision"
    CONFLICT = "conflict"
    CALM = "calm"
    NEW_ABOUT_SELF = "new_about_self"
    UNCLEAR = "unclear"
    CUSTOM = "custom"


class ExperienceLevel(str, enum.Enum):
    NEWBIE = "newbie"
    SOME_EXPERIENCE = "some_experience"
    IN_THERAPY = "in_therapy"


class LlmService(str, enum.Enum):
    OPENROUTER = "openrouter"
    WHISPER = "whisper"


class UsageMetric(str, enum.Enum):
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    AUDIO_SECONDS = "audio_seconds"


class ImageShape(str, enum.Enum):
    """Форма образа во сне (spec §5 P4 — определяет адаптацию вопросов)."""

    HUMAN_LIKE = "human-like"
    ANIMAL = "animal"
    OBJECT = "object"
    ELEMENT = "element"  # огонь, вода, воздух, земля
    PHENOMENON = "phenomenon"  # туман, гроза, тишина
    SENSATION = "sensation"  # удушье, давление, тепло
    ABSTRACT = "abstract"  # «чувство», «знание», «понимание»


class IfsCategory(str, enum.Enum):
    """IFS-классификация роли образа (внутренняя — юзеру не показываем)."""

    MANAGER = "manager"
    FIREFIGHTER = "firefighter"
    EXILE = "exile"
    UNCLEAR = "unclear"


class SessionPhase(str, enum.Enum):
    """Текущая фаза Deep-сессии. Quick-сессии оставляют NULL."""

    P2_FOCUS = "p2_focus"
    P3_FEEL = "p3_feel"
    SELF_GATE = "self_gate"
    UNBLENDING = "unblending"
    P4 = "p4"
    P5 = "p5"
    P6 = "p6"


class P4Substep(str, enum.Enum):
    """Подфазы P4 (beFriend). Идут линейно embodiment → origin → role → attitude."""

    EMBODIMENT = "embodiment"
    ORIGIN = "origin"
    ROLE = "role"
    ATTITUDE = "attitude"


class SessionMessageDirection(str, enum.Enum):
    USER_TO_BOT = "user_to_bot"
    BOT_TO_USER = "bot_to_user"


class SessionMessageContentType(str, enum.Enum):
    TEXT = "text"
    VOICE = "voice"


class LlmRole(str, enum.Enum):
    GENERATOR = "generator"
    CLASSIFIER = "classifier"


DISPLAY_NAME_MAX_LEN = 128

# --- models ---


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [m.value for m in enum_cls]


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(DISPLAY_NAME_MAX_LEN))
    username: Mapped[str | None] = mapped_column(String(64))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=true(), default=True)
    blocked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dreams_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0", default=0)
    qa_answers_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0", default=0)
    locale: Mapped[str] = mapped_column(String(8), default="ru", server_default="ru")
    acquisition_source: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Тиры доступа (2026-08-01): порог трат зависит от группы, см. src/domain/access.py
    access_tier: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=TIER_NEW, default=TIER_NEW
    )
    access_tier_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Точка отсчёта лимита и факты о доступе.
    # spend_reset_at — траты для лимита считаются ПОСЛЕ неё; в админке при этом
    # видна полная история трат.
    spend_reset_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paywall_shown_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Дата ПЕРВОГО интервью: переживает истечение безлимита и смену тира.
    interview_passed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Тир, в который вернуть по кнопке «Обычный тестер»: иначе снятие интервью
    # стирало бы бейджик «1» у первых тестеров.
    tier_before_interview: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Phase 1: профиль из онбординга
    profile_life_context: Mapped[str | None] = mapped_column(Text)
    profile_experience_level: Mapped[ExperienceLevel | None] = mapped_column(
        Enum(
            ExperienceLevel,
            name="experience_level",
            values_callable=_enum_values,
        )
    )
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Версия набора команд в меню «☰», уже выставленного этому юзеру. Telegram
    # хранит per-chat список вечно, поэтому при изменении COMMANDS_VERSION здесь
    # остаётся старое число — и первое же сообщение юзера обновляет ему меню.
    menu_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    dreams: Mapped[list[Dream]] = relationship(back_populates="user")
    sessions: Mapped[list[Session]] = relationship(back_populates="user")
    usage_events: Mapped[list[LlmUsageEvent]] = relationship(back_populates="user")


class Dream(Base):
    __tablename__ = "dreams"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    transcription: Mapped[str] = mapped_column(String)
    raw_audio_file_id: Mapped[str | None] = mapped_column(String(256))
    title: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[DreamStatus] = mapped_column(
        Enum(DreamStatus, name="dream_status", values_callable=_enum_values),
        default=DreamStatus.RECORDED,
        server_default="recorded",
    )
    # Тип обращения (спека B1): dream / external_situation / internal_conflict.
    # Строка (не Enum) — новый тип не потребует миграции; значения — KIND_* из
    # domain.entry_types. Физическое имя таблицы остаётся dreams, ментальная
    # модель — «обращение».
    kind: Mapped[str] = mapped_column(
        String(24), nullable=False, default="dream", server_default="dream", index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="dreams")
    sessions: Mapped[list[Session]] = relationship(back_populates="dream")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    dream_id: Mapped[int] = mapped_column(ForeignKey("dreams.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    type: Mapped[SessionType] = mapped_column(
        Enum(SessionType, name="session_type", values_callable=_enum_values)
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status", values_callable=_enum_values),
        default=SessionStatus.IN_PROGRESS,
        server_default="in_progress",
    )
    per_session_context: Mapped[str | None] = mapped_column(String(2048))
    is_onboarding_session: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=false(), default=False
    )
    request_branch: Mapped[RequestBranch | None] = mapped_column(
        Enum(RequestBranch, name="request_branch", values_callable=_enum_values)
    )

    # Phase 3: Deep IFS state machine
    target_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("dream_images.id", use_alter=True, name="fk_sessions_target_image")
    )
    current_phase: Mapped[SessionPhase | None] = mapped_column(
        Enum(SessionPhase, name="session_phase", values_callable=_enum_values)
    )
    p4_substep: Mapped[P4Substep | None] = mapped_column(
        Enum(P4Substep, name="p4_substep", values_callable=_enum_values)
    )
    pivot_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    dream: Mapped[Dream] = relationship(back_populates="sessions")
    user: Mapped[User] = relationship(back_populates="sessions")
    usage_events: Mapped[list[LlmUsageEvent]] = relationship(back_populates="session")


class LlmUsageEvent(Base):
    """Учёт каждого внешнего LLM/STT вызова (spec §7).

    `session_id` nullable: в Phase 1 Whisper-вызовы относятся не к сессии,
    а к процессу записи сна (или вообще к онбордингу).
    """

    __tablename__ = "llm_usage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"), index=True)
    service: Mapped[LlmService] = mapped_column(
        Enum(LlmService, name="llm_service", values_callable=_enum_values)
    )
    model: Mapped[str] = mapped_column(String(64))
    usage_metric: Mapped[UsageMetric] = mapped_column(
        Enum(UsageMetric, name="usage_metric", values_callable=_enum_values)
    )
    usage_amount: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User | None] = relationship(back_populates="usage_events")
    session: Mapped[Session | None] = relationship(back_populates="usage_events")


class AnalyticsEvent(Base):
    """Append-only сырые события для аналитики (воронка онбординга + daily_active).

    Агрегация (когорты/периоды/source) считается В МОМЕНТ ОТЧЁТА, не здесь.
    `event_type` — обычная строка (НЕ Enum), чтобы новый тип не требовал миграции.
    R_analysis/qa/return/spend тут НЕ хранятся — считаются по доменным таблицам.
    """

    __tablename__ = "analytics_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=True
    )
    # Дублируем tg_id для устойчивости выгрузок (юзер мог быть удалён).
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # date(occurred_at) — для дешёвого dedup daily_active по (user_id, event_day).
    event_day: Mapped[date] = mapped_column(Date, index=True)
    event_metadata: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)


class DreamImage(Base):
    """Образ из сна, выявленный P2_FIND classifier'ом (Phase 3)."""

    __tablename__ = "dream_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    dream_id: Mapped[int] = mapped_column(ForeignKey("dreams.id"), index=True)
    name: Mapped[str] = mapped_column(String(256))
    ifs_category: Mapped[IfsCategory] = mapped_column(
        Enum(IfsCategory, name="ifs_category", values_callable=_enum_values)
    )
    shape: Mapped[ImageShape] = mapped_column(
        Enum(ImageShape, name="image_shape", values_callable=_enum_values)
    )
    analyzed_in_session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dream: Mapped[Dream] = relationship()


class SessionState(Base):
    """1:1 с sessions: structured snapshot текущего IFS-знания о target_part.

    Хранится как JSON (sqlite TEXT / postgres JSONB). Схема нормализована
    в domain/states.py:SessionStateData (Pydantic).
    """

    __tablename__ = "session_state"

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id"), primary_key=True
    )
    data: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class SessionMessage(Base):
    """Append-only лог реплик внутри сессии (юзер и бот).

    Используется для:
    - resume: чтобы воспроизвести юзеру последнюю реплику бота после рестарта;
    - few-shot истории в LLM-вызовах (последние 4 пары);
    - в Phase 4 — для feature «перечитать что было».
    """

    __tablename__ = "session_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    direction: Mapped[SessionMessageDirection] = mapped_column(
        Enum(
            SessionMessageDirection,
            name="session_message_direction",
            values_callable=_enum_values,
        )
    )
    content_type: Mapped[SessionMessageContentType] = mapped_column(
        Enum(
            SessionMessageContentType,
            name="session_message_content_type",
            values_callable=_enum_values,
        )
    )
    content: Mapped[str] = mapped_column(String)
    audio_file_id: Mapped[str | None] = mapped_column(String(256))
    llm_role_used: Mapped[LlmRole | None] = mapped_column(
        Enum(LlmRole, name="llm_role", values_callable=_enum_values)
    )
    llm_phase_label: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserMemoryItem(Base):
    """Типизированная единица долговременной памяти о реальной жизни юзера.

    Пишется экстрактором (memory.extract) на паузе сессии ТОЛЬКО из реплик юзера.
    realm ∈ {real_life, association} — dream_content сюда не попадает никогда.
    kind/realm/source/status — строки (не Enum): новый вид без миграции.
    """

    __tablename__ = "user_memory_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24), index=True)
    text: Mapped[str] = mapped_column(String(500))
    entity_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    realm: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(24))
    source_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True
    )
    dream_id: Mapped[int | None] = mapped_column(ForeignKey("dreams.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), server_default="active", default="active")
    superseded_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("user_memory_items.id"), nullable=True
    )
    confidence: Mapped[int] = mapped_column(SmallInteger, server_default="1", default=1)
    occurred_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class FsmState(Base):
    """Персистентное хранилище FSM-state aiogram3.

    До quick.v2.1 использовался RAM `MemoryStorage` (см. [[feedback-bot-restart-loses-fsm]]):
    рестарт uvicorn стирал state у всех юзеров. Эта таблица — backend для
    `SqlAlchemyFSMStorage` (adapters/telegram/fsm_storage.py), переживающего
    рестарт. key — сериализованный aiogram StorageKey:
    `{bot_id}:{chat_id}:{user_id}:{thread_id}:{business_id}:{destiny}`.
    """

    __tablename__ = "fsm_state"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    state: Mapped[str | None] = mapped_column(String(128), nullable=True)
    data: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
