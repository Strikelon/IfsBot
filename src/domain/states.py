"""Чистый domain-модуль IFS state machine.

Структура session_state.data (JSON в БД), схемы Pydantic для валидации,
shape-таблицы для адаптации промптов P4. Не знает про БД и Telegram.

Используется в:
- adapters/telegram/handlers/deep.py — чтение/запись session_state
- llm/prompts/p4.py — выбор формулировки вопроса под shape
- llm/prompts/p2_focus.py — somatic-вопрос под shape
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from data.models import ImageShape

# --- session_state.data schema ---

UnblendingOutcome = Literal["released", "blended"]


class UnblendingAttempt(BaseModel):
    """Одна попытка unblending'а: какая эмоция всплыла и удалось ли отойти."""

    emotion: str
    outcome: UnblendingOutcome


class SessionStateData(BaseModel):
    """JSON-структура поля session_state.data.

    Заполняется инкрементально handler'ами: после P2_FOCUS — somatic_location;
    после успешного P3_FEEL+SELF_GATE — пусто (мы просто переходим в P4);
    после P4_embodiment — part_voice_form; после P4_origin — part_origin_context;
    после P4_role — part_role; после P4_attitude — переходим в P5;
    после P5 — part_fears.

    pivot_history накапливается каждый раз при триггере pivot (см. transitions.py).
    """

    target_part_name: str
    target_part_shape: ImageShape
    target_part_ifs_category: str  # IfsCategory.value (lowercase string)

    somatic_location: str | None = None
    part_voice_form: str | None = None  # из P4_embodiment
    part_origin_context: str | None = None  # из P4_origin
    part_role: str | None = None  # из P4_role
    part_fears: str | None = None  # из P5

    unblending_history: list[UnblendingAttempt] = Field(default_factory=list)
    pivot_history: list[str] = Field(default_factory=list)


# --- shape таблицы для prompt-адаптации (spec §5 P4) ---

_SHAPE_LABEL: dict[ImageShape, str] = {
    ImageShape.HUMAN_LIKE: "фигура",
    ImageShape.ANIMAL: "животное",
    ImageShape.OBJECT: "предмет",
    ImageShape.ELEMENT: "стихия",
    ImageShape.PHENOMENON: "явление",
    ImageShape.SENSATION: "ощущение",
    ImageShape.ABSTRACT: "это",
}

_P4_ROLE_VERB: dict[ImageShape, str] = {
    ImageShape.HUMAN_LIKE: "делает",
    ImageShape.ANIMAL: "охраняет",
    ImageShape.OBJECT: "защищает",
    ImageShape.ELEMENT: "пытается сделать",
    ImageShape.PHENOMENON: "напоминает или скрывает",
    ImageShape.SENSATION: "сигнализирует",
    ImageShape.ABSTRACT: "имеет в виду",
}

_P4_ATTITUDE_PHRASE: dict[ImageShape, str] = {
    ImageShape.HUMAN_LIKE: "довольна ли часть тем, что делает; хочет ли иначе",
    ImageShape.ANIMAL: "выбрала бы оно другую роль, если бы могло",
    ImageShape.OBJECT: "осталось бы оно тем же предметом, если бы могло иначе",
    ImageShape.ELEMENT: "хочет ли эта стихия проявляться по-другому",
    ImageShape.PHENOMENON: "хочет ли это явление продолжать так же или изменилось бы",
    ImageShape.SENSATION: "хочет ли это ощущение продолжать сигналить так же",
    ImageShape.ABSTRACT: "хочет ли это сохраняться или меняться",
}

_P4_EMBODIMENT_CHANNEL: dict[ImageShape, str] = {
    ImageShape.HUMAN_LIKE: "как образ, голос или ощущение",
    ImageShape.ANIMAL: "как движение, звук, повадка",
    ImageShape.OBJECT: "как материал, форма, тяжесть",
    ImageShape.ELEMENT: "как температура, движение, плотность",
    ImageShape.PHENOMENON: "как ощущение времени, видимости, плотности",
    ImageShape.SENSATION: "как место в теле, тип давления или тепла",
    ImageShape.ABSTRACT: "как любое слово, картинка или ощущение",
}

_P4_ORIGIN_FRAME: dict[ImageShape, str] = {
    ImageShape.HUMAN_LIKE: "к какому моменту вашей жизни эта часть относится",
    ImageShape.ANIMAL: "в каком моменте жизни могло появиться это животное",
    ImageShape.OBJECT: "к какой ситуации в вашей жизни этот предмет может быть привязан",
    ImageShape.ELEMENT: "в какой момент жизни эта стихия могла войти",
    ImageShape.PHENOMENON: "когда вы впервые встретили это в жизни",
    ImageShape.SENSATION: "в какой момент жизни вы впервые ощутили это",
    ImageShape.ABSTRACT: "в какой момент жизни это могло сформироваться",
}


def shape_label(shape: ImageShape) -> str:
    """Существительное, обозначающее форму образа («фигура»/«животное»/...).

    Используется в P4-промптах и в UI-формировании списка образов.
    """
    return _SHAPE_LABEL[shape]


def p4_role_verb(shape: ImageShape) -> str:
    """Глагол для P4_role вопроса. «что эта часть {делает|охраняет|...}»."""
    return _P4_ROLE_VERB[shape]


def p4_attitude_phrase(shape: ImageShape) -> str:
    """Фраза для P4_attitude вопроса. Слово «довольна» — только для human-like."""
    return _P4_ATTITUDE_PHRASE[shape]


def p4_embodiment_channel(shape: ImageShape) -> str:
    """Какой канал предлагать для P4_embodiment в зависимости от shape."""
    return _P4_EMBODIMENT_CHANNEL[shape]


def p4_origin_frame(shape: ImageShape) -> str:
    """Формулировка «момента появления» для P4_origin."""
    return _P4_ORIGIN_FRAME[shape]
