"""Реестр промпт-модулей по типу обращения (спека B2/B4).

kind → EntryPromptBundle: версии + сборщики messages + парсер пре-вопросов +
сборщик блока ответов. Dream-бандл указывает на СУЩЕСТВУЮЩИЕ функции
quick.py / image_questions.py — поведение ветки сна не меняется ни на байт
(закреплено tests/unit/test_entry_prompts_registry.py через `is`-сравнения).

Здесь же — общие хелперы пре-вопросов новых типов: parse_pre_questions
(строгий JSON {"questions":[{"topic","question"}]}) и build_pre_answers_block.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION
from llm.classifier import _strip_fences
from llm.prompts import conflict, quick, situation
from llm.prompts.image_questions import (
    IMAGE_QUESTIONS_VERSION,
    ImageQuestion,
    build_image_answers_block,
    build_image_questions_messages,
    parse_image_questions,
)
from llm.prompts.quick import Message, QuickContext

_MAX_QUESTIONS = 3
# Обрезка одного ответа юзера в блоке initial-промпта (паттерн image_questions).
_MAX_ANSWER_CHARS = 600


def parse_pre_questions(raw: str) -> tuple[list[ImageQuestion], str | None]:
    """Распарсить строгий JSON пре-вопросов новых типов. Никогда не бросает.

    Контракт результата 1:1 с parse_image_questions: (items, fail_reason).
    topic кладётся в ImageQuestion.name — оркестратор и аналитика используют его
    как «имя» вопроса (метаданные событий, session_state.items[].name).
    """
    try:
        payload = json.loads(_strip_fences(raw.strip()))
    except (json.JSONDecodeError, ValueError):
        return [], "parse_error"
    if not isinstance(payload, dict):
        return [], "parse_error"
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return [], "parse_error"
    out: list[ImageQuestion] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        question = item.get("question")
        if not isinstance(topic, str) or not isinstance(question, str):
            continue
        topic, question = topic.strip(), question.strip()
        if not topic or not question:
            continue
        out.append(ImageQuestion(name=topic, question=question))
        if len(out) >= _MAX_QUESTIONS:
            break
    if not out:
        return [], "empty"
    return out, None


def build_pre_answers_block(items: list[dict]) -> str:
    """Блок «что юзер ответил на уточняющие вопросы» для initial-промпта.

    Только отвеченные items (answer непустой); скипы не попадают.
    Пустая строка = блок не подставляется.
    """
    answered = [
        i for i in items
        if isinstance(i, dict) and isinstance(i.get("answer"), str) and i["answer"].strip()
    ]
    if not answered:
        return ""
    lines = [
        "ЧТО ПОЛЬЗОВАТЕЛЬ ОТВЕТИЛ НА УТОЧНЯЮЩИЕ ВОПРОСЫ (собрано до разбора):"
    ]
    for it in answered:
        answer = it["answer"].strip()[:_MAX_ANSWER_CHARS]
        question = str(it.get("question", "")).strip()
        lines.append(f"– На вопрос «{question}»: {answer}")
    return "\n".join(lines)


@dataclass(frozen=True)
class EntryPromptBundle:
    """Промптовая часть одного типа обращения (версии пишутся в llm_usage_events)."""

    version_initial: str
    version_followup: str
    version_pre_questions: str
    build_initial_messages: Callable[[QuickContext], list[Message]]
    build_followup_messages: Callable[..., list[Message]]
    build_pre_questions_messages: Callable[[QuickContext], list[Message]]
    parse_pre_questions: Callable[[str], tuple[list[ImageQuestion], str | None]]
    build_answers_block: Callable[[list[dict]], str]


PROMPTS_BY_KIND: dict[str, EntryPromptBundle] = {
    KIND_DREAM: EntryPromptBundle(
        version_initial=quick.PROMPT_VERSION_INITIAL,
        version_followup=quick.PROMPT_VERSION_FOLLOWUP,
        version_pre_questions=IMAGE_QUESTIONS_VERSION,
        build_initial_messages=quick.build_initial_messages,
        build_followup_messages=quick.build_followup_messages,
        build_pre_questions_messages=build_image_questions_messages,
        parse_pre_questions=parse_image_questions,
        build_answers_block=build_image_answers_block,
    ),
    KIND_SITUATION: EntryPromptBundle(
        version_initial=situation.SITUATION_VERSION_INITIAL,
        version_followup=situation.SITUATION_VERSION_FOLLOWUP,
        version_pre_questions=situation.SITUATION_VERSION_QUESTIONS,
        build_initial_messages=situation.build_initial_messages,
        build_followup_messages=situation.build_followup_messages,
        build_pre_questions_messages=situation.build_questions_messages,
        parse_pre_questions=parse_pre_questions,
        build_answers_block=build_pre_answers_block,
    ),
    KIND_CONFLICT: EntryPromptBundle(
        version_initial=conflict.CONFLICT_VERSION_INITIAL,
        version_followup=conflict.CONFLICT_VERSION_FOLLOWUP,
        version_pre_questions=conflict.CONFLICT_VERSION_QUESTIONS,
        build_initial_messages=conflict.build_initial_messages,
        build_followup_messages=conflict.build_followup_messages,
        build_pre_questions_messages=conflict.build_questions_messages,
        parse_pre_questions=parse_pre_questions,
        build_answers_block=build_pre_answers_block,
    ),
}


def prompt_bundle(kind: str | None) -> EntryPromptBundle:
    """Бандл по kind; незнакомое/пустое значение → dream (fail-safe)."""
    return PROMPTS_BY_KIND.get(kind or KIND_DREAM, PROMPTS_BY_KIND[KIND_DREAM])
