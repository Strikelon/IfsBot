"""Реестр промпт-бандлов: dream = те же функции quick/image_questions; парсер вопросов."""

from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION
from llm.prompts import conflict, quick, situation
from llm.prompts.entry_prompts import (
    PROMPTS_BY_KIND,
    build_pre_answers_block,
    parse_pre_questions,
    prompt_bundle,
)
from llm.prompts.image_questions import (
    IMAGE_QUESTIONS_VERSION,
    build_image_answers_block,
    build_image_questions_messages,
    parse_image_questions,
)


def test_dream_bundle_is_exactly_the_existing_functions():
    b = PROMPTS_BY_KIND[KIND_DREAM]
    assert b.build_initial_messages is quick.build_initial_messages
    assert b.build_followup_messages is quick.build_followup_messages
    assert b.build_pre_questions_messages is build_image_questions_messages
    assert b.parse_pre_questions is parse_image_questions
    assert b.build_answers_block is build_image_answers_block
    assert b.version_initial == quick.PROMPT_VERSION_INITIAL
    assert b.version_followup == quick.PROMPT_VERSION_FOLLOWUP
    assert b.version_pre_questions == IMAGE_QUESTIONS_VERSION


def test_new_kind_bundles_wired():
    s = PROMPTS_BY_KIND[KIND_SITUATION]
    assert s.build_initial_messages is situation.build_initial_messages
    assert s.version_initial == "situation.v1"
    assert s.version_pre_questions == "situation_questions.v1"
    assert s.parse_pre_questions is parse_pre_questions
    assert s.build_answers_block is build_pre_answers_block
    c = PROMPTS_BY_KIND[KIND_CONFLICT]
    assert c.build_followup_messages is conflict.build_followup_messages
    assert c.version_followup == "conflict.v1.followup"


def test_prompt_bundle_fallback_to_dream():
    assert prompt_bundle("garbage") is PROMPTS_BY_KIND[KIND_DREAM]
    assert prompt_bundle(None) is PROMPTS_BY_KIND[KIND_DREAM]


def test_parse_pre_questions_happy_path_and_cap():
    raw = (
        '{"questions":[{"topic":"Чувства в моменте","question":"Что вы почувствовали?"},'
        '{"topic":"Похожие ситуации","question":"Бывали ли похожие ситуации?"},'
        '{"topic":"Опасения","question":"Чего вы опасаетесь больше всего?"},'
        '{"topic":"Лишний","question":"Не должен попасть"}]}'
    )
    items, fail = parse_pre_questions(raw)
    assert fail is None
    assert [i.name for i in items] == ["Чувства в моменте", "Похожие ситуации", "Опасения"]
    assert items[0].question == "Что вы почувствовали?"


def test_parse_pre_questions_errors():
    assert parse_pre_questions("не json") == ([], "parse_error")
    assert parse_pre_questions('{"questions": []}') == ([], "empty")
    assert parse_pre_questions('{"questions": "x"}') == ([], "parse_error")
    # ```-фенсы вокруг JSON — переживаем (как parse_image_questions)
    items, fail = parse_pre_questions(
        '```json\n{"questions":[{"topic":"Т","question":"В?"}]}\n```'
    )
    assert fail is None and items[0].name == "Т"


def test_build_pre_answers_block():
    items = [
        {"name": "Чувства", "question": "Что вы почувствовали?", "answer": "обиду",
         "skipped": False},
        {"name": "Скип", "question": "Про что?", "answer": None, "skipped": True},
    ]
    block = build_pre_answers_block(items)
    assert "ЧТО ПОЛЬЗОВАТЕЛЬ ОТВЕТИЛ НА УТОЧНЯЮЩИЕ ВОПРОСЫ" in block
    assert "«Что вы почувствовали?»: обиду" in block
    assert "Скип" not in block
    assert build_pre_answers_block([]) == ""
