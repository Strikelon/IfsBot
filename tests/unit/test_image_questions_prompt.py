"""Тесты image_questions.v1: парсер строгого JSON и сборка messages."""
from data.models import RequestBranch
from llm.prompts.image_questions import (
    IMAGE_QUESTIONS_VERSION,
    ImageQuestion,
    build_image_answers_block,
    build_image_questions_messages,
    parse_image_questions,
)
from llm.prompts.quick import QuickContext


def _ctx(**kw) -> QuickContext:
    defaults = dict(
        user_display_name="Александр",
        user_life_context="Эмигрант в Грузии, двое детей.",
        per_session_context="Думал о запуске проекта.",
        request_branch=RequestBranch.UNCLEAR,
        dream_transcription="Мне снился отец и старый автомобиль.",
    )
    defaults.update(kw)
    return QuickContext(**defaults)


def test_version():
    assert IMAGE_QUESTIONS_VERSION == "image_questions.v1"


def test_parse_valid_two_images():
    raw = ('{"images":[{"name":"Отец","question":"Как бы вы охарактеризовали '
           'отношения с отцом в реальности?"},'
           '{"name":"Старый автомобиль","question":"Какие эмоции у вас вызывают '
           'старые автомобили в реальности?"}]}')
    items, reason = parse_image_questions(raw)
    assert reason is None
    assert [i.name for i in items] == ["Отец", "Старый автомобиль"]
    assert all(isinstance(i, ImageQuestion) and i.question for i in items)


def test_parse_strips_fences():
    raw = '```json\n{"images":[{"name":"Тигр","question":"Как вы относитесь к тиграм?"}]}\n```'
    items, reason = parse_image_questions(raw)
    assert reason is None
    assert items[0].name == "Тигр"


def test_parse_broken_json_is_parse_error():
    items, reason = parse_image_questions("{'oops': broken")
    assert items == [] and reason == "parse_error"


def test_parse_non_dict_is_parse_error():
    items, reason = parse_image_questions('["a","b"]')
    assert items == [] and reason == "parse_error"


def test_parse_empty_images_is_empty():
    items, reason = parse_image_questions('{"images":[]}')
    assert items == [] and reason == "empty"


def test_parse_caps_at_three():
    raw = ('{"images":[' + ",".join(
        f'{{"name":"Образ {i}","question":"Вопрос {i}?"}}' for i in range(5)
    ) + "]}")
    items, reason = parse_image_questions(raw)
    assert reason is None and len(items) == 3


def test_parse_drops_invalid_elements():
    raw = ('{"images":[{"name":"","question":"?"},{"name":"Ок","question":""},'
           '{"name":42,"question":"?"},"строка",'
           '{"name":"Отец","question":"Как отношения с отцом в реальности?"}]}')
    items, reason = parse_image_questions(raw)
    assert reason is None
    assert [i.name for i in items] == ["Отец"]


def test_parse_all_invalid_is_empty():
    items, reason = parse_image_questions('{"images":[{"name":"","question":""}]}')
    assert items == [] and reason == "empty"


def test_system_forbids_pre_interpretation_of_image():
    """GATE-итерация: вопрос про САМ образ, бот не подменяет его своей расшифровкой."""
    system = build_image_questions_messages(_ctx())[0]["content"]
    assert "НЕ подменяй образ своей интерпретацией" in system
    assert "ОБЯЗАНО присутствовать в вопросе" in system
    # Конкретный BAD→GOOD пример с вампиром из обратной связи
    assert "ПЛОХО (образ подменён интерпретацией)" in system
    assert "скрытая угроза, забирающая ваши силы" in system
    assert "ХОРОШО (про сам образ)" in system
    assert "относитесь к вампирам" in system


def test_messages_contain_dream_and_memory():
    msgs = build_image_questions_messages(_ctx(memory_block="ЧТО УЖЕ ИЗВЕСТНО:\n- [факт] X"))
    assert msgs[0]["role"] == "system"
    assert "1–3" in msgs[0]["content"] or "1-3" in msgs[0]["content"]
    assert "старый автомобиль" in msgs[1]["content"].lower()
    assert "ЧТО УЖЕ ИЗВЕСТНО" in msgs[1]["content"]


def test_answers_block_only_answered():
    items = [
        {"name": "Отец", "question": "q1", "answer": "Отношения тёплые", "skipped": False},
        {"name": "Тигр", "question": "q2", "answer": None, "skipped": True},
    ]
    block = build_image_answers_block(items)
    assert "Отец" in block and "Отношения тёплые" in block
    assert "Тигр" not in block


def test_answers_block_empty_when_none_answered():
    assert build_image_answers_block([{"name": "Тигр", "answer": None, "skipped": True}]) == ""


def test_answers_block_truncates_long_answer():
    # _MAX_ANSWER_CHARS = 600: ответ режется ровно до 600 символов
    # (брифовый ассерт len(block) < 700 не сходился с заголовком блока — 97 симв.)
    items = [{"name": "Отец", "question": "q", "answer": "х" * 1000, "skipped": False}]
    block = build_image_answers_block(items)
    assert "х" * 600 in block
    assert "х" * 601 not in block
