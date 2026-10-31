"""Юнит-тесты кодовых защит memory_extract (Task 5, память о юзере).

Дизайн: мокаем llm.memory_extract.classify (monkeypatch), подсовываем сырой
parsed (_Extract), проверяем что кодовые защиты фильтруют кандидатов. Чистые
юнит-тесты — без БД и без реального LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from llm.classifier import ClassifierParseError
from llm.memory_extract import (
    MEMORY_EXTRACT_VERSION,
    ExistingItemView,
    extract_memory_items,
)


@dataclass(frozen=True)
class _FakeClassifierResult:
    parsed: object
    input_tokens: int = 11
    output_tokens: int = 7


def _mock_classify(
    monkeypatch: pytest.MonkeyPatch, *, items: list[dict], core: list[dict] | None = None
) -> AsyncMock:
    """Подменить classify так, чтобы он вернул _Extract с заданными items (+ core)."""
    from llm.memory_extract import _Extract

    parsed = _Extract.model_validate({"items": items, "core": core or []})
    spy = AsyncMock(return_value=_FakeClassifierResult(parsed=parsed))
    monkeypatch.setattr("llm.memory_extract.classify", spy)
    return spy


async def _run(**overrides):
    kwargs = dict(
        client=object(),
        model="m",
        existing_items=[],
        user_replies=[(None, "реплика юзера")],
        grounded_labels=[],
        per_session_context="",
    )
    kwargs.update(overrides)
    return await extract_memory_items(**kwargs)


@pytest.mark.asyncio
async def test_version_constant():
    assert MEMORY_EXTRACT_VERSION == "memory.extract.v4"


def test_system_prompt_core_contract():
    """v4: закрытый список категорий, «пустой core — норма», JSON с двумя списками."""
    from llm.memory_extract import _SYSTEM

    for key in ("age", "family", "work", "place", "health", "tone"):
        assert key in _SYSTEM
    assert "core=[]" in _SYSTEM
    assert "нормальный случай" in _SYSTEM
    assert '"core":[{"entity_key"' in _SYSTEM


def test_core_categories_are_exactly_six():
    """Прод-прогон 2026-08-15: change почти всегда однодневное обстоятельство,
    name — 0 записей на 31 юзера (имя живёт в анкете-профиле). Обе выпилены."""
    from llm.memory_extract import CORE_ENTITY_KEYS

    assert CORE_ENTITY_KEYS == frozenset(
        {"age", "family", "work", "place", "health", "tone"}
    )
    assert "change" not in CORE_ENTITY_KEYS
    assert "name" not in CORE_ENTITY_KEYS


def test_system_prompt_core_json_contract_lists_six_categories():
    """JSON-контракт в промпте перечисляет ровно шесть категорий, без name/change."""
    from llm.memory_extract import _SYSTEM

    assert '"entity_key":"age|family|work|place|health|tone"' in _SYSTEM
    assert "|change|" not in _SYSTEM
    assert '"name|' not in _SYSTEM


@pytest.mark.asyncio
async def test_core_dropped_change_and_name_keys(monkeypatch):
    """Убранные категории режутся кодом как любой неизвестный entity_key."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "change", "text": "Находится в отпуске", "source": "qa_answer"},
        {"entity_key": "name", "text": "Зовут Александр", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


def test_system_prompt_non_dream_and_new_sources():
    """v4: правило про текст обращения (не сон) и оба новых source в контракте."""
    from llm.memory_extract import _SYSTEM

    assert "ТЕКСТ ОБРАЩЕНИЯ" in _SYSTEM
    assert "entry_text" in _SYSTEM
    assert "pre_questions" in _SYSTEM


def test_system_prompt_core_dedup_and_tone_stability():
    """Круг правок 1: core-факт не дублируется в items; tone — только устойчивые
    правила общения, не разовая эмоция; есть few-shot пример на ядровой факт."""
    from llm.memory_extract import _SYSTEM

    # Правило «место ядрового факта — только core, в items не дублировать».
    assert "ТОЛЬКО в core" in _SYSTEM
    assert "НЕ дублируй" in _SYSTEM
    # tone ловит устойчивые предпочтения, а не эмоцию в моменте.
    assert "Разовая эмоция" in _SYSTEM
    # Few-shot пример-антипример по ядровым фактам (по образцу примеров items).
    assert "Айгуль" in _SYSTEM


def test_system_prompt_realm_filter_intact():
    """Гарды v3.2 не ослаблены: запретные realm по-прежнему описаны."""
    from llm.memory_extract import _SYSTEM

    assert "dream_content" in _SYSTEM
    assert "interpretation_echo" in _SYSTEM


def test_system_prompt_has_parts_and_patterns_instructions():
    """v2: в _SYSTEM есть блок про части, названные юзером, и recall-гайд по паттернам."""
    from llm.memory_extract import _SYSTEM

    # Части: нормализованный entity_key по сути роли + слово юзера как алиас в text.
    assert "kind=part" in _SYSTEM
    assert "entity_key" in _SYSTEM
    assert "ревизор" in _SYSTEM  # few-shot с частью, названной самим юзером
    assert "critic-devaluing" in _SYSTEM  # пример нормализованного ключа роли
    # Recall-гайд по паттернам реальной жизни (GATE-пропуски).
    assert "kind=pattern" in _SYSTEM
    assert "самозванец" in _SYSTEM
    # Качество фиксации: референт ассоциации, человек вместо ярлыка ветки.
    assert "референт" in _SYSTEM
    # Realm-фильтр НЕ ослаблен: правила про запретные realm на месте.
    assert "dream_content" in _SYSTEM
    assert "interpretation_echo" in _SYSTEM


def test_system_prompt_has_grounding_rule():
    """v3: правило про маркированные заземляющие вопросы (фикс image_q → память).

    Пара с маркером = юзер отвечает на ПРЯМОЙ вопрос о реальной жизни — извлекать
    как обычно с source=qa_grounding, не считать разговором о сне.
    """
    from data.constants import GROUNDING_QUESTION_MARKER
    from llm.memory_extract import _SYSTEM

    assert GROUNDING_QUESTION_MARKER.strip() in _SYSTEM
    assert "qa_grounding" in _SYSTEM


def test_grounding_marker_constant_shape():
    """Маркер начинается с '[' и заканчивается '] ' — виден в начале вопроса."""
    from data.constants import GROUNDING_QUESTION_MARKER

    assert GROUNDING_QUESTION_MARKER.startswith("[")
    assert GROUNDING_QUESTION_MARKER.endswith("] ")


@pytest.mark.asyncio
async def test_marked_grounding_pair_yields_qa_grounding_fact(monkeypatch):
    """Помеченная image_q-пара: реальный факт (дочь) проходит защиты с source=qa_grounding.

    Регресс дев-смоука: ответы на заземляющие вопросы дропались как dream-talk,
    хотя юзер отвечал про РЕАЛЬНУЮ жизнь.
    """
    from data.constants import GROUNDING_QUESTION_MARKER

    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "person",
         "text": "У юзера есть дочь; тёплые чувства к ней как к своему ребёнку.",
         "realm": "real_life", "source": "qa_grounding"}])
    res = await _run(user_replies=[(
        GROUNDING_QUESTION_MARKER
        + "Что в вашей реальной жизни связано с образом одинокой девочки?",
        "есть ассоциация с моей дочкой, тёплые чувства к своему ребёнку",
    )])
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.source == "qa_grounding"
    assert c.realm == "real_life"
    assert "доч" in c.text


def test_system_prompt_grounding_separates_dream_from_reallife():
    """v3: правило требует ОТДЕЛЯТЬ атрибуты образа сна от реального факта о юзере.

    Дефект: «оставляют одну, плохо обращаются» — про девочку из сна, а не
    про реальную дочь; такое НЕЛЬЗЯ приписывать реальному человеку.
    """
    from llm.memory_extract import _SYSTEM

    # Явно про отделение реального факта от свойств/сценария образа сна.
    assert "напоминает" in _SYSTEM or "ассоциир" in _SYSTEM
    assert "не переноси" in _SYSTEM.lower() or "не приписыв" in _SYSTEM.lower()


@pytest.mark.asyncio
async def test_grounding_strips_dream_image_attributes_daughter(monkeypatch):
    """Дефект: свойства образа сна («родители оставляют одну») НЕ должны
    попадать в реальный факт про дочь. В память идёт только «есть дочь, тёплые чувства».

    Это тест кодового пути: при корректной работе промпта модель вернёт очищенный
    text; проверяем, что защиты его пропускают и он НЕ содержит атрибутов образа.
    """
    from data.constants import GROUNDING_QUESTION_MARKER

    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "person",
         "text": "У юзера есть маленькая дочь; тёплые чувства к ней как к своему ребёнку.",
         "realm": "real_life", "source": "qa_grounding"}])
    res = await _run(user_replies=[(
        GROUNDING_QUESTION_MARKER
        + "Что в вашей реальной жизни связано с образом этой одинокой девочки?",
        "есть ассоциация с моей дочкой, тёплые чувства как к своему ребёнку, "
        "жалко что родители с ней так обращаются и оставляют одну в таком месте",
    )])
    assert len(res.candidates) == 1
    text = res.candidates[0].text.lower()
    assert "доч" in text
    # Атрибуты ОБРАЗА СНА (девочка на балконе) не должны быть в реальном факте.
    for forbidden in ("родител", "оставля", "плохо обращ", "бросают"):
        assert forbidden not in text, f"dream-image attribute leaked: {forbidden!r}"


def test_system_prompt_reserves_qa_grounding_source():
    """v3.2: qa_grounding зарезервирован строго за фактами из МАРКИРОВАННЫХ пар.

    Дефект: v3.1-правило заставляло модель красить в qa_grounding и факты
    из per_session_context, и из обычных Q&A. Правило должно явно это запрещать.
    """
    from llm.memory_extract import _SYSTEM

    assert "ТОЛЬКО" in _SYSTEM  # qa_grounding — только для маркированных
    # Все три источника разведены в правиле source.
    assert "per_session" in _SYSTEM
    assert "qa_answer" in _SYSTEM
    assert "qa_grounding" in _SYSTEM


@pytest.mark.asyncio
async def test_mixed_input_source_per_origin(monkeypatch):
    """Смешанный вход: маркированная grounding-пара + обычная qa-пара + per_session.

    Источник решает LLM; здесь мокаем «правильный» ответ и проверяем, что кодовый
    путь сохраняет per-candidate source (qa_grounding / qa_answer / per_session),
    а не схлопывает всё в один. Поведенческое доказательство — в живом демо.
    """
    from data.constants import GROUNDING_QUESTION_MARKER

    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "person",
         "text": "У юзера есть маленькая дочь; тёплые чувства к ней.",
         "realm": "real_life", "source": "qa_grounding"},
        {"action": "add", "kind": "fact", "text": "Работает врачом в больнице.",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Недавно переехал в Тбилиси.",
         "realm": "real_life", "source": "per_session"}])
    res = await _run(
        user_replies=[
            (GROUNDING_QUESTION_MARKER + "Что связано с образом девочки?",
             "ассоциация с моей дочкой"),
            ("Где в жизни похожая тревога?", "на работе, я врач"),
        ],
        per_session_context="Недавно переехал в Тбилиси, тревожно на новом месте",
    )
    by_text = {c.text: c.source for c in res.candidates}
    # Текст приезжает уже безличным: слово «юзер» вычищается кодом (правка 2026-08-16).
    assert by_text["Есть маленькая дочь; тёплые чувства к ней."] == "qa_grounding"
    assert by_text["Работает врачом в больнице."] == "qa_answer"
    assert by_text["Недавно переехал в Тбилиси."] == "per_session"


@pytest.mark.asyncio
async def test_grounding_strips_dream_scenario_gender(monkeypatch):
    """Дефект: «прожить жизнь в женском теле» — сценарий СНА, не реальный
    факт. В память идёт только «привык быть мужчиной, симпатия к женщинам»."""
    from data.constants import GROUNDING_QUESTION_MARKER

    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact",
         "text": "Юзер привык быть мужчиной и испытывает симпатию к женщинам.",
         "realm": "real_life", "source": "qa_grounding"}])
    res = await _run(user_replies=[(
        GROUNDING_QUESTION_MARKER
        + "Какие мысли и чувства у вас возникают, когда вы думаете о возможности "
        "прожить жизнь в женском теле?",
        "меня такая перспектива пугает, я привык быть мужчиной и испытывать симпатию "
        "к женщинам",
    )])
    assert len(res.candidates) == 1
    text = res.candidates[0].text.lower()
    assert "мужчин" in text
    # Сценарий СНА (стать/жить женщиной) не должен кодироваться как реальный факт.
    assert "женском теле" not in text
    assert "стать женщиной" not in text


@pytest.mark.asyncio
async def test_unmarked_dream_reflection_still_dropped(monkeypatch):
    """Регресс: обычная (НЕмаркированная) Q&A-рефлексия о сне по-прежнему фильтруется.

    Общий фильтр «сон ≠ правда» для немаркированных пар не ослаблен.
    """
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact",
         "text": "Голлумы преследовали юзера по тёмному коридору",
         "realm": "dream_content", "source": "qa_answer"}])
    res = await _run(user_replies=[(
        "Что вы чувствовали, когда голлумы приближались?",
        "во сне я бежал по коридору, а голлумы догоняли",
    )])
    assert res.candidates == []


@pytest.mark.asyncio
async def test_user_named_part_passes_guards(monkeypatch):
    """Часть, названная юзером: kind=part с entity_key и словом юзера в text проходит защиты."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "part",
         "text": "Внутренний критик, который обесценивает результаты; "
                 "похожую рабочую сторону юзер зовёт «ревизор».",
         "realm": "real_life", "entity_key": "critic-devaluing", "source": "qa_answer"}])
    res = await _run(user_replies=[(None, "Она как ревизор, обесценивает что я делаю")])
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.kind == "part"
    assert c.entity_key == "critic-devaluing"
    assert "ревизор" in c.text  # слово юзера сохранено как алиас


@pytest.mark.asyncio
async def test_drops_dream_content_realm(monkeypatch):
    # LLM вернул кандидата realm=dream_content → он не должен пройти
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Вампир на чердаке",
         "realm": "dream_content", "source": "qa_answer"}])
    res = await _run(user_replies=[(None, "Вампир на чердаке")])
    assert res.candidates == []


@pytest.mark.asyncio
async def test_drops_interpretation_echo_realm(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Согласен с гипотезой",
         "realm": "interpretation_echo", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_drops_dream_marker_in_text(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "во сне я летал над городом",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []  # стоп-маркер «во сне»


@pytest.mark.asyncio
async def test_drops_unknown_kind(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "vibe", "text": "Что-то странное",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_drops_empty_and_oversized_text(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "   ",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "х" * 501,
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_keeps_real_life_and_association(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "attitude", "text": "К подросткам держится подальше",
         "realm": "association", "source": "qa_grounding"},
        {"action": "add", "kind": "fact", "text": "Прыгал с тарзанки на тренинге",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert {c.text for c in res.candidates} == {
        "К подросткам держится подальше", "Прыгал с тарзанки на тренинге"}


@pytest.mark.asyncio
async def test_caps_at_max_items(monkeypatch):
    # v2: cap поднят 6→8 (recall-тюнинг), сверх лимита — отбрасывается.
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": f"факт {i}", "realm": "real_life",
         "source": "qa_answer"} for i in range(12)])
    res = await _run()
    assert len(res.candidates) == 8


@pytest.mark.asyncio
async def test_dedup_by_entity_key_and_normalized_text(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Работает врачом",
         "realm": "real_life", "entity_key": "job", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Сменил профессию",
         "realm": "real_life", "entity_key": "job", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Любит  горы",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "любит горы",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 2
    assert res.candidates[0].entity_key == "job"


@pytest.mark.asyncio
async def test_overlength_entity_key_dropped_unit_kept(monkeypatch):
    """entity_key длиннее колонки String(64) обнуляется, сама единица сохраняется.

    Иначе INSERT падает StringDataRightTruncation на Postgres и травит общую
    транзакцию запроса (SQLite в тестах длину не проверяет).

    Для kind=part ключ теперь выводится из текста (пункт 6 прод-правок) — этот
    случай проверяет test_overlength_part_key_gets_derived_replacement.
    """
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Внутренний критик",
         "realm": "real_life", "entity_key": "x" * 65, "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].entity_key is None
    assert res.candidates[0].text == "Внутренний критик"


@pytest.mark.asyncio
async def test_overlength_or_unknown_source_clamped(monkeypatch):
    """source вне allow-list или длиннее String(24) клампится в qa_answer."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Живёт в Тбилиси",
         "realm": "real_life", "source": "s" * 25},
        {"action": "add", "kind": "fact", "text": "Работает врачом",
         "realm": "real_life", "source": "weird_source"}])
    res = await _run()
    assert len(res.candidates) == 2
    assert [c.source for c in res.candidates] == ["qa_answer", "qa_answer"]


@pytest.mark.asyncio
async def test_kind_association_aliased_to_attitude(monkeypatch):
    """Модель путает realm с kind: kind=association не дропается, а становится attitude."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "association", "text": "Вампиров знает только по фильмам",
         "realm": "association", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].kind == "attitude"


def test_build_user_message_tolerates_none_reply():
    """Контракт «никогда не бросает»: (None, None)-пара реплик не роняет билдер."""
    from llm.memory_extract import _build_user_message

    msg = _build_user_message([(None, None)], [], "", [])
    assert isinstance(msg, str)


@pytest.mark.asyncio
async def test_confirm_requires_valid_target(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "confirm", "target_id": 999, "kind": "fact", "text": "x",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[ExistingItemView(1, "fact", "y", None)])
    assert res.candidates == []  # target_id 999 не существует → отброшен


@pytest.mark.asyncio
async def test_confirm_with_valid_target_passes(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "confirm", "target_id": 1, "kind": "fact", "text": "Работает врачом",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[ExistingItemView(1, "fact", "Работает врачом", None)])
    assert len(res.candidates) == 1
    assert res.candidates[0].action == "confirm"
    assert res.candidates[0].target_id == 1


@pytest.mark.asyncio
async def test_supersede_requires_valid_target(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "supersede", "target_id": None, "kind": "fact", "text": "Вышел на новую работу",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[ExistingItemView(1, "fact", "Ищет работу", None)])
    assert res.candidates == []  # target_id отсутствует → отброшен


@pytest.mark.asyncio
async def test_usage_tokens_passed_through(monkeypatch):
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Живёт в Тбилиси",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.input_tokens == 11
    assert res.output_tokens == 7
    assert res.reject_reason is None


@pytest.mark.asyncio
async def test_raises_classifier_output_ceiling(monkeypatch):
    """v4 отдаёт ДВА списка (core + items): дефолтных 600 токенов не хватает.

    Обрыв JSON по потолку — это no-op на всю сессию (оба списка теряются молча),
    поэтому потолок прокинут параметром и поднят именно для экстрактора.
    """
    from llm.classifier import _CLASSIFIER_MAX_TOKENS
    from llm.memory_extract import _EXTRACT_MAX_TOKENS

    spy = _mock_classify(monkeypatch, items=[])
    await _run()
    assert spy.await_args.kwargs["max_tokens"] == _EXTRACT_MAX_TOKENS
    # Худший случай v4 ≈ 8 core * 50 ток. + 8 items * 58 ток. + обёртка JSON.
    assert _EXTRACT_MAX_TOKENS >= 8 * 50 + 8 * 58
    assert _EXTRACT_MAX_TOKENS > _CLASSIFIER_MAX_TOKENS


@pytest.mark.asyncio
async def test_parse_failure_is_noop(monkeypatch):
    monkeypatch.setattr("llm.memory_extract.classify",
                        AsyncMock(side_effect=ClassifierParseError()))
    res = await _run()
    assert res.candidates == [] and res.input_tokens == 0
    assert res.reject_reason == "parse_failed"


@pytest.mark.asyncio
async def test_call_failure_is_noop(monkeypatch):
    monkeypatch.setattr("llm.memory_extract.classify",
                        AsyncMock(side_effect=RuntimeError("network down")))
    res = await _run()
    assert res.candidates == []
    assert res.reject_reason == "call_failed"


@pytest.mark.asyncio
async def test_empty_input_no_call(monkeypatch):
    spy = AsyncMock()
    monkeypatch.setattr("llm.memory_extract.classify", spy)
    res = await _run(user_replies=[], per_session_context="")
    spy.assert_not_awaited()
    assert res.reject_reason == "empty_input"


@pytest.mark.asyncio
async def test_per_session_only_still_calls(monkeypatch):
    spy = _mock_classify(monkeypatch, items=[])
    res = await _run(user_replies=[], per_session_context="Недавно переехал в Тбилиси")
    spy.assert_awaited_once()
    assert res.candidates == []
    assert res.reject_reason is None


def test_schema_tolerates_confirm_without_kind_text_realm():
    """Регресс: confirm-item без kind/text/realm парсится (иначе роняет всю пачку)."""
    from llm.memory_extract import _Extract

    parsed = _Extract.model_validate({"items": [{"action": "confirm", "target_id": 5}]})
    assert parsed.items[0].action == "confirm"
    assert parsed.items[0].target_id == 5


@pytest.mark.asyncio
async def test_confirm_without_fields_survives_and_confirms(monkeypatch):
    """confirm без полей → confirm-кандидат по target_id (не дропается, не роняет пачку)."""
    _mock_classify(monkeypatch, items=[
        {"action": "confirm", "target_id": 11},
        {"action": "confirm", "target_id": 8},
    ])
    existing = [
        ExistingItemView(11, "fact", "x", None),
        ExistingItemView(8, "part", "y", "k"),
    ]
    res = await _run(existing_items=existing)
    assert res.reject_reason is None
    assert {c.target_id for c in res.candidates if c.action == "confirm"} == {11, 8}


@pytest.mark.asyncio
async def test_confirm_does_not_drop_new_add(monkeypatch):
    """Регресс (баг смоука): confirm существующей + новый add — add НЕ теряется."""
    _mock_classify(monkeypatch, items=[
        {"action": "confirm", "target_id": 11},
        {"action": "add", "kind": "fact",
         "text": "Похудел на 20 кг по методике Олега, 3.5 года в его проекте",
         "realm": "real_life", "source": "qa_answer"},
    ])
    res = await _run(existing_items=[ExistingItemView(11, "person", "Олег", None)])
    assert any(c.action == "confirm" and c.target_id == 11 for c in res.candidates)
    assert any(c.action == "add" and "20 кг" in c.text for c in res.candidates)


@pytest.mark.asyncio
async def test_confirm_invalid_target_dropped(monkeypatch):
    """confirm на несуществующий id — отбрасывается, но пачку не роняет."""
    _mock_classify(monkeypatch, items=[{"action": "confirm", "target_id": 999}])
    res = await _run(existing_items=[ExistingItemView(11, "fact", "x", None)])
    assert res.candidates == []
    assert res.reject_reason is None


# === Память v2: ядровые факты (kind="core") ===


@pytest.mark.asyncio
async def test_core_candidate_normalized(monkeypatch):
    """Валидный core-факт проходит: kind=core, realm принудительно real_life,
    entity_key нормализован к нижнему регистру, текст обрезан от пробелов."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "Work", "text": " Работает продакт-менеджером в IT ",
         "source": "per_session"}])
    res = await _run()
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.kind == "core"
    assert c.action == "add"
    assert c.entity_key == "work"
    assert c.realm == "real_life"
    assert c.source == "per_session"
    assert c.text == "Работает продакт-менеджером в IT"


@pytest.mark.asyncio
async def test_core_unknown_entity_key_dropped(monkeypatch):
    """entity_key вне закрытого списка категорий — единица режется целиком."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "salary", "text": "Зарабатывает много", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_core_kind_inside_items_rejected(monkeypatch):
    """kind='core' в списке items не проходит: core идёт ТОЛЬКО через список core."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "core", "text": "Работает в найме",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_core_dedup_same_category(monkeypatch):
    """Две записи одной категории за проход — остаётся первая."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "work", "text": "Работает в найме", "source": "qa_answer"},
        {"entity_key": "work", "text": "Пилит петпроект", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].text == "Работает в найме"


@pytest.mark.asyncio
async def test_core_counts_towards_shared_limit(monkeypatch):
    """Общий лимит 8 ново-хранящих единиц: 5 core + 6 items-add → 8 всего (5 core + 3 add)."""
    core = [
        {"entity_key": k, "text": f"Ядровый факт {k}", "source": "qa_answer"}
        for k in ("age", "family", "work", "place", "health")
    ]
    items = [
        {"action": "add", "kind": "fact", "text": f"Обычный факт номер {i}",
         "realm": "real_life", "source": "qa_answer"}
        for i in range(6)
    ]
    _mock_classify(monkeypatch, items=items, core=core)
    res = await _run()
    assert len(res.candidates) == 8
    kinds = [c.kind for c in res.candidates]
    assert kinds[:5] == ["core"] * 5
    assert kinds[5:] == ["fact"] * 3


@pytest.mark.asyncio
async def test_core_dream_marker_rejected(monkeypatch):
    """Эшелон 3 «сон ≠ правда» действует и на core (дефолтный entry_kind=dream)."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "work", "text": "Во сне у него была работа мечты",
         "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_core_empty_is_normal(monkeypatch):
    """Пустой core — нормальный случай, не ошибка: no-op не ставится, usage считается."""
    _mock_classify(monkeypatch, items=[], core=[])
    res = await _run()
    assert res.candidates == []
    assert res.reject_reason is None
    assert res.input_tokens == 11  # вызов модели состоялся (_FakeClassifierResult)


# === Память v2: источники entry_text / pre_questions и не-сновые типы ===


@pytest.mark.asyncio
async def test_source_entry_text_accepted(monkeypatch):
    """source='entry_text' проходит белый список, не переписывается в qa_answer."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Недавно сократили на работе",
         "realm": "real_life", "source": "entry_text"}])
    res = await _run(
        entry_kind="external_situation", entry_text="меня сократили, я в растерянности"
    )
    assert len(res.candidates) == 1
    assert res.candidates[0].source == "entry_text"


@pytest.mark.asyncio
async def test_entry_text_block_in_user_message(monkeypatch):
    """Текст обращения попадает в user-сообщение отдельным помеченным блоком."""
    spy = _mock_classify(monkeypatch, items=[])
    await _run(entry_kind="external_situation", entry_text="конфликт с начальником")
    user_msg = spy.await_args.kwargs["messages"][1]["content"]
    assert "ТЕКСТ ОБРАЩЕНИЯ" in user_msg
    assert "конфликт с начальником" in user_msg
    assert 'source="entry_text"' in user_msg


@pytest.mark.asyncio
async def test_dream_marker_not_rejected_for_non_dream(monkeypatch):
    """Для ситуации/конфликта фильтр «сон ≠ правда» неприменим: маркер сна не режет."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact",
         "text": "После этой ссоры плохо спит, снилось всякое",
         "realm": "real_life", "source": "entry_text"}])
    res = await _run(entry_kind="external_situation", entry_text="ссора с женой")
    assert len(res.candidates) == 1


@pytest.mark.asyncio
async def test_dream_marker_still_rejected_for_dream(monkeypatch):
    """Для снов (дефолтный entry_kind) правило остаётся прежним."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Во сне он летал над городом",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_pre_questions_block_and_source(monkeypatch):
    """Ответы на уточняющие вопросы идут отдельным блоком; source='pre_questions' проходит."""
    spy = _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Работает из дома, двое детей",
         "realm": "real_life", "source": "pre_questions"}])
    res = await _run(
        user_replies=[],
        pre_question_replies=[
            ("Как устроен ваш обычный день?", "Работаю из дома, у меня двое детей")
        ],
    )
    user_msg = spy.await_args.kwargs["messages"][1]["content"]
    assert "ОТВЕТЫ НА УТОЧНЯЮЩИЕ ВОПРОСЫ" in user_msg
    assert 'source="pre_questions"' in user_msg
    assert res.candidates[0].source == "pre_questions"


@pytest.mark.asyncio
async def test_entry_text_alone_is_not_empty_input(monkeypatch):
    """Только текст обращения (без реплик и per_session) — валидный вход, модель зовём."""
    spy = _mock_classify(monkeypatch, items=[])
    res = await _run(
        user_replies=[], entry_kind="internal_conflict", entry_text="прокрастинирую"
    )
    spy.assert_awaited_once()
    assert res.reject_reason is None


@pytest.mark.asyncio
async def test_default_dream_call_has_no_new_blocks(monkeypatch):
    """Инвариант: дефолтный (сновый) вызов — user-сообщение БЕЗ новых блоков v2.

    Регресс-щит для части B: если будущий рефакторинг уберёт условия
    `if entry:` / `if pre_question_replies:` или поменяет дефолты, разбор снов
    на проде получит лишний текст в LLM-контексте — этот тест обязан упасть.
    """
    spy = _mock_classify(monkeypatch, items=[])
    await _run(
        user_replies=[(None, "обычная реплика")],
        per_session_context="Недавно переехал",
    )
    user_msg = spy.await_args.kwargs["messages"][1]["content"]
    assert "ТЕКСТ ОБРАЩЕНИЯ" not in user_msg
    assert "ОТВЕТЫ НА УТОЧНЯЮЩИЕ ВОПРОСЫ" not in user_msg
    # И меток новых источников в сновом сообщении тоже нет.
    assert 'source="entry_text"' not in user_msg
    assert 'source="pre_questions"' not in user_msg


@pytest.mark.asyncio
async def test_empty_pre_question_replies_list_like_absent(monkeypatch):
    """pre_question_replies=[] (пустой список, не None) ведёт себя как отсутствие:
    блока в сообщении нет, а один пустой список НЕ делает вход валидным."""
    spy = _mock_classify(monkeypatch, items=[])
    await _run(pre_question_replies=[])
    user_msg = spy.await_args.kwargs["messages"][1]["content"]
    assert "ОТВЕТЫ НА УТОЧНЯЮЩИЕ ВОПРОСЫ" not in user_msg

    # Только пустой список (без реплик/per_session/entry) → empty_input, модели нет.
    spy2 = AsyncMock()
    monkeypatch.setattr("llm.memory_extract.classify", spy2)
    res = await _run(user_replies=[], pre_question_replies=[])
    spy2.assert_not_awaited()
    assert res.reject_reason == "empty_input"


def test_system_prompt_core_self_sufficiency_rule():
    """Фикс живого смоука 2026-08-15: в family уехал огрызок «дочь юзера младше
    дочки бывшего тренера на год» — тренер был персонажем СНА, факт бессмысленен
    вне того сна и занял единственную active-ячейку категории. Ядровой факт
    обязан быть самодостаточным: без контекста сна и без ссылок на третьих лиц."""
    from llm.memory_extract import _SYSTEM

    assert "САМОДОСТАТОЧН" in _SYSTEM
    # Пример-антипример в стиле промпта: огрызок со сравнением — плохо.
    assert "младше дочки тренера" in _SYSTEM
    assert "Есть дочь 3 лет" in _SYSTEM
    # Нет самодостаточной формулировки → в core не класть вовсе (items — можно).
    assert "в core НЕ клади" in _SYSTEM


# === Правки по прод-прогону 2026-08-15 ===


def test_system_prompt_core_lifetime_test():
    """Пункт 1: тест на срок жизни ядрового факта.

    Разовое событие через месяц перестаёт быть правдой → это не ядро. Правка 6:
    антипримеры заданы КАТЕГОРИЯМИ («занятие на один вечер, принятое за профессию»),
    а не дословными фразами прошлого прогона — те возвращались в выдаче якорями.
    """
    from llm.memory_extract import _SYSTEM

    assert "через месяц" in _SYSTEM
    assert "на один вечер" in _SYSTEM
    assert "бюрократическ" in _SYSTEM.lower()
    assert "обстоятельства недели" in _SYSTEM.lower()


def test_system_prompt_replaces_outdated_rule():
    """Пункт 1: флаг обновления ставится только при ПРОТИВОРЕЧИИ прежнему факту."""
    from llm.memory_extract import _SYSTEM

    assert "replaces_outdated" in _SYSTEM
    assert "ПРОТИВОРЕЧ" in _SYSTEM.upper()
    # Явные примеры «настоящего» обновления.
    assert "сменил" in _SYSTEM.lower() or "смена работы" in _SYSTEM.lower()
    assert '"replaces_outdated":' in _SYSTEM


@pytest.mark.asyncio
async def test_core_replaces_outdated_defaults_false(monkeypatch):
    """Флага в ответе нет → кандидат НЕ помечен обновлением (ядро не разрушаем)."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "work", "text": "Работает в саду", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].replaces_outdated is False


@pytest.mark.asyncio
async def test_core_replaces_outdated_passed_through(monkeypatch):
    """Флаг true доезжает до кандидата — на нём висит право вытеснить ячейку."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "place", "text": "Переехал в Белград", "source": "qa_answer",
         "replaces_outdated": True}])
    res = await _run()
    assert res.candidates[0].replaces_outdated is True


def test_system_prompt_empty_result_is_welcome():
    """Пункт 2: пустой результат — норма и желанный исход; запрет задан КАТЕГОРИЯМИ."""
    from llm.memory_extract import _SYSTEM

    assert "ЖЕЛАН" in _SYSTEM.upper()
    assert "общеизвестн" in _SYSTEM.lower()
    # Категории мусора вместо дословных фраз (см. test_..._are_categories_not_quotes).
    assert "предпочтения в еде" in _SYSTEM
    assert "ЗАПРЕЩЕНО записывать" in _SYSTEM


def test_system_prompt_third_person_rule():
    """Правка 3 (2026-08-15): чужая история — события ДРУГОГО человека, а тяжёлый
    факт о самом юзере записывается как его собственный.

    Прежний антипример «уехала в Болгарию после рецидива зависимости» был НЕВЕРЕН:
    в сырых репликах юзер говорит «моего рецидива», «я ездила» — это факт О СЕБЕ,
    и антипример учил модель выбрасывать настоящие факты о человеке.
    """
    from llm.memory_extract import _SYSTEM

    assert "ТРЕТЬИ ЛИЦА" in _SYSTEM
    # Корректные примеры чужой истории — событие с ДРУГИМ человеком в подлежащем.
    assert "подруга развелась" in _SYSTEM
    assert "умер отец" in _SYSTEM
    # Ошибочный якорь выпилен.
    assert "Болгари" not in _SYSTEM
    # Тяжёлый/стыдный факт о самом юзере — записывать как его собственный.
    assert "рецидив" in _SYSTEM.lower()
    assert "стыдн" in _SYSTEM.lower()
    assert "«мой рецидив»" in _SYSTEM or "мой рецидив" in _SYSTEM


def test_system_prompt_dream_paraphrase_antiexamples():
    """Пункт 3: перефразированная сцена сна — тоже сон; запрет задан категориями."""
    from llm.memory_extract import _SYSTEM

    assert "СОН — НЕ РЕАЛЬНОСТЬ" in _SYSTEM
    assert "сцен" in _SYSTEM.lower()
    # Категории вместо дословных сцен прошлого прогона.
    assert "предмет из сна" in _SYSTEM
    assert "эмоция, адресованная образу" in _SYSTEM


def test_system_prompt_antiexamples_are_categories_not_quotes():
    """Правка 6 (2026-08-15): дословные антипримеры работали ЯКОРЯМИ — модель
    возвращала их в выдаче («не любит сгущенку», «ВКонтакте принадлежит мейл ру»).
    В промпте остаются КАТЕГОРИИ нежелательного, а не конкретные фразы."""
    from llm.memory_extract import _SYSTEM

    anchors = (
        "сгущенк",
        "ВКонтакте",
        "мейл ру",
        "мягкую кровать",
        "мягкая кровать",
        "поезд метро",
        "неподходящую обувь",
        "работает в саду",
        "документы на визу",
        "ресторане на высоте",
        "микроавтобус",
        "весло",
        "две девушки",
        "няня",
    )
    for anchor in anchors:
        assert anchor not in _SYSTEM, f"дословный якорь остался в промпте: {anchor!r}"
    # На их месте — названия категорий.
    assert "предпочтения в еде" in _SYSTEM
    assert "общеизвестные факты" in _SYSTEM
    assert "бюрократическ" in _SYSTEM.lower()


def test_dream_marker_catches_paraphrases():
    """Пункт 3: расширенный кодовый гард ловит формы, которые проходили в проде."""
    from llm.memory_extract import _DREAM_MARKER_RE

    caught = [
        "Юзер помнит, что сон о взрыве был очень ярким",
        "Юзер испытывает самые нежные эмоции к образу двух девушек в постели",
        "В сновидении юзер оказался на крыше",
        "Сцена сна повторяется у юзера регулярно",
        "Юзеру снилась мать",
        "Приснившийся город показался знакомым",
        "Сон про поезд",
    ]
    for text in caught:
        assert _DREAM_MARKER_RE.search(text), f"не отрезано: {text!r}"


def test_dream_marker_keeps_real_life_texts():
    """Пункт 3: расширение не режет законные тексты о реальной жизни."""
    from llm.memory_extract import _DREAM_MARKER_RE

    kept = [
        "Работает врачом в больнице",
        "Ведёт здоровый образ жизни, бегает по утрам",
        "Таким образом юзер справляется со стрессом",
        "Работает в сфере образования",
        "Юзеру не хватает сна из-за маленького ребёнка",
        "Живёт в Тбилиси, учит испанский",
        "Внутренний критик, который обесценивает результаты",
    ]
    for text in kept:
        assert not _DREAM_MARKER_RE.search(text), f"ложный срез: {text!r}"


@pytest.mark.asyncio
async def test_core_dream_paraphrase_dropped(monkeypatch):
    """Прод-дефект: «сон о взрыве» в ядровой записи режется кодовым гардом."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "health", "text": "Юзер помнит, что сон о взрыве был очень ярким",
         "source": "qa_answer"}])
    res = await _run()
    assert res.candidates == []


@pytest.mark.asyncio
async def test_dedup_between_core_and_items(monkeypatch):
    """Пункт 5: один и тот же факт в core и в items — остаётся только ядровая запись.

    Прод: у одного юзера «выиграла суд» записано и core work, и обычным fact.
    """
    _mock_classify(
        monkeypatch,
        core=[{"entity_key": "work", "text": "Юзер выиграл судебное дело",
               "source": "qa_answer"}],
        items=[{"action": "add", "kind": "fact", "text": "юзер  Выиграл судебное  дело",
                "realm": "real_life", "source": "qa_answer"}],
    )
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].kind == "core"


@pytest.mark.asyncio
async def test_dedup_inside_items_ignores_entity_key(monkeypatch):
    """Пункт 5: одинаковый текст под разными entity_key — всё равно один кандидат."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "part", "text": "Учит испанский",
         "realm": "real_life", "entity_key": "serbian", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "учит испанский.",
         "realm": "real_life", "entity_key": "language", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1


@pytest.mark.asyncio
async def test_duplicate_of_existing_item_becomes_confirm(monkeypatch):
    """Пункт 5: текст совпал с уже активной записью → confirm, а не вторая строка."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Тревога из-за необходимости скрывать правду",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[
        ExistingItemView(7, "fact", "тревога из-за необходимости скрывать правду.", None)])
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.action == "confirm"
    assert c.target_id == 7


@pytest.mark.asyncio
async def test_core_duplicate_of_existing_item_becomes_confirm(monkeypatch):
    """Пункт 5: кросс-сессионный дубль ядрового факта тоже схлопывается в confirm."""
    _mock_classify(monkeypatch, items=[], core=[
        {"entity_key": "work", "text": "Работает программистом", "source": "qa_answer"}])
    res = await _run(existing_items=[
        ExistingItemView(3, "fact", "Работает  программистом", None)])
    assert len(res.candidates) == 1
    assert res.candidates[0].action == "confirm"
    assert res.candidates[0].target_id == 3


def test_system_prompt_kind_distinctions():
    """Пункт 6: у каждого вида внятное определение и пример; fact — крайний случай."""
    from llm.memory_extract import _SYSTEM

    for kind in ("pattern", "person", "episode", "attitude", "part", "fact"):
        assert kind in _SYSTEM
    assert "ничего из перечисленного не подходит" in _SYSTEM
    # entity_key для части обязателен.
    assert "ОБЯЗАТЕЛЕН" in _SYSTEM


def test_slug_from_text_is_deterministic_latin():
    """Пункт 6: детерминированный латинский слаг из русского текста, в лимите длины."""
    from llm.memory_extract import _MAX_ENTITY_KEY, _slug_from_text

    slug = _slug_from_text("Сторона, готовая к риску")
    assert slug == _slug_from_text("Сторона, готовая к риску")
    assert slug and slug.isascii()
    assert len(slug) <= _MAX_ENTITY_KEY
    assert slug.strip("-") == slug
    # Разные части — разные ключи (иначе схлопнутся в одну строку памяти).
    assert slug != _slug_from_text("Сторона, сдерживающая переживания")
    # Длинный текст обрезан по лимиту колонки.
    assert len(_slug_from_text("оч" * 200)) <= _MAX_ENTITY_KEY


@pytest.mark.asyncio
async def test_part_without_key_gets_derived_key(monkeypatch):
    """Пункт 6: часть без entity_key получает выведенный ключ — матчинг возможен.

    Прод: у одного юзера три части одной сессии добавлены без ключей.
    """
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "part", "text": "Сторона, готовая к риску",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "part", "text": "Сторона, сдерживающая переживания",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    keys = [c.entity_key for c in res.candidates]
    assert all(k for k in keys), "часть без ключа не матчится между сессиями"
    assert len(set(keys)) == 2


@pytest.mark.asyncio
async def test_part_keeps_model_key_and_other_kinds_untouched(monkeypatch):
    """Ключ от модели не переписывается; не-part без ключа так и остаётся без него."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "part", "text": "Внутренний критик",
         "realm": "real_life", "entity_key": "critic-devaluing", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Живёт в Тбилиси",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    by_kind = {c.kind: c.entity_key for c in res.candidates}
    assert by_kind["part"] == "critic-devaluing"
    assert by_kind["fact"] is None


@pytest.mark.asyncio
async def test_overlength_part_key_gets_derived_replacement(monkeypatch):
    """Слишком длинный ключ части заменяется выведенным, а не теряется совсем."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "part", "text": "Сторона, готовая к риску",
         "realm": "real_life", "entity_key": "x" * 65, "source": "qa_answer"}])
    from llm.memory_extract import _MAX_ENTITY_KEY

    res = await _run()
    key = res.candidates[0].entity_key
    assert key and key != "x" * 65
    assert len(key) <= _MAX_ENTITY_KEY


def test_system_prompt_genderless_wording():
    """Пункт 7 + правка 4 (2026-08-15): род юзера неизвестен, а формулировки шли в
    мужском («Юзер обрадовался, что уехал», «Подруга отвезла его в больницу»).

    Усилено: рецепт переписывания (настоящее время + отглагольные конструкции),
    явный запрет «он/она/его/её» про самого юзера, запрет первого лица и пары
    «плохо → хорошо» на реальных примерах прогона.
    """
    from llm.memory_extract import _SYSTEM

    assert "БЕЗЛИЧНЫЕ ФОРМУЛИРОВКИ" in _SYSTEM
    assert "род" in _SYSTEM.lower()
    # Рецепт переписывания.
    assert "настоящее время" in _SYSTEM
    assert "отглагольное существительное" in _SYSTEM
    # Явный запрет местоимений про самого юзера.
    for pronoun in ("«он»", "«она»", "«его»", "«её»"):
        assert pronoun in _SYSTEM, f"нет запрета местоимения {pronoun}"
    # Запрет первого лица (дефект gpt-5-mini: «Я любила этого мужчину…»).
    assert "первое лицо" in _SYSTEM.lower()
    assert "Юзер была" in _SYSTEM
    # Пары «плохо → хорошо», не меньше четырёх.
    assert _SYSTEM.count(" → ") >= 4
    assert "обрадовался" in _SYSTEM
    # Правка 2026-08-16: подлежащее «Юзер» из образцов убрано — оно само по себе
    # мужского рода и тянет согласование (см. test_system_prompt_forbids_user_subject).
    assert "Рад отъезду" in _SYSTEM
    assert "Подруга отвезла в больницу" in _SYSTEM


# === Правка 1 (2026-08-16): слово «Юзер» из формулировок памяти убрано ===


def test_system_prompt_forbids_user_subject():
    """Корень проблемы рода — само слово «юзер»: оно мужского рода и тянет за собой
    согласование («Юзер оплатил» про женщину). Записи по определению про юзера,
    подлежащее не нужно — промпт обязан требовать формулировок без него.
    """
    from llm.memory_extract import _SYSTEM

    assert "БЕЗ СЛОВА «ЮЗЕР»" in _SYSTEM
    assert "подлежащее не нужно" in _SYSTEM


def test_system_prompt_record_examples_have_no_user_word():
    """Механизм, а не уговоры: ни один ОБРАЗЕЦ записи в промпте не содержит «юзер».

    Модель копирует форму примеров — пока в них жило «У юзера есть дочь», правило
    «пиши без юзера» проигрывало образцу.
    """
    import re

    from llm.memory_extract import _SYSTEM

    texts = re.findall(r'"text":"(.*?)"', _SYSTEM)
    assert texts, "в промпте должны быть образцы записей"
    bad = [t for t in texts if "юзер" in t.lower()]
    assert not bad, f"образцы записей со словом «юзер»: {bad}"


def test_depersonalize_removes_user_as_subject():
    """Кодовая страховка: слово «юзер» вычищается из текста записи в любой позиции."""
    from llm.memory_extract import _depersonalize

    assert _depersonalize("Юзер работает программистом.") == "Работает программистом."
    assert _depersonalize("У юзера есть дочь 2,5 года.") == "Есть дочь 2,5 года."
    assert _depersonalize("Подруга отвезла юзера в больницу.") == "Подруга отвезла в больницу."
    assert (
        _depersonalize("Есть дочь; у юзера тёплые чувства к ней.")
        == "Есть дочь; тёплые чувства к ней."
    )
    assert _depersonalize("Мать юзера живёт в Москве") == "Мать живёт в Москве"


def test_depersonalize_leaves_clean_text_untouched():
    """Текст без слова «юзер» не трогаем — включая слова, начинающиеся так же."""
    from llm.memory_extract import _depersonalize

    assert _depersonalize("Работает нотариусом в частной конторе") == (
        "Работает нотариусом в частной конторе"
    )
    assert _depersonalize("") == ""


@pytest.mark.asyncio
async def test_candidates_are_depersonalized(monkeypatch):
    """Сквозь экстрактор: ни в core, ни в items слова «юзер» не остаётся."""
    _mock_classify(
        monkeypatch,
        core=[{"entity_key": "work", "text": "Юзер работает продакт-менеджером",
               "source": "qa_answer"}],
        items=[{"action": "add", "kind": "person",
                "text": "У юзера есть маленькая дочь; тёплые чувства к ней.",
                "realm": "real_life", "source": "qa_answer"}],
    )
    res = await _run()
    assert [c.text for c in res.candidates] == [
        "Работает продакт-менеджером",
        "Есть маленькая дочь; тёплые чувства к ней.",
    ]


# === Правка 5 (2026-08-15): перефразный дедуп в коде ===


def test_significant_words_drops_stopwords_and_short():
    """Нормализация: регистр/пунктуация/пробелы, стоп-слова и короткие слова прочь.

    Правка 2 (2026-08-16): значимое слово усекается до основы (первые буквы) —
    иначе «состоит» и «состою» для алгоритма разные слова.
    """
    from llm.memory_extract import _significant_words

    assert _significant_words("  Юзер   СОСТОИТ в браке!! ") == frozenset(
        {"состо", "браке"}
    )
    # «юзер»/«есть»/«это» — служебный шум, он есть почти в каждой записи.
    assert _significant_words("Есть это у юзера") == frozenset()
    # Короткие числа значимы: «дочь 2 года» и «дочь 7 лет» — разные факты.
    assert "2" in _significant_words("У юзера дочь 2 года")


def test_paraphrase_threshold_is_conservative():
    """Порог 3/4 значимых слов: один отличающийся смысловой корень разводит записи."""
    from llm.memory_extract import _is_paraphrase, _significant_words

    a = _significant_words("Учит испанский язык каждый день")
    b = _significant_words("Учит английский язык каждый день")
    assert not _is_paraphrase(a, b)
    # Меньше трёх значимых слов — перефразный дедуп не применяется вовсе.
    short_a = _significant_words("Живёт одна")
    short_b = _significant_words("Живёт один")
    assert not _is_paraphrase(short_a, short_b)


def test_prefix_dedup_survives_russian_inflection():
    """Правка 2 (2026-08-16): дедуп по началам слов оживляет перефразный слой.

    Прод-прогон 2026-08-15: сработал один раз за весь прогон — русские словоформы
    («состоит»/«состою», «воспитывает»/«воспитываю») сравнивались как разные слова
    и пересечение проваливалось ниже порога.
    """
    from llm.memory_extract import _is_paraphrase, _significant_words

    a = _significant_words("Состоит в браке, воспитывает дочь")
    b = _significant_words("Состою в браке, воспитываю дочь")
    assert _is_paraphrase(a, b)


def test_prefix_dedup_does_not_merge_different_facts():
    """Страховка усечения: разные корни в начале слова по-прежнему разводят записи."""
    from llm.memory_extract import _is_paraphrase, _significant_words

    a = _significant_words("Мать живёт в Москве, отношения тёплые")
    b = _significant_words("Отец живёт в Москве, отношения тёплые")
    assert not _is_paraphrase(a, b)


@pytest.mark.asyncio
async def test_paraphrase_dedup_inside_items(monkeypatch):
    """Прод: у одного юзера «Состоит в браке…» приехало тремя перефразами."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Состоит в браке, воспитывает дочь",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact",
         "text": "Воспитывает дочь; состоит в браке с мужем.",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 1


@pytest.mark.asyncio
async def test_paraphrase_dedup_between_core_and_items(monkeypatch):
    """Перефраз ядрового факта в items не проходит: остаётся только ядровая запись."""
    _mock_classify(
        monkeypatch,
        core=[{"entity_key": "work",
               "text": "Работает на двух работах, обеспечивает семью",
               "source": "qa_answer"}],
        items=[{"action": "add", "kind": "fact",
                "text": "Обеспечивает семью; работает на двух работах.",
                "realm": "real_life", "source": "qa_answer"}],
    )
    res = await _run()
    assert len(res.candidates) == 1
    assert res.candidates[0].kind == "core"


@pytest.mark.asyncio
async def test_paraphrase_of_existing_item_becomes_confirm(monkeypatch):
    """Сильное совпадение с уже накопленной записью — подтверждение, не новая строка."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact",
         "text": "Воспитывает дочь; состоит в браке с мужем.",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[
        ExistingItemView(11, "fact", "Состоит в браке, воспитывает дочь", None)])
    assert len(res.candidates) == 1
    assert res.candidates[0].action == "confirm"
    assert res.candidates[0].target_id == 11


@pytest.mark.asyncio
async def test_different_facts_with_shared_words_not_merged(monkeypatch):
    """Главная страховка порога: два РАЗНЫХ факта с общими словами не склеиваются."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact", "text": "Учит испанский язык каждый день",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Учит английский язык каждый день",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "person", "text": "Мать юзера живёт в Москве",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "person", "text": "Отец юзера живёт в Москве",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run()
    assert len(res.candidates) == 4


@pytest.mark.asyncio
async def test_paraphrase_of_existing_item_does_not_shadow_new_fact(monkeypatch):
    """Перефраз схлопнулся в confirm, а новый факт той же пачки доехал как add."""
    _mock_classify(monkeypatch, items=[
        {"action": "add", "kind": "fact",
         "text": "Воспитывает дочь; состоит в браке с мужем.",
         "realm": "real_life", "source": "qa_answer"},
        {"action": "add", "kind": "fact", "text": "Работает нотариусом в частной конторе",
         "realm": "real_life", "source": "qa_answer"}])
    res = await _run(existing_items=[
        ExistingItemView(11, "fact", "Состоит в браке, воспитывает дочь", None)])
    actions = sorted(c.action for c in res.candidates)
    assert actions == ["add", "confirm"]
