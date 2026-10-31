"""Unit-тесты сборки Quick-промпта. Чистая логика — без I/O."""

from __future__ import annotations

import json

from data.models import RequestBranch
from llm.memory_select import HEADER as MEMORY_HEADER
from llm.prompts.quick import (
    PROMPT_VERSION,
    QuickContext,
    build_followup_messages,
    build_initial_messages,
)


def _ctx(memory_block: str = "", image_answers_block: str = "") -> QuickContext:
    return QuickContext(
        user_display_name="Катя",
        user_life_context="меняю работу",
        per_session_context="вчера было сложное собрание",
        request_branch=RequestBranch.FEEL_STATE,
        dream_transcription="я шла по лесу и увидела тень",
        memory_block=memory_block,
        image_answers_block=image_answers_block,
    )


def test_prompt_version_bumped_for_v2() -> None:
    """quick.v3.15: ответы об образах = мост вглубь (без цитирования), map без grounded."""
    assert PROMPT_VERSION == "quick.v3.15"


def test_initial_system_prompt_demands_brevity_and_no_verbatim_echo() -> None:
    """Промпт требует краткости («Пиши, сокращай»), запрещает вводные-пустышки и
    дословный повтор слов юзера."""
    system = build_initial_messages(_ctx())[0]["content"]
    assert "сокращай" in system.lower()  # блок краткости
    assert "я заметил" in system.lower()  # пример запрещённой вводной
    assert "дословно" in system.lower()  # запрет повтора слов юзера


def test_initial_messages_is_zero_shot() -> None:
    """quick.v2: few-shot убран (примеры противоречили новой философии)."""
    msgs = build_initial_messages(_ctx())

    # Zero-shot: ровно [system, user] — никаких ассистентских примеров.
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert len(msgs) == 2


def test_initial_system_prompt_contains_ifs_directives() -> None:
    """quick.v3: IFS — внутренняя оптика, но термины переводятся в простой язык; запреты на месте."""
    msgs = build_initial_messages(_ctx())
    system = msgs[0]["content"]
    # IFS остаётся внутренней оптикой модели (термины упомянуты в инструкции):
    assert "Самость" in system
    assert "Защитник" in system
    # ...но наружу термины НЕ показываются — есть правило перевода в простой язык:
    assert "не показыва" in system.lower() or "переводи" in system.lower()
    assert "уязвимая часть" in system.lower()  # пример простого перевода (вместо «Изгнанник»)
    # Вовлекающий вопрос в конце разбора — теперь часть структуры:
    assert "вовлекающ" in system.lower()
    # Запреты:
    assert "markdown" in system.lower()
    assert "сонник" in system.lower() or "крокодил" in system.lower()


def test_initial_user_block_contains_all_context() -> None:
    msgs = build_initial_messages(_ctx())
    user_block = msgs[-1]["content"]
    assert "Катя" in user_block
    assert "меняю работу" in user_block
    assert "сложное собрание" in user_block
    assert "я шла по лесу" in user_block


def test_initial_user_block_has_no_branch_line() -> None:
    """Шаг выбора ветки убран — в промпте больше нет «ВЕТКА РАЗБОРА»."""
    msgs = build_initial_messages(_ctx())
    assert "ВЕТКА РАЗБОРА" not in msgs[-1]["content"]


def test_initial_system_prompt_instructs_quote_and_bold_markup() -> None:
    """Промпт просит размечать цитату сна `> ` и финальный вопрос `**…**`."""
    system = build_initial_messages(_ctx())[0]["content"]
    assert "> " in system  # маркер цитаты
    assert "**" in system  # маркер жирного финального вопроса


def test_followup_messages_uses_history() -> None:
    ctx = _ctx()
    history = [
        ("я не знаю, что чувствую", "Где сейчас в теле эта неясность?"),
        ("где-то в груди", "А если послушать эту тяжесть — что она хочет сказать?"),
    ]
    msgs = build_followup_messages(ctx, qa_history=history, user_followup="не знаю")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    # 2 пары истории + последняя реплика юзера = минимум 5 после system (1 + 4 + 1 = 6)
    assert len(msgs) >= 6
    assert msgs[-1]["content"] == "не знаю"


def test_followup_messages_truncates_history_to_4() -> None:
    ctx = _ctx()
    history = [("u%d" % i, "a%d" % i) for i in range(10)]
    msgs = build_followup_messages(ctx, qa_history=history, user_followup="?")
    # последние 4 пары + system + последний user = 1 + 8 + 1 = 10
    assert len(msgs) == 10
    # Первая историческая пара должна быть #6 (10-4), не #0:
    assert msgs[1]["content"] == "u6"
    assert msgs[2]["content"] == "a6"


def test_followup_system_prompt_mentions_qa_mode() -> None:
    msgs = build_followup_messages(_ctx(), qa_history=[], user_followup="?")
    system = msgs[0]["content"]
    # Системный промпт follow-up отличается от первичного: упоминает «уточнение»
    assert "уточн" in system.lower()


def test_followup_system_contains_dream_context() -> None:
    msgs = build_followup_messages(_ctx(), qa_history=[], user_followup="?")
    system = msgs[0]["content"]
    assert "я шла по лесу" in system  # dream transcription embedded
    assert "сложное собрание" in system  # per_session context embedded


from llm.prompts.quick import (  # noqa: E402
    PROMPT_VERSION_FOLLOWUP,
    PROMPT_VERSION_INITIAL,
    QA_OPTIONS_SENTINEL,
    split_qa_options,
)


def test_initial_system_instructs_tail_sentinel() -> None:
    system = build_initial_messages(_ctx())[0]["content"]
    assert "===QA_OPTIONS===" in system
    assert '"kind"' in system and '"options"' in system


def test_followup_system_instructs_tail_sentinel() -> None:
    system = build_followup_messages(
        _ctx(), qa_history=[], user_followup="да"
    )[0]["content"]
    assert "===QA_OPTIONS===" in system
    assert '"kind"' in system and '"options"' in system


def test_split_qa_options_choice_two_options() -> None:
    raw = (
        "Текст разбора.\n\n**Если интересно копнуть глубже:**\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind": "choice", "options": ["Линия А", "Линия Б"]}'
    )
    prose, kind, options, _ = split_qa_options(raw)
    assert prose == "Текст разбора.\n\n**Если интересно копнуть глубже:**"
    assert kind == "choice"
    assert options == ["Линия А", "Линия Б"]


def test_split_qa_options_confirm() -> None:
    raw = (
        "Текст.\n\n**Хотите разобрать линию про работу?**\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind": "confirm", "options": []}'
    )
    prose, kind, options, _ = split_qa_options(raw)
    assert kind == "confirm"
    assert prose.endswith("работу?**")


def test_split_qa_options_caps_choice_at_two() -> None:
    raw = f"P\n{QA_OPTIONS_SENTINEL}\n" + '{"kind":"choice","options":["1","2","3"]}'
    _, _, options, _ = split_qa_options(raw)
    assert options == ["1", "2"]


def test_split_qa_options_drops_empty_labels() -> None:
    raw = f"P\n{QA_OPTIONS_SENTINEL}\n" + '{"kind":"choice","options":["","  ","Ок"]}'
    _, _, options, _ = split_qa_options(raw)
    assert options == ["Ок"]


def test_split_qa_options_drops_finish_duplicate_chips() -> None:
    """LLM иногда «протекает» вариантом-кнопкой завершения вопреки запрету в промпте.
    Детерминированный гард обязан выкинуть такие чипы — завершение только через
    фиксированную кнопку снизу (см. quick.py / QA_FINISH_BTN)."""
    finish_like = [
        "Подвести итог сна",
        "Завершить разбор",
        "✅ Сохранить и завершить",
        "Сохранить и вернуться позже",
        "Сохранить и разобрать новое",
        "Закончить на сегодня",
        "Подытожить",
    ]
    for label in finish_like:
        raw = (
            f"P\n{QA_OPTIONS_SENTINEL}\n"
            + '{"kind":"choice","options":["Линия про работу", %s]}' % json.dumps(  # noqa: UP031
                label, ensure_ascii=False
            )
        )
        _, kind, options, _ = split_qa_options(raw)
        assert kind == "choice"
        assert label not in options, f"finish-дубль не отфильтрован: {label!r}"
        assert options == ["Линия про работу"]


def test_split_qa_options_keeps_legit_line_labels() -> None:
    """Гард не должен резать нормальные линии разбора."""
    raw = (
        f"P\n{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"choice","options":["Собаки внизу","Вампир на чердаке"]}'
    )
    _, _, options, _ = split_qa_options(raw)
    assert options == ["Собаки внизу", "Вампир на чердаке"]


def test_split_qa_options_no_sentinel_is_fallback() -> None:
    raw = "Просто разбор без хвоста.  "
    prose, kind, options, _ = split_qa_options(raw)
    assert prose == "Просто разбор без хвоста."
    assert kind == "choice"
    assert options == []


def test_split_qa_options_broken_json_is_fallback() -> None:
    raw = f"Проза.\n{QA_OPTIONS_SENTINEL}\nне json {{{{"
    prose, kind, options, _ = split_qa_options(raw)
    assert prose == "Проза."
    assert kind == "choice"
    assert options == []


def test_split_qa_options_unknown_kind_is_fallback() -> None:
    raw = f"Проза.\n{QA_OPTIONS_SENTINEL}\n" + '{"kind":"weird","options":["x"]}'
    _, kind, options, _ = split_qa_options(raw)
    assert kind == "choice"
    assert options == []


def test_split_qa_options_strips_json_fences() -> None:
    raw = (
        f"Проза.\n{QA_OPTIONS_SENTINEL}\n"
        '```json\n{"kind":"choice","options":["A"]}\n```'
    )
    _, kind, options, _ = split_qa_options(raw)
    assert kind == "choice"
    assert options == ["A"]


def test_zapreshcheno_no_longer_says_vybor_iz_dvukh_liniy() -> None:
    """Fix 1: ни один системный промпт не инструктирует перечислять варианты в прозе.

    Старая фраза «выбор из двух линий» в ЗАПРЕЩЕНО прямо противоречила дизайну,
    при котором варианты уходят кнопками, а не текстом. Оба промпта должны
    содержать явный запрет на перечисление вариантов в тексте и не содержать
    старую формулировку.
    """
    initial_system = build_initial_messages(_ctx())[0]["content"]
    followup_system = build_followup_messages(
        _ctx(), qa_history=[], user_followup="да"
    )[0]["content"]

    # Старая противоречивая фраза должна быть удалена из обоих промптов:
    assert "выбор из двух линий" not in initial_system
    assert "выбор из двух линий" not in followup_system

    # Новая формулировка: варианты уходят кнопками, без перечисления в тексте
    # (проверяем case-insensitive, т.к. промпт использует заглавные «БЕЗ»/«без»):
    assert "перечисления вариантов в тексте" in initial_system.lower()
    assert "перечисления вариантов в тексте" in followup_system.lower()
    # Оба промпта явно указывают, что варианты уходят кнопками:
    assert "уходят кнопками" in initial_system
    assert "уходят кнопками" in followup_system


def test_split_qa_options_clarify_kind() -> None:
    raw = (
        "Текст.\n\n**Тигр был агрессивным или добрым?**\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"clarify","options":[],"map":{"focus":"тигр","type":"unknown","stage":"вглубь","covered":[]}}'
    )
    prose, kind, options, qa_map = split_qa_options(raw)
    assert kind == "clarify"
    assert options == []
    assert qa_map == {"focus": "тигр", "type": "unknown", "stage": "вглубь", "covered": []}


def test_split_qa_options_returns_map_for_choice() -> None:
    raw = (
        f"P\n{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"choice","options":["А","Б"],"map":{"focus":"f","type":"manager","stage":"знакомство","covered":["x"]}}'
    )
    _, kind, options, qa_map = split_qa_options(raw)
    assert kind == "choice" and options == ["А", "Б"]
    assert qa_map["focus"] == "f" and qa_map["covered"] == ["x"]


def test_split_qa_options_map_none_when_absent() -> None:
    raw = f"P\n{QA_OPTIONS_SENTINEL}\n" + '{"kind":"choice","options":["А"]}'
    _, _, options, qa_map = split_qa_options(raw)
    assert options == ["А"]
    assert qa_map is None


def test_split_qa_options_no_sentinel_4tuple() -> None:
    prose, kind, options, qa_map = split_qa_options("просто разбор.  ")
    assert prose == "просто разбор."
    assert kind == "choice" and options == [] and qa_map is None


def test_split_qa_options_broken_json_4tuple() -> None:
    raw = f"Проза.\n{QA_OPTIONS_SENTINEL}\nне json {{{{"
    prose, kind, options, qa_map = split_qa_options(raw)
    assert prose == "Проза." and kind == "choice" and options == [] and qa_map is None


def test_split_qa_options_clarify_with_bad_map_is_none() -> None:
    raw = f"P\n{QA_OPTIONS_SENTINEL}\n" + '{"kind":"clarify","options":[],"map":"notdict"}'
    _, kind, _, qa_map = split_qa_options(raw)
    assert kind == "clarify" and qa_map is None


# ── Task 2: quick.v3.8 tests ────────────────────────────────────────────────

def test_prompt_versions_bumped_to_v38() -> None:
    # Updated: v3.15 — image_answers_block как «мост вглубь», map без grounded
    assert PROMPT_VERSION_INITIAL == "quick.v3.15"
    assert PROMPT_VERSION_FOLLOWUP == "quick.v3.13.followup"


def test_followup_system_documents_clarify_and_map() -> None:
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    low = system.lower()
    assert "clarify" in low
    assert '"map"' in system
    assert "один" in low and "вопрос" in low  # правило одного вопроса


def test_followup_system_documents_image_grounding() -> None:
    """Followup-промпт содержит обязательный шаг заземления образа + поле map.grounded."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    low = system.lower()
    assert "заземл" in low  # правило заземления образа
    assert "grounded" in low  # трек «уже заземлён» в карте беседы


def test_followup_injects_conversation_map() -> None:
    qa_map = {
        "focus": "тигр", "type": "manager", "stage": "вглубь",
        "covered": ["лес"], "grounded": ["отец"],
    }
    system = build_followup_messages(
        _ctx(), qa_history=[], user_followup="да", qa_map=qa_map
    )[0]["content"]
    assert "Карта беседы" in system
    assert "тигр" in system and "вглубь" in system
    # заземлённые образы прокидываются обратно, чтобы не переспрашивать
    assert "отец" in system


def test_followup_no_map_block_when_absent() -> None:
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    assert "Карта беседы" not in system


def test_initial_system_seeds_map_in_tail_spec() -> None:
    system = build_initial_messages(_ctx())[0]["content"]
    assert '"map"' in system


# ── quick.v3.9 tests ────────────────────────────────────────────────────────

def test_prompt_versions_v39() -> None:
    """Version bump: quick.v3.15 / quick.v3.13.followup — image_answers_block
    в initial (Task 12 фичи image questions, фикс v3.15); followup не менялся."""
    assert PROMPT_VERSION_INITIAL == "quick.v3.15"
    assert PROMPT_VERSION_FOLLOWUP == "quick.v3.13.followup"


def test_initial_first_question_is_always_choice_of_two_sides() -> None:
    """Первый вопрос после разбора — ВСЕГДА выбор стороны (choice, ровно 2 кнопки),
    а не да/нет вопрос про одну линию (confirm). Скриншот-фидбэк 2026-06-27."""
    system = build_initial_messages(_ctx())[0]["content"]
    # Initial-промпт больше не предлагает confirm-формат для первого вопроса:
    assert "confirm" not in system
    # Явно требует ровно 2 варианта-стороны:
    assert "РОВНО 2" in system
    # Формулировка-приглашение в духе «какую сторону исследовать»:
    assert "какую сторону" in system.lower()


def test_initial_system_requires_short_option_labels() -> None:
    """Fix 1: initial tail/options spec требует короткие лейблы для кнопок-чипов."""
    system = build_initial_messages(_ctx())[0]["content"]
    # Must mention short label constraint — either "2–4 слова" or "до ~25 символов"
    assert "2–4 слова" in system or "до ~25 символов" in system


def test_followup_system_requires_short_option_labels() -> None:
    """Fix 1: followup tail/options spec требует короткие лейблы для кнопок-чипов."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    # Must mention short label constraint — either "2–4 слова" or "до ~25 символов"
    assert "2–4 слова" in system or "до ~25 символов" in system


def test_followup_no_duplicate_finish_button() -> None:
    """Fix 2: followup не дублирует кнопку завершения — она только через стандартную кнопку."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    low = system.lower()
    # Must forbid offering finish as an option
    assert "только через стандартную кнопку" in low
    # And must mention завершение in the context of being forbidden as option
    assert "завершени" in low


def test_followup_caps_clarify_at_two() -> None:
    """Fix 3: followup ограничивает число clarify-вопросов — не более 2 за линию."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    # Must contain "МАКСИМУМ 2" or "максимум 2"
    assert "максимум 2" in system.lower()


def test_followup_bans_feel_toward_questions() -> None:
    """Fix 5: followup запрещает рефлексивные вопросы про чувства к 'части/стороне'."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    low = system.lower()
    # Must contain a ban on feel-toward questions
    assert "не спрашивай про чувства к части" in low or "запрещены рефлексивные" in low


def test_followup_interpretation_is_default_not_clarify() -> None:
    """Fix 3+5: followup формулирует интерпретацию как главное действие, clarify — редкое исключение."""
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    low = system.lower()
    # Must emphasize interpretation/insight as primary, clarify as exception
    assert "интерпретаци" in low or "инсайт" in low
    # Must say clarify is rare/exception
    assert "редк" in low or "исключение" in low


# ── Task 9: инжект памяти в промпты (quick.v3.13) ───────────────────────────

_MEM_BLOCK = (
    MEMORY_HEADER
    + "- [сторона] Внутренний Критик — включается на работе\n"
    + "- [паттерн] откладывает важные решения"
)


def test_memory_block_injected_into_initial_user_message() -> None:
    """memory_block попадает в user-сообщение initial-промпта: после «В фокусе
    жизни», до «ЗАПРОС НА ЭТУ СЕССИЮ»."""
    user = build_initial_messages(_ctx(memory_block=_MEM_BLOCK))[1]["content"]
    assert "Внутренний Критик" in user
    assert MEMORY_HEADER.strip() in user
    assert (
        user.index("В фокусе жизни")
        < user.index("Внутренний Критик")
        < user.index("ЗАПРОС НА ЭТУ СЕССИЮ")
    )


def test_memory_block_injected_into_followup_system() -> None:
    """memory_block попадает в справочный блок followup-system (после «в фокусе жизни»)."""
    system = build_followup_messages(
        _ctx(memory_block=_MEM_BLOCK), qa_history=[], user_followup="да"
    )[0]["content"]
    assert "откладывает важные решения" in system
    assert system.index("в фокусе жизни") < system.index("откладывает важные решения")
    assert system.index("откладывает важные решения") < system.index("- Запрос:")


def test_no_memory_block_when_empty() -> None:
    """Пустой memory_block ⇒ ни заголовка памяти, ни лишнего текста в сообщениях."""
    user = build_initial_messages(_ctx())[1]["content"]
    assert MEMORY_HEADER.strip() not in user
    system = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    assert MEMORY_HEADER.strip() not in system


def test_empty_memory_block_keeps_user_block_byte_identical() -> None:
    """Для юзеров без памяти user-блок байт-в-байт как до v3.12."""
    user = build_initial_messages(_ctx())[1]["content"]
    assert user == (
        "ПРОФИЛЬ:\n"
        "Имя: Катя\n"
        "В фокусе жизни: меняю работу\n\n"
        "ЗАПРОС НА ЭТУ СЕССИЮ: «вчера было сложное собрание»\n\n"
        "СОН:\n"
        "«я шла по лесу и увидела тень»"
    )


def test_prompt_versions_v312() -> None:
    assert PROMPT_VERSION_INITIAL == "quick.v3.15"
    assert PROMPT_VERSION_FOLLOWUP == "quick.v3.13.followup"
    assert PROMPT_VERSION == PROMPT_VERSION_INITIAL


def test_system_prompts_explain_memory_block_semantics() -> None:
    """Оба system-промпта объясняют блок «ЧТО УЖЕ ИЗВЕСТНО»: это знание о
    РЕАЛЬНОЙ жизни юзера из прошлых разборов; опираться, но не пересказывать."""
    initial = build_initial_messages(_ctx())[0]["content"]
    followup = build_followup_messages(_ctx(), qa_history=[], user_followup="да")[0]["content"]
    for system in (initial, followup):
        assert "ЧТО УЖЕ ИЗВЕСТНО" in system
        assert "преамбул" in system.lower()


# ── Task 12: инжект image_answers_block в initial (quick.v3.14) ─────────────

def test_image_answers_block_injected():
    ctx = _ctx(image_answers_block='ЧТО ПОЛЬЗОВАТЕЛЬ РАССКАЗАЛ О КЛЮЧЕВЫХ ОБРАЗАХ СНА '
               '(о своей РЕАЛЬНОЙ жизни, собрано до разбора):\n– Образ "Отец": тёплые')
    msgs = build_initial_messages(ctx)
    body = msgs[1]["content"]
    assert 'Образ "Отец"' in body
    # Блок стоит МЕЖДУ памятью и «ЗАПРОС НА ЭТУ СЕССИЮ»
    assert body.index("ОБРАЗАХ СНА") < body.index("ЗАПРОС НА ЭТУ СЕССИЮ")


def test_no_image_answers_block_when_empty():
    ctx = _ctx(image_answers_block="")
    assert "ОБРАЗАХ СНА" not in build_initial_messages(ctx)[1]["content"]


def test_initial_system_has_no_grounded_field():
    """v3.15: модель НЕ эмитит grounded в initial-map (формат v3.13 — flash надёжно
    отдаёт хвост). map.grounded сидируется детерминированно в _deliver_analysis."""
    from llm.prompts.quick import _SYSTEM_INITIAL
    assert "grounded" not in _SYSTEM_INITIAL


def test_initial_system_image_answers_are_bridge_not_quotes():
    """v3.15: блок ответов об образах — мост вглубь, дословное цитирование запрещено."""
    from llm.prompts.quick import _SYSTEM_INITIAL
    assert "ЧТО ПОЛЬЗОВАТЕЛЬ РАССКАЗАЛ О КЛЮЧЕВЫХ ОБРАЗАХ СНА" in _SYSTEM_INITIAL
    assert "МОСТ" in _SYSTEM_INITIAL  # ответ = мост к глубине, не факт для вставки
    assert "цитировать" in _SYSTEM_INITIAL  # запрет дословного повтора ответов


def test_prompt_versions_v315():
    assert PROMPT_VERSION_INITIAL == "quick.v3.15"
    assert PROMPT_VERSION_FOLLOWUP == "quick.v3.13.followup"  # followup не менялся

