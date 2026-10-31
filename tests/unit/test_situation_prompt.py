"""Промпты situation.v1: сборка messages, версии, контракт хвоста, «не про сон»."""

from data.models import RequestBranch
from llm.prompts.quick import QuickContext
from llm.prompts.situation import (
    SITUATION_VERSION_FOLLOWUP,
    SITUATION_VERSION_INITIAL,
    SITUATION_VERSION_QUESTIONS,
    build_followup_messages,
    build_initial_messages,
    build_questions_messages,
)


def _ctx(**over) -> QuickContext:
    base = dict(
        user_display_name="Катя",
        user_life_context="меняю работу",
        per_session_context="боюсь подвести команду",
        request_branch=RequestBranch.UNCLEAR,
        dream_transcription="поссорилась с руководителем из-за дедлайна",
        memory_block="",
        image_answers_block="",
    )
    base.update(over)
    return QuickContext(**base)


def test_versions():
    assert SITUATION_VERSION_INITIAL == "situation.v1"
    assert SITUATION_VERSION_FOLLOWUP == "situation.v1.followup"
    assert SITUATION_VERSION_QUESTIONS == "situation_questions.v1"


def test_initial_structure_and_content():
    msgs = build_initial_messages(_ctx())
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "===QA_OPTIONS===" in system          # контракт хвоста как у снов
    assert "ВОЛНУЮЩАЯ СИТУАЦИЯ" in user
    assert "поссорилась с руководителем" in user
    assert "боюсь подвести команду" in user
    assert "СОН:" not in user                     # это не разбор сна
    assert "сновидени" not in system.lower()


def test_followup_truncates_history_to_4_pairs_and_injects_map():
    history = [(f"q{i}", f"a{i}") for i in range(6)]
    msgs = build_followup_messages(
        _ctx(),
        qa_history=history,
        user_followup="а почему я так завелась?",
        qa_summary="краткое содержание",
        qa_map={"focus": "контроль", "type": "manager", "stage": "вглубь", "covered": []},
    )
    assert msgs[0]["role"] == "system"
    assert "краткое содержание" in msgs[0]["content"]
    assert "контроль" in msgs[0]["content"]
    # system + 4 пары + финальный user_followup
    assert len(msgs) == 1 + 4 * 2 + 1
    assert msgs[1]["content"] == "q2"              # первые 2 пары отрезаны
    assert msgs[-1] == {"role": "user", "content": "а почему я так завелась?"}


def test_questions_prompt_contract():
    msgs = build_questions_messages(_ctx())
    system = msgs[0]["content"]
    assert '{"questions"' in system                # строгий JSON-контракт
    assert "1–3" in system or "1-3" in system
    assert "сон" not in system.lower().replace("персон", "")  # не про сны


def test_answers_block_included_when_present():
    msgs = build_initial_messages(_ctx(image_answers_block="ЧТО ПОЛЬЗОВАТЕЛЬ ОТВЕТИЛ:\n– x"))
    assert "ЧТО ПОЛЬЗОВАТЕЛЬ ОТВЕТИЛ" in msgs[1]["content"]
