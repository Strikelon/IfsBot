"""Промпты conflict.v1: сборка, версии, контракт, содержание про две стороны."""

from data.models import RequestBranch
from llm.prompts.conflict import (
    CONFLICT_VERSION_FOLLOWUP,
    CONFLICT_VERSION_INITIAL,
    CONFLICT_VERSION_QUESTIONS,
    build_followup_messages,
    build_initial_messages,
    build_questions_messages,
)
from llm.prompts.quick import QuickContext


def _ctx() -> QuickContext:
    return QuickContext(
        user_display_name="Катя",
        user_life_context="пишу диплом",
        per_session_context="злюсь на себя за лень",
        request_branch=RequestBranch.UNCLEAR,
        dream_transcription="хочу сесть за диплом, но неделями откладываю",
        memory_block="",
        image_answers_block="",
    )


def test_versions():
    assert CONFLICT_VERSION_INITIAL == "conflict.v1"
    assert CONFLICT_VERSION_FOLLOWUP == "conflict.v1.followup"
    assert CONFLICT_VERSION_QUESTIONS == "conflict_questions.v1"


def test_initial_structure_and_two_sides_focus():
    msgs = build_initial_messages(_ctx())
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "===QA_OPTIONS===" in system
    assert "обе" in system.lower()                 # обе стороны хотят хорошего
    assert "цена" in system.lower() or "цену" in system.lower()
    assert "ВНУТРЕННИЙ КОНФЛИКТ" in user
    assert "откладываю" in user
    assert "СОН:" not in user
    assert "сновидени" not in system.lower()


def test_followup_signature_matches_quick():
    msgs = build_followup_messages(
        _ctx(), qa_history=[("q", "a")], user_followup="дальше",
        qa_summary=None, qa_map=None,
    )
    assert msgs[-1] == {"role": "user", "content": "дальше"}
    assert len(msgs) == 1 + 2 + 1


def test_questions_prompt_contract():
    system = build_questions_messages(_ctx())[0]["content"]
    assert '{"questions"' in system
    assert "сторон" in system.lower()              # вопросы про обе стороны конфликта
