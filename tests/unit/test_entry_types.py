"""Реестр типов обращения: байт-паритет dream-конфига с texts.py + тире-гард."""

from adapters.telegram import texts
from domain.entry_types import (
    ALL_KINDS,
    ENTRY_TYPES,
    KIND_CONFLICT,
    KIND_DREAM,
    KIND_SITUATION,
    entry_config,
)


def test_kinds_are_prod_column_values():
    assert KIND_DREAM == "dream"
    assert KIND_SITUATION == "external_situation"
    assert KIND_CONFLICT == "internal_conflict"
    assert ALL_KINDS == (KIND_DREAM, KIND_SITUATION, KIND_CONFLICT)
    assert set(ENTRY_TYPES) == set(ALL_KINDS)


def test_dream_config_is_byte_identical_to_texts_constants():
    """Спека B2: ветка сна получает конфиг, воспроизводящий текущее поведение
    байт-в-байт. domain не может импортировать adapters, поэтому dream-конфиг
    дублирует значения литералами — этот тест закрепляет паритет."""
    cfg = ENTRY_TYPES[KIND_DREAM]
    assert cfg.loop_prompt == texts.LOOP_ADD_PROMPT_DREAM
    assert cfg.per_session_question == texts.ASK_PER_SESSION_CONTEXT
    assert cfg.preq_offer_text == texts.IMG_Q_OFFER_TEXT
    assert cfg.preq_tmpl_many == texts.IMG_Q_QUESTION_TMPL_MANY
    assert cfg.preq_tmpl_one == texts.IMG_Q_QUESTION_TMPL_ONE
    assert cfg.preq_unclear == texts.IMG_Q_UNCLEAR_TEXT
    assert list(cfg.thinking_statuses) == texts.THINKING_STATUSES
    assert list(cfg.preq_thinking_statuses) == texts.IMG_Q_THINKING_STATUSES
    assert cfg.cf_analysis == texts.LLM_ERROR_CONTENT_FILTER
    assert cfg.cf_onboarding == texts.LLM_ERROR_CONTENT_FILTER_ONBOARDING
    assert cfg.cf_qa == texts.LLM_ERROR_CONTENT_FILTER_QA
    assert cfg.already_active == texts.QUICK_ALREADY_ACTIVE
    assert cfg.auto_paused_notice == texts.AUTO_PAUSED_NOTICE
    assert cfg.card_title_tmpl == texts.DREAM_CARD_TEMPLATE
    assert cfg.list_header == texts.DREAMS_LIST_HEADER
    assert cfg.list_empty == texts.NO_DREAMS_YET
    assert cfg.confirm_analyze_prompt == texts.DREAM_CONFIRM_ANALYZE_PROMPT
    assert cfg.saved_without_analysis == texts.DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE
    assert cfg.delete_confirm_1 == texts.DREAM_DELETE_CONFIRM_1
    assert cfg.delete_confirm_2 == texts.DREAM_DELETE_CONFIRM_2
    assert cfg.deleted_notice == texts.DREAM_DELETED_NOTICE
    assert cfg.repeat_invite == texts.NEWDREAM_PROMPT
    assert cfg.new_command == "newdream"
    assert cfg.list_command == "dreams"
    assert cfg.phase_pre_q == "image_q"


def test_commands_match_spec_b6():
    assert ENTRY_TYPES[KIND_SITUATION].new_command == "new_external_situation"
    assert ENTRY_TYPES[KIND_SITUATION].list_command == "external_situations"
    assert ENTRY_TYPES[KIND_CONFLICT].new_command == "new_internal_conflict"
    assert ENTRY_TYPES[KIND_CONFLICT].list_command == "internal_conflicts"


def test_onb_invites_verbatim_from_spec_b3():
    assert ENTRY_TYPES[KIND_SITUATION].onb_invite == (
        "{name}!\n\n"
        "Давайте вместе посмотрим на ситуацию, которая вас волнует.\n\n"
        "Опишите её подробно, голосом или текстом."
    )
    assert ENTRY_TYPES[KIND_CONFLICT].onb_invite == (
        "{name}!\n\n"
        "Давайте вместе посмотрим на ваш внутренний конфликт.\n\n"
        "Опишите его подробно, голосом или текстом."
    )
    assert ENTRY_TYPES[KIND_DREAM].onb_invite == (
        "{name}!\n\n"
        "Давайте посмотрим ваш первый сон вместе.\n\n"
        "Расскажите сон, который вас интересует, голосом или текстом."
    )


def test_no_em_dash_in_new_type_texts():
    for kind in (KIND_SITUATION, KIND_CONFLICT):
        cfg = ENTRY_TYPES[kind]
        for field in (
            cfg.onb_invite, cfg.repeat_invite, cfg.loop_prompt,
            cfg.per_session_question, cfg.preq_offer_text, cfg.preq_unclear,
            cfg.cf_analysis, cfg.cf_onboarding, cfg.cf_qa, cfg.already_active,
            cfg.card_title_tmpl, cfg.list_header, cfg.list_empty,
            cfg.confirm_analyze_prompt, cfg.saved_without_analysis,
            cfg.auto_paused_notice,
            cfg.delete_confirm_1, cfg.delete_confirm_2, cfg.deleted_notice,
            *cfg.thinking_statuses, *cfg.preq_thinking_statuses,
        ):
            assert "—" not in field, f"em-dash в тексте типа {kind}: {field!r}"


def test_entry_config_fallback_to_dream():
    assert entry_config("") is ENTRY_TYPES[KIND_DREAM]
    assert entry_config("garbage") is ENTRY_TYPES[KIND_DREAM]
    assert entry_config(KIND_CONFLICT) is ENTRY_TYPES[KIND_CONFLICT]
