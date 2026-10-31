"""Тесты user-facing текстов при блокировке контента фильтром провайдера.

Регрессия инцидента 2026-07-20: человек рассказал боту о пережитом насилии
и получил в ответ «Что-то сломалось в коде бота», а затем остался без выхода
из состояния.
"""

import pytest

from adapters.telegram.texts import (
    LLM_ERROR_CONTENT_FILTER,
    LLM_ERROR_CONTENT_FILTER_ONBOARDING,
    LLM_ERROR_CONTENT_FILTER_QA,
    LLM_ERROR_PERMANENT,
    llm_error_message,
)
from llm.generator import ErrorCategory

_ALL_THREE = (
    LLM_ERROR_CONTENT_FILTER,
    LLM_ERROR_CONTENT_FILTER_ONBOARDING,
    LLM_ERROR_CONTENT_FILTER_QA,
)


@pytest.mark.parametrize("text", _ALL_THREE)
def test_content_filter_texts_share_opening_and_project_dash(text):
    """Общая первая фраза во всех трёх + тире по норме проекта."""
    assert text.startswith("Сработал безопасный режим:")
    assert "—" not in text
    assert "―" not in text


@pytest.mark.parametrize("text", _ALL_THREE)
def test_content_filter_texts_drop_old_technical_wording(text):
    """Старая формулировка «провайдер… переформулируйте запрос» не должна вернуться."""
    lowered = text.lower()
    assert "провайдер" not in lowered
    assert "переформулируйте запрос" not in lowered


def test_analysis_text_sends_user_to_newdream():
    assert "/newdream" in LLM_ERROR_CONTENT_FILTER


def test_onboarding_text_does_not_send_user_to_newdream():
    """На онбординге мы перезапускаем онбординг, а не отправляем в /newdream."""
    assert "/newdream" not in LLM_ERROR_CONTENT_FILTER_ONBOARDING


def test_qa_text_keeps_user_in_place():
    """В Q&A юзер остаётся в разборе – текст не должен звать никуда уходить."""
    assert "/newdream" not in LLM_ERROR_CONTENT_FILTER_QA
    assert "продолжим с этого места" in LLM_ERROR_CONTENT_FILTER_QA


def test_category_maps_to_analysis_text_by_default():
    """Общий маппинг отдаёт вариант обычного разбора – самый безопасный дефолт."""
    assert llm_error_message(ErrorCategory.CONTENT_FILTER) == LLM_ERROR_CONTENT_FILTER


def test_permanent_category_still_maps_to_permanent_text():
    assert llm_error_message(ErrorCategory.PERMANENT) == LLM_ERROR_PERMANENT
