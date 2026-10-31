"""Стражи хирургии Памяти v2: конвейер «per_session → профиль» удалён.

Хэндлеры разбора не ссылаются на update_life_context — автоматика больше
не трогает profile_life_context (спека A2)."""

from __future__ import annotations


def test_handlers_do_not_reference_update_life_context():
    import adapters.telegram.handlers.onboarding as onboarding
    import adapters.telegram.handlers.quick as quick

    assert not hasattr(quick, "update_life_context")
    assert not hasattr(quick, "CONTEXT_UPDATE_VERSION")
    assert not hasattr(onboarding, "update_life_context")
    assert not hasattr(onboarding, "CONTEXT_UPDATE_VERSION")


def test_context_update_module_removed():
    """Спека A2: llm/context_update.py удалён вместе с вызовами."""
    import importlib.util

    assert importlib.util.find_spec("llm.context_update") is None


def test_onboarding_still_has_compaction():
    """Компоновка самоописания (часть пути онбординга) сохранена."""
    import adapters.telegram.handlers.onboarding as onboarding

    assert hasattr(onboarding, "compact_life_context")
    assert hasattr(onboarding, "CONTEXT_COMPACT_VERSION")
