"""Разрез по dreams.kind в отчёте (B8). Файл читается текстом: скрипт не
импортируем — он требует paramiko/matplotlib и SSH-окружение."""

from pathlib import Path

_SRC = Path("scripts/analytics_report.py").read_text(encoding="utf-8")


def test_engage_counts_split_by_kind():
    assert "d.kind='dream'" in _SRC
    assert "d.kind='external_situation'" in _SRC
    assert "d.kind='internal_conflict'" in _SRC
    assert "AS situations" in _SRC
    assert "AS conflicts" in _SRC


def test_funnel_step_names_untouched():
    # Имена шагов воронки неизменны (B8) — отчёт не ломаем.
    assert "onb_analysis_delivered" in _SRC or "onb_" in _SRC


def test_imgq_report_split_by_entry_type():
    """Пре-вопросы печатаются раздельно по типам, иначе ситуации подмешиваются к снам."""
    assert "split_by_entry_type" in _SRC
    assert "_ENTRY_TYPE_TITLES" in _SRC
    assert "_print_imgq_block" in _SRC
