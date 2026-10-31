"""Greeting в ONBOARDING_INTRO: имя — восклицательный знак — приглашение к сну."""

from adapters.telegram.texts import ONBOARDING_INTRO


def test_onboarding_intro_uses_exclamation_after_name():
    rendered = ONBOARDING_INTRO.format(name="Александр")
    # «Александр!» а не «Александр.»
    assert "Александр!" in rendered
    assert "Александр." not in rendered


def test_onboarding_intro_breaks_line_before_dream_request():
    rendered = ONBOARDING_INTRO.format(name="Иван")
    # Приветствие и приглашение к сну через пустую строку, затем двойной перенос.
    assert "Иван!\n\nДавайте посмотрим" in rendered
    assert "\n\nРасскажите" in rendered
