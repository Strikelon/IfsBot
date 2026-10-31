"""Тесты domain/states.py — чистый модуль, без БД и LLM."""

from __future__ import annotations

import pytest

from data.models import ImageShape
from domain.states import (
    SessionStateData,
    UnblendingAttempt,
    p4_attitude_phrase,
    p4_role_verb,
    shape_label,
)


def test_session_state_data_round_trip():
    data = SessionStateData(
        target_part_name="Тёмная фигура",
        target_part_shape=ImageShape.HUMAN_LIKE,
        target_part_ifs_category="firefighter",
        somatic_location="давит в груди",
    )
    dumped = data.model_dump(mode="json")
    parsed = SessionStateData.model_validate(dumped)
    assert parsed == data


def test_session_state_data_defaults_are_empty():
    data = SessionStateData(
        target_part_name="x",
        target_part_shape=ImageShape.OBJECT,
        target_part_ifs_category="unclear",
    )
    assert data.somatic_location is None
    assert data.part_role is None
    assert data.unblending_history == []
    assert data.pivot_history == []


def test_session_state_data_unblending_history_append():
    data = SessionStateData(
        target_part_name="x",
        target_part_shape=ImageShape.OBJECT,
        target_part_ifs_category="manager",
    )
    data.unblending_history.append(
        UnblendingAttempt(emotion="страх", outcome="released")
    )
    data.unblending_history.append(
        UnblendingAttempt(emotion="злость", outcome="blended")
    )
    assert len(data.unblending_history) == 2
    assert data.unblending_history[1].outcome == "blended"


@pytest.mark.parametrize(
    "shape,expected_keyword",
    [
        (ImageShape.HUMAN_LIKE, "фигура"),
        (ImageShape.ANIMAL, "животное"),
        (ImageShape.OBJECT, "предмет"),
        (ImageShape.ELEMENT, "стихия"),
        (ImageShape.PHENOMENON, "явление"),
        (ImageShape.SENSATION, "ощущение"),
        (ImageShape.ABSTRACT, "это"),
    ],
)
def test_shape_label_returns_russian_word(shape, expected_keyword):
    assert shape_label(shape) == expected_keyword


@pytest.mark.parametrize("shape", list(ImageShape))
def test_p4_role_verb_has_value_for_every_shape(shape):
    verb = p4_role_verb(shape)
    assert verb and isinstance(verb, str)


def test_p4_attitude_question_uses_satisfied_word_only_for_human_like():
    human = p4_attitude_phrase(ImageShape.HUMAN_LIKE)
    object_ = p4_attitude_phrase(ImageShape.OBJECT)
    sensation = p4_attitude_phrase(ImageShape.SENSATION)
    assert "довольна" in human
    assert "довольна" not in object_
    assert "довольна" not in sensation
