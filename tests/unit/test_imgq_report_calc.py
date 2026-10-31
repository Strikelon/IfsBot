from analytics.image_questions import (
    ImgQEvent,
    answered_histogram,
    drop_by_question,
    offer_choice_split,
)


def _e(t, **meta):
    return ImgQEvent(user_id=meta.pop("uid", 1), event_type=t, metadata=meta)


def test_offer_choice_split_dedups_sessions():
    events = [
        _e("img_q_offered", session_id=1, is_onboarding=False),
        _e("img_q_offered", session_id=1, is_onboarding=False),  # дубль (resume)
        _e("img_q_choice", session_id=1, choice="answer", via="button"),
        _e("img_q_offered", session_id=2, is_onboarding=True),
        _e("img_q_choice", session_id=2, choice="decline", via="text"),
        _e("img_q_offered", session_id=3, is_onboarding=False),  # застрял без выбора
    ]
    res = offer_choice_split(events)
    assert res["offered"] == 3 and res["answer"] == 1 and res["decline"] == 1
    assert res["no_choice"] == 1
    assert res["by_via"]["button"] == 1 and res["by_via"]["text"] == 1


def test_answered_histogram():
    events = [
        _e("img_q_flow_done", session_id=1, answered=3, skipped=0, total=3),
        _e("img_q_flow_done", session_id=2, answered=3, skipped=0, total=3),
        _e("img_q_flow_done", session_id=3, answered=1, skipped=2, total=3),
    ]
    assert answered_histogram(events) == {(3, 3): 2, (1, 3): 1}


def test_drop_by_question():
    events = [
        _e("img_q_choice", session_id=1, choice="answer", via="button"),
        _e("img_q_answered", session_id=1, idx=0, total=3, image="Отец", via="text"),
        # flow_done нет → бросил на вопросе 2 (idx 0 отвечен, следующий — №2)
        _e("img_q_choice", session_id=2, choice="answer", via="button"),
        # ни одного answered/skipped → бросил на вопросе 1
        _e("img_q_choice", session_id=3, choice="answer", via="button"),
        _e("img_q_answered", session_id=3, idx=0, total=2, image="A", via="text"),
        _e("img_q_answered", session_id=3, idx=1, total=2, image="B", via="text"),
        _e("img_q_flow_done", session_id=3, answered=2, skipped=0, total=2),  # дошёл
    ]
    assert drop_by_question(events) == {2: 1, 1: 1}
