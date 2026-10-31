"""Регресс-гард конвенции тире: в новых IMG_Q_* константах нет em-dash «—»."""
from adapters.telegram import texts


def test_no_em_dash_in_imgq_constants():
    imgq_values = []
    for name in dir(texts):
        if name.startswith("IMG_Q_"):
            value = getattr(texts, name)
            imgq_values.extend(value if isinstance(value, list) else [value])
    assert imgq_values, "IMG_Q_* константы должны существовать"
    for v in imgq_values:
        assert "—" not in v, f"em-dash в константе: {v!r}"
        assert "―" not in v
