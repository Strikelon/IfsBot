"""Разметка вывода генератора разбора в безопасный Telegram-HTML.

Бот работает в `parse_mode=HTML`, а вывод LLM нельзя слать сырым HTML (любой
`<`/`&` в тексте свалит send 400). Поэтому генератор размечает результат
лёгкими маркерами, а мы здесь:
1. экранируем ВЕСЬ текст (`html.escape`) — контент становится безопасным;
2. конвертируем маркеры в намеренные теги:
   - строки, начинающиеся с `> ` (markdown-цитата) → `<blockquote>…</blockquote>`
     (подряд идущие строки объединяются в один блок);
   - `**…**` → `<b>…</b>` (для финального вовлекающего вопроса).

Результат — валидный HTML с уже экранированным контентом; слать через
`answer_long(..., escape_html=False)`.
"""

from __future__ import annotations

import re
from html import escape as html_escape

# После html.escape маркер цитаты `> ` превращается в `&gt; `.
_ESCAPED_QUOTE_PREFIX = "&gt;"
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

# Нормализация пунктуации до экранирования: любые тире → среднее тире «–» (единый
# стиль с захардкоженными текстами бота), типографские кавычки → прямая ". Работает
# по сырому тексту LLM, поэтому ASCII-маркеры разметки (`> ` для цитаты, `**` для
# жирного) не затрагиваются. Апострофы и одиночные кавычки не трогаем — это не кавычки.
_DASH_RE = re.compile(r"[—―]")  # em-dash, horizontal bar → среднее тире «–»
_DQUOTE_RE = re.compile(r"[«»„“”]")  # двойные типографские кавычки → прямая "


def normalize_punctuation(text: str) -> str:
    """Длинные тире → среднее «–», типографские кавычки → прямая `"`.

    Публичная: используется и рендером разбора, и plain-text выводом фичи
    image questions (вопросы, answer классификаторов).
    """
    text = _DASH_RE.sub("–", text)
    return _DQUOTE_RE.sub('"', text)


# Обратная совместимость внутренних вызовов/тестов.
_normalize_punctuation = normalize_punctuation


def _wrap_blockquotes(escaped_text: str) -> str:
    """Подряд идущие строки-цитаты (`&gt; …`) обернуть в один <blockquote>."""
    lines = escaped_text.split("\n")
    out: list[str] = []
    quote_buf: list[str] = []

    def flush() -> None:
        if quote_buf:
            out.append("<blockquote>" + "\n".join(quote_buf) + "</blockquote>")
            quote_buf.clear()

    for line in lines:
        if line.startswith(_ESCAPED_QUOTE_PREFIX):
            inner = line[len(_ESCAPED_QUOTE_PREFIX):]
            if inner.startswith(" "):
                inner = inner[1:]
            quote_buf.append(inner)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def render_quick_markup(text: str) -> str:
    """Преобразовать размеченный вывод генератора в безопасный Telegram-HTML."""
    escaped = html_escape(normalize_punctuation(text))
    with_quotes = _wrap_blockquotes(escaped)
    return _BOLD_RE.sub(r"<b>\1</b>", with_quotes)
