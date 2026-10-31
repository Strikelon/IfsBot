"""render_quick_markup: разметка LLM-вывода в безопасный Telegram-HTML.

LLM размечает цитату сна строкой `> ...`, финальный вопрос — `**...**`.
Рендерер экранирует весь текст и конвертирует маркеры в `<blockquote>`/`<b>`.
"""

from adapters.telegram.quick_markup import render_quick_markup


def test_plain_text_is_escaped_without_tags() -> None:
    assert render_quick_markup("связь X < Y & Z") == "связь X &lt; Y &amp; Z"


def test_bold_final_question() -> None:
    out = render_quick_markup("Резюме текста.\n\n**Хотите разобрать A или B?**")
    assert out == "Резюме текста.\n\n<b>Хотите разобрать A или B?</b>"


def test_blockquote_single_line() -> None:
    out = render_quick_markup("До цитаты\n> мне снился лес\nПосле цитаты")
    assert out == "До цитаты\n<blockquote>мне снился лес</blockquote>\nПосле цитаты"


def test_blockquote_multiline_groups_into_one() -> None:
    out = render_quick_markup("> строка один\n> строка два")
    assert out == "<blockquote>строка один\nстрока два</blockquote>"


def test_blockquote_inner_content_is_escaped() -> None:
    out = render_quick_markup("> связь <X> и <Y>")
    assert out == "<blockquote>связь &lt;X&gt; и &lt;Y&gt;</blockquote>"


def test_bold_content_is_escaped() -> None:
    out = render_quick_markup("**a < b**")
    assert out == "<b>a &lt; b</b>"


def test_raw_html_in_text_is_not_executable() -> None:
    out = render_quick_markup("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_no_markers_equals_plain_escape() -> None:
    from html import escape

    text = "Обычный разбор без разметки.\n\nВторой абзац."
    assert render_quick_markup(text) == escape(text)


# --- Нормализация пунктуации (тире → среднее «–», кавычки → прямые) ---


def test_em_dash_becomes_en_dash() -> None:
    assert render_quick_markup("он молчал — долго") == "он молчал – долго"


def test_en_dash_preserved() -> None:
    # среднее тире — целевой символ, остаётся как есть
    assert render_quick_markup("стр. 2–3") == "стр. 2–3"


def test_horizontal_bar_becomes_en_dash() -> None:
    assert render_quick_markup("реплика ― ответ") == "реплика – ответ"


def test_guillemets_become_straight_quotes() -> None:
    # прямая " после html.escape отдаётся как &quot; (Telegram рисует её как ")
    assert render_quick_markup("назвал «стража»") == "назвал &quot;стража&quot;"


def test_curly_double_quotes_become_straight() -> None:
    assert render_quick_markup("назвал “стража”") == "назвал &quot;стража&quot;"


def test_low_double_quote_becomes_straight() -> None:
    assert render_quick_markup("„цитата“") == "&quot;цитата&quot;"


def test_curly_apostrophe_is_not_normalized() -> None:
    # апостроф / одиночная кавычка — не кавычки, не трогаем
    assert render_quick_markup("don’t") == "don’t"


def test_dash_normalized_inside_blockquote_marker_intact() -> None:
    out = render_quick_markup("> я шёл — долго")
    assert out == "<blockquote>я шёл – долго</blockquote>"


def test_quotes_normalized_inside_blockquote() -> None:
    out = render_quick_markup("> он сказал «беги»")
    assert out == "<blockquote>он сказал &quot;беги&quot;</blockquote>"


def test_dash_inside_bold_marker_intact() -> None:
    out = render_quick_markup("**идём дальше — да?**")
    assert out == "<b>идём дальше – да?</b>"
