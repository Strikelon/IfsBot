"""Rolling summary для Quick Q&A через Flash-модель.

Опция C из плана 2026-05-19: после 8+ turns старые пары сжимаются в short
summary и хранятся в session_state.data. Следующие follow-up'ы инжектят
summary в system и режут историю до последних 4 пар. Стоимость каждого
Flash-вызова ~$0.001 (Gemini 2.5 Flash).
"""

from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from llm.prompts.quick_summary import SUMMARY_PROMPT_VERSION, build_summary_messages

# Порог — после скольки накопленных turns (включая initial) начинаем суммаризацию.
SUMMARY_THRESHOLD = 8
# Сколько последних пар оставлять в полном виде (не сжимать).
RECENT_KEEP = 4

# Где хранить summary в session_state.data
SUMMARY_KEY = "quick_qa_summary"

log = structlog.get_logger(__name__)


async def compute_summary(
    *,
    client: AsyncOpenAI,
    model: str,
    qa_pairs_to_compress: list[tuple[str, str]],
    previous_summary: str | None = None,
) -> tuple[str, int, int]:
    """Вызвать Flash для генерации/обновления summary.

    Returns:
        (summary_text, input_tokens, output_tokens).
    """
    messages = build_summary_messages(qa_pairs_to_compress, previous_summary=previous_summary)
    log.info(
        "quick.summary.start",
        model=model,
        n_pairs=len(qa_pairs_to_compress),
        has_previous=previous_summary is not None,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # низкая — нам нужна фактологичность, не креатив
        extra_body={"usage": {"include": True}},
    )
    usage = getattr(response, "usage", None)
    text = (response.choices[0].message.content if response.choices else "") or ""
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    log.info(
        "quick.summary.done",
        model=model,
        prompt_version=SUMMARY_PROMPT_VERSION,
        text_len=len(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return text, input_tokens, output_tokens


def should_summarize(qa_history_len: int) -> bool:
    """Триггер: считаем ли мы summary на этом turn'е.

    Пересчитываем каждый turn начиная с SUMMARY_THRESHOLD — это просто, и
    стоимость Flash-вызова небольшая (~$0.001). Альтернатива «раз в N turns»
    усложняет код без существенной экономии.
    """
    return qa_history_len >= SUMMARY_THRESHOLD


def split_history_for_summary(
    qa_pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Разрезать историю на «сжать» (всё кроме последних RECENT_KEEP) и «оставить как есть».

    Returns: (to_compress, to_keep_verbatim).
    """
    if len(qa_pairs) <= RECENT_KEEP:
        return [], qa_pairs
    return qa_pairs[:-RECENT_KEEP], qa_pairs[-RECENT_KEEP:]
