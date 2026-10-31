"""Промпт для сжатия истории Quick Q&A в короткий summary.

Используется в опции C (rolling summary): после N=8 turns старые пары
(всё кроме последних M=4) суммаризируются через Flash, summary хранится в
session_state.data.quick_qa_summary и инжектится в system follow-up'а.

Цель — не пересказать диалог дословно, а сохранить:
- какие IFS-инсайты уже были (Части, Защитники, Изгнанники, связки с life_context)
- какие темы юзер уже задавал (чтобы модель не повторялась)
- ключевые цитаты из сна, к которым уже привязывали интерпретации
"""

from __future__ import annotations

from typing import Any

SUMMARY_PROMPT_VERSION = "quick_summary.v1"

_SYSTEM_PROMPT = """\
Ты — компрессор IFS-диалога. На вход даётся серия пар (вопрос юзера — ответ \
ассистента-IFS-практика), которые уже произошли при разборе сна. Может также \
прилагаться предыдущий summary (если диалог уже сжимался ранее).

Твоя задача — собрать актуальный summary в 180-250 слов на русском, который \
сохранит контекст для следующих ответов ассистента. В summary включи:

1. Какие IFS-элементы уже названы и обсуждены: Самость, конкретные Части \
(Защитники, Менеджеры, Изгнанники), их роли и связки с образами сна.
2. Какие связки с life_context и per-session запросом юзера уже сделаны.
3. Какие темы и вопросы юзер уже задавал — чтобы ассистент не повторялся.
4. Если был предыдущий summary — интегрируй его, не выкидывая важное.

Стиль:
- Третье лицо, фактологически. «Ассистент отметил, что…», «юзер уточнил про…», \
«была проведена связка между X и Y».
- Без markdown (никаких ##, **, *), только сплошной текст в 2-3 абзацах.
- Без приветствий и заключений — это рабочий технический artifact для другой LLM, \
не для юзера.

ВАЖНО: не интерпретируй сон заново, не добавляй новых наблюдений. Только сжимай \
то, что уже было сказано.
"""


def build_summary_messages(
    qa_pairs_to_compress: list[tuple[str, str]],
    *,
    previous_summary: str | None = None,
) -> list[dict[str, Any]]:
    """Собрать messages для Flash-вызова, который сжимает qa-пары в summary.

    Args:
        qa_pairs_to_compress: пары (вопрос юзера, ответ ассистента) которые надо
            сжать. Первая пара по конвенции — ("[initial]", initial-разбор).
        previous_summary: если на предыдущей итерации уже был summary — передаём,
            чтобы новый summary поглотил старый + добавил то, что между ними.

    Returns:
        Messages для chat.completions.create.
    """
    user_payload_parts: list[str] = []
    if previous_summary:
        user_payload_parts.append(f"ПРЕДЫДУЩИЙ SUMMARY:\n{previous_summary}\n")
    user_payload_parts.append("ДИАЛОГ (пары вопрос→ответ):")
    for q, a in qa_pairs_to_compress:
        if q == "[initial]":
            user_payload_parts.append(f"\n--- Первый IFS-разбор сна ---\n{a}")
        else:
            user_payload_parts.append(f"\n--- Юзер ---\n{q}\n--- Ассистент ---\n{a}")
    user_payload = "\n".join(user_payload_parts)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]
