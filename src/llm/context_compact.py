"""Компоновка (сжатие) разросшегося самоописания юзера — context.compact.

Память v2 (2026-08-15): profile_life_context — ТОЛЬКО самоописание юзера;
конвейер update_life_context удалён, а компоновка — часть пути «пишется из
онбординга» (уплотняет то, что юзер сам надиктовал в LIFE_CONTEXT_LOOP) —
переехала сюда из удалённого модуля «живой памяти». Текст промпта не менялся,
версия остаётся context.compact.v1.

Безопасный дефолт при ЛЮБОМ сбое (сеть/таймаут/парсинг/подозрительный вывод):
НЕ менять текст (changed=False, tokens=0).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from llm.classifier import ClassifierParseError, classify

log = structlog.get_logger(__name__)

CONTEXT_COMPACT_VERSION = "context.compact.v1"

# Порог запуска компоновки: если контекст длиннее — сжимаем (компоновка, не обрезание).
COMPACT_TRIGGER_CHARS = 1200
# Подозрительно агрессивное сжатие (потеря фактов) — отклоняем.
_COMPACT_MIN_RATIO = 0.25
_COMPACT_MIN_CHARS = 80


class _Compacted(BaseModel):
    compacted: str


@dataclass(frozen=True)
class CompactResult:
    """Результат компоновки (сжатия) разросшегося контекста + usage.

    changed=True только если компоновка прошла все защиты и реально укоротила текст.
    При сбое/под-пороге/подозрительном выводе — changed=False, текст оставляем прежним.
    """

    changed: bool
    new_text: str | None
    input_tokens: int
    output_tokens: int
    reject_reason: str | None = None


_COMPACT_SYSTEM = """\
Ты сжимаешь «общий контекст о жизни пользователя» — короткую анкету фактов от первого \
лица, — которая разрослась. Перепиши её КОРОЧЕ и аккуратнее (до ~1200 символов), \
СОХРАНИВ ВСЕ важные факты жизни. Это компоновка, а НЕ обрезание.

Правила:
- Объединяй повторы и близкие по смыслу куски, убирай воду и дубли.
- НИ ОДИН важный факт жизни не должен пропасть: переезд/эмиграция, семья/брак/развод/\
дети, работа/потеря или смена работы/профессия, учёба, своё дело и проекты, серьёзные \
изменения здоровья и прочие крупные обстоятельства.
- Ничего НЕ выдумывай и не добавляй от себя — только то, что уже есть в тексте.
- Сохраняй первое лицо и нейтральный фактологичный стиль.
- Не интерпретируй, не оценивай, не давай советов.

Отвечай ТОЛЬКО строгим JSON без markdown и ```-фенсов:
{"compacted": "сжатый текст контекста"}\
"""


def _compact_no_op(reason: str) -> CompactResult:
    return CompactResult(changed=False, new_text=None, input_tokens=0, output_tokens=0,
                         reject_reason=reason)


async def compact_life_context(
    *, client: AsyncOpenAI, model: str, text: str
) -> CompactResult:
    """Сжать разросшееся самоописание (компоновка, не обрезание).

    Запускать, когда текст превысил COMPACT_TRIGGER_CHARS. Никогда не бросает: при
    любом сбое/под-пороге/подозрительном выводе возвращает no-op — исходный текст остаётся.
    """
    src = (text or "").strip()
    if len(src) <= COMPACT_TRIGGER_CHARS:
        return _compact_no_op("under_threshold")

    messages = [
        {"role": "system", "content": _COMPACT_SYSTEM},
        {"role": "user", "content": f"ТЕКУЩИЙ ОБЩИЙ КОНТЕКСТ:\n«{src}»"},
    ]
    try:
        res = await classify(client=client, model=model, messages=messages, schema=_Compacted)
    except ClassifierParseError:
        log.warning("context_compact.parse_failed")
        return _compact_no_op("parse_failed")
    except Exception:  # noqa: BLE001 — сеть/таймаут/прочее: не роняем онбординг
        log.warning("context_compact.call_failed", exc_info=True)
        return _compact_no_op("call_failed")

    new_text = (res.parsed.compacted or "").strip()
    in_tok, out_tok = res.input_tokens, res.output_tokens

    # --- Защиты ---
    if not new_text:
        return CompactResult(False, None, in_tok, out_tok, "empty")
    if len(new_text) >= len(src):
        # Не укоротила — смысла применять нет, оставляем исходный.
        return CompactResult(False, None, in_tok, out_tok, "not_shorter")
    if len(new_text) < _COMPACT_MIN_CHARS or len(new_text) < len(src) * _COMPACT_MIN_RATIO:
        log.warning("context_compact.suspicious_shrink", old_len=len(src), new_len=len(new_text))
        return CompactResult(False, None, in_tok, out_tok, "suspicious_shrink")

    log.info("context_compact.applied", old_len=len(src), new_len=len(new_text))
    return CompactResult(
        changed=True, new_text=new_text, input_tokens=in_tok, output_tokens=out_tok,
        reject_reason=None,
    )
