"""Константы предметной области, разделяемые между data-слоем и адаптерами.

Вынесены сюда, чтобы избежать циклического импорта:
  adapters.telegram.session_log → data.repositories (ОК)
  data.repositories → adapters.telegram.session_log (цикл — НЕЛЬЗЯ)

Любой модуль в data/ и adapters/ может безопасно импортировать отсюда.
"""

# llm_phase_label-конвенция Quick-сессии (см. session_log.py)
QUICK_PHASE_INITIAL = "initial"
QUICK_PHASE_QA = "qa"
QUICK_PHASE_IMAGE_Q = "image_q"  # пре-initial вопросы по образам (фича image questions)

# Маркер, которым user_replies_with_prompts помечает вопросы фазы image_q для
# экстрактора памяти (memory.extract.v3): ответ юзера на такой вопрос — про его
# РЕАЛЬНУЮ жизнь, а не пересказ сна, даже если в вопросе упомянут образ из сна.
# Без пометки экстрактор дропал заземляющие ответы как dream-talk (дев-смоук
# 2026-07-06: «ассоциация с моей дочкой» не попадала в память).
GROUNDING_QUESTION_MARKER = "[Прямой вопрос о РЕАЛЬНОЙ жизни юзера, НЕ про сам сон] "

# Фаза пре-initial вопросов для НЕ-сновых типов (ситуация/конфликт). У снов —
# QUICK_PHASE_IMAGE_Q; отдельная фаза нужна, чтобы экстрактор памяти различал
# source="qa_grounding" (сны) и source="pre_questions" (новые типы), спека A2/B3.
QUICK_PHASE_PRE_Q = "pre_q"

# Маркер вопросов фазы pre_q для user_replies_with_prompts → архивариус кладёт
# ответы с source="pre_questions". Ответ юзера здесь — о его реальной жизни.
PRE_QUESTION_MARKER = "[Уточняющий вопрос перед разбором; ответ юзера – о его РЕАЛЬНОЙ жизни] "
