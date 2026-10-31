from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from domain.access import normalize_exempt
from llm.memory_select import MEMORY_BLOCK_BUDGET


class Settings(BaseSettings):
    """Глобальный конфиг приложения. Загружается из переменных окружения и `.env`.

    Все секреты — через `SecretStr`, чтобы случайно не залогировать.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    bot_token: SecretStr = Field(..., description="Telegram Bot API token (prod)")
    bot_dev_token: SecretStr | None = Field(
        None,
        description=(
            "Опциональный токен dev-бота для локальной отладки. Если задан — бот стартует на нём "
            "вместо BOT_TOKEN. На проде не задаётся (deploy.py его не пишет в серверный .env)."
        ),
    )
    database_url: str = Field(..., description="SQLAlchemy async URL")
    log_level: str = Field("INFO", description="Уровень логирования: DEBUG/INFO/WARNING/ERROR")
    webhook_enabled: bool = Field(
        False, description="True для production webhook; False — long-polling (dev/Phase 0)"
    )
    webhook_base_url: str | None = Field(
        None,
        description="Публичный HTTPS URL для webhook (используется только при webhook_enabled=True)",
    )

    # --- Phase 1: STT через OpenAI Whisper ---
    openai_api_key: SecretStr = Field(..., description="OpenAI API key (используется для Whisper)")
    llm_proxy: SecretStr | None = Field(
        None,
        description=(
            "Опциональный HTTP/HTTPS-прокси для исходящих AI-вызовов (Whisper, в будущем OpenRouter). "
            "Формат: 'http://user:pass@host:port'. Пусто = без прокси (dev вне РФ). "
            "На РФ-VPS обязателен — OpenAI с РФ-IP заблокирован."
        ),
    )

    # --- Phase 2: OpenRouter для генератора (Gemini 3 Pro) ---
    openrouter_api_key: SecretStr = Field(
        ...,
        description="OpenRouter API key. Используется для Gemini 3 Pro (generator) и Flash (classifier, Phase 3+).",
    )
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1",
        description="OpenRouter API base URL. Меняем только при перетягивании на другой провайдер.",
    )
    openrouter_generator_model: str = Field(
        "google/gemini-3-pro",
        description=(
            "Slug модели OpenRouter для generator (Quick analysis в Phase 2, P4-P6 generator в Phase 3+). "
            "Сверять со списком на openrouter.ai/models перед запуском."
        ),
    )
    openrouter_classifier_model: str = Field(
        "google/gemini-2.5-flash",
        description=(
            "Slug модели OpenRouter для classifier'а (P2_FIND, SELF_GATE, soft Self-check в Phase 3+). "
            "Дешёвая быстрая модель. Сверять со списком на openrouter.ai/models перед запуском."
        ),
    )
    openrouter_intent_model: str = Field(
        "openai/gpt-4o-mini",
        description=(
            "Slug дешёвой модели OpenRouter для intent-классификаторов (продолжение/отказ в "
            "loop'ах, завершение Q&A, разобрать/сохранить, сжатие длинного кастомного запроса). "
            "Отдельно от classifier'а — здесь нужна минимальная цена при простой задаче «понять намерение»."
        ),
    )
    openrouter_context_model: str = Field(
        "google/gemini-3.1-flash-lite",
        description=(
            "Slug модели OpenRouter для конвейеров памяти: экстрактор единиц "
            "(memory.extract на паузе сессии) и компоновка самоописания в онбординге "
            "(context.compact). Строгий JSON + защиты в коде. "
            "Выбрана по сухому прогону на прод-данных 2026-08-15: единственная из "
            "трёх, у кого работает обновление ядровых фактов (replaces_outdated), "
            "виды единиц ожили, штатного потолка 1200 токенов хватает. "
            "Тариф прописан в llm/pricing.py (gemini_31_flash_lite_cost_usd) — при "
            "смене модели обновлять и там, иначе лимит трат юзера будет врать."
        ),
    )

    # --- Долговременная память о юзере (user_memory_items) ---
    memory_write_enabled: bool = Field(
        True,
        description=(
            "Экстрактор памяти на паузе сессии (shadow-режим): пишем единицы в "
            "user_memory_items. False = полное выключение записи памяти."
        ),
    )
    memory_inject_enabled: bool = Field(
        True,
        description=(
            "Подстановка единиц памяти в промпты разбора/Q&A (quick.v3.13). "
            "Включена после GATE-проверки накопленной памяти глазами (Task 7)."
        ),
    )
    memory_compact_enabled: bool = Field(
        True,
        description=(
            "Уплотнение памяти («сухой остаток») после экстракции на паузе: "
            "разрозненные ячейки одной темы сливаются в одну плотную запись. "
            "False = kill-switch, память копится как раньше."
        ),
    )
    memory_compact_budget_chars: int = Field(
        MEMORY_BLOCK_BUDGET,
        description=(
            "Лимит блока памяти в символах, по которому запускается уплотнение "
            "проход идёт не после каждого разбора, "
            "а только когда память перестала помещаться в этот бюджет и часть "
            "фактов уже не доезжает до промпта. Значение по умолчанию — тот же "
            "бюджет, с которым блок реально собирается (llm/memory_select.py)."
        ),
    )

    # --- Вопросы по образам сна перед разбором (image questions) ---
    image_questions_enabled: bool = Field(
        True,
        description=(
            "Опциональные заземляющие вопросы по образам сна перед initial-разбором "
            "(offer-first, оба потока: обычный + онбординг). False = kill-switch: "
            "поведение обоих потоков байт-в-байт как до фичи."
        ),
    )

    # --- Concurrency gates (queue, NOT per-user rate-limit) ---
    llm_max_concurrency: int = Field(
        default=30,
        description="Макс. одновременных in-flight LLM-вызовов (вентиль). 0 = без лимита.",
    )
    stt_max_concurrency: int = Field(
        default=10,
        description="Макс. одновременных Whisper STT-вызовов (отдельный ключ).",
    )
    tg_send_rate_per_sec: int = Field(
        default=25,
        description=(
            "Макс. исходящих Telegram Bot API вызовов в секунду (token-bucket throttle). "
            "Telegram режет ~30/сек глобально на бота; дефолт 25 оставляет запас."
        ),
    )

    # --- Spend limit (per-user neural-cost cap; paywall foundation) ---
    user_spend_limit_usd: float = Field(
        1.0,
        description=(
            "Порог для тира 'early_tester' (первые тестеры) — кумулятивные траты на нейросеть "
            "на пользователя (USD). При достижении платные действия блокируются. "
            "Меняется без правок кода (operator + restart)."
        ),
    )
    new_user_spend_limit_usd: float = Field(
        1.0,
        description=(
            "Порог кумулятивных трат для тира 'new' — всех, кто пришёл после запуска рекламы "
            "(2026-08-01). Первые тестеры остаются на USER_SPEND_LIMIT_USD. "
            "Батч 2026-08-18: поднят с $0.5 до $1 — на $0.5 не помещался даже один разбор "
            "($0.44–0.63), человек упирался в пейвол, не увидев результата."
        ),
    )
    admin_id: int | None = Field(
        default=None,
        description=(
            "Telegram ID владельца бота. Только ему доступна команда /interview — "
            "выдать человеку безлимит, когда админка под рукой не открыта. "
            "Та же переменная, что читает админка; deploy.py уже везёт её на сервер."
        ),
    )
    support_contact_url: str = Field(
        ...,
        description=(
            "Куда писать человеку, упёршемуся в лимит, — ссылка в тексте пейвола. "
            "Напр.: 'https://t.me/your_username'."
        ),
    )
    spend_limit_exempt_usernames: str = Field(
        "",
        description=(
            "Comma-separated Telegram usernames, освобождённые от лимита (case-insensitive, "
            "ведущий @ снимается). Напр.: 'owner_username,partner_username'."
        ),
    )

    # --- Observability ---
    sentry_dsn: SecretStr | None = Field(
        default=None,
        description=(
            "Sentry DSN для отправки ошибок и трейсов. Если не задан — Sentry не инициализируется "
            "(полный no-op, пакет sentry-sdk не обязан быть установлен в dev/тестах). "
            "Задаётся в проде через переменную окружения SENTRY_DSN."
        ),
    )

    @field_validator("llm_proxy", "bot_dev_token", "sentry_dsn", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @property
    def spend_limit_exempt_set(self) -> frozenset[str]:
        """Нормализованный набор exempt-юзернеймов (lowercase, без @, без пустых)."""
        return normalize_exempt(self.spend_limit_exempt_usernames)

    @property
    def bot_is_dev(self) -> bool:
        """True, если запущены на dev-боте (BOT_DEV_TOKEN задан и непустой)."""
        return self.bot_dev_token is not None

    @property
    def effective_bot_token(self) -> SecretStr:
        """Токен, на котором реально стартует бот.

        Dev имеет приоритет: задан BOT_DEV_TOKEN → работаем на dev-боте (локальная отладка).
        Пусто/нет → BOT_TOKEN (прод). На проде BOT_DEV_TOKEN не пишется, поэтому там всегда prod.
        """
        return self.bot_dev_token or self.bot_token
