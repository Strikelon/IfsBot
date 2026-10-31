import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.bot import create_bot, create_dispatcher
from app.config import Settings
from app.deps import AppDependencies, create_dependencies, dispose_dependencies
from app.logging import setup_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan: создаёт зависимости, стартует aiogram polling, на shutdown — чистка."""
    deps: AppDependencies = app.state.deps
    log.info("bot.env", mode="dev" if deps.settings.bot_is_dev else "prod")
    bot = create_bot(
        deps.settings.effective_bot_token,
        proxy=deps.settings.llm_proxy,
        send_rate_per_sec=deps.settings.tg_send_rate_per_sec,
    )
    dp = create_dispatcher(
        session_factory=deps.session_factory,
        workflow_data={
            "openai_client": deps.openai_client,
            "openrouter_client": deps.openrouter_client,
            "openrouter_generator_model": deps.settings.openrouter_generator_model,
            "openrouter_classifier_client": deps.openrouter_classifier_client,
            "openrouter_classifier_model": deps.settings.openrouter_classifier_model,
            "openrouter_intent_model": deps.settings.openrouter_intent_model,
            "openrouter_context_model": deps.settings.openrouter_context_model,
            "settings": deps.settings,
        },
    )
    polling_task: asyncio.Task[None] | None = None

    if app.state.start_bot:
        log.info("bot.polling.start")
        await bot.delete_webhook(drop_pending_updates=False)
        polling_task = asyncio.create_task(dp.start_polling(bot, handle_signals=False))

    app.state.bot = bot
    app.state.dispatcher = dp
    try:
        yield
    finally:
        if polling_task is not None:
            log.info("bot.polling.stop")
            await dp.stop_polling()
            polling_task.cancel()
            try:
                await polling_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.warning("bot.polling.teardown_error", exc_info=True)
        await bot.session.close()
        await dispose_dependencies(deps)


def create_app(
    *,
    start_bot: bool = True,
    settings: Settings | None = None,
    deps: AppDependencies | None = None,
) -> FastAPI:
    """Создать FastAPI-приложение.

    `start_bot=False` в тестах.
    `deps` можно передать вручную в тестах (с in-memory engine).
    """
    settings = settings or Settings()
    setup_logging(level=settings.log_level)

    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn.get_secret_value(), traces_sample_rate=0.0)

    deps = deps or create_dependencies(settings)

    app = FastAPI(title="IfsBot", version="0.1.0", lifespan=_lifespan)
    app.state.settings = settings
    app.state.deps = deps
    app.state.start_bot = start_bot

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def __getattr__(name: str) -> object:
    """Lazy module-level `app` for `uvicorn app.main:app`."""
    if name == "app":
        return create_app(start_bot=True)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
