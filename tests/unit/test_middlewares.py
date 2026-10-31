from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.telegram.middlewares import DbSessionMiddleware


@pytest.mark.asyncio
async def test_session_middleware_opens_and_commits_on_success() -> None:
    session = AsyncMock()
    session_factory_cm = AsyncMock()
    session_factory_cm.__aenter__ = AsyncMock(return_value=session)
    session_factory_cm.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=session_factory_cm)

    middleware = DbSessionMiddleware(session_factory)
    handler = AsyncMock(return_value="ok")
    event = object()
    data: dict[str, object] = {}

    result = await middleware(handler, event, data)

    assert result == "ok"
    handler.assert_awaited_once()
    assert data["session"] is session
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_middleware_rollbacks_on_exception() -> None:
    session = AsyncMock()
    session_factory_cm = AsyncMock()
    session_factory_cm.__aenter__ = AsyncMock(return_value=session)
    session_factory_cm.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=session_factory_cm)

    middleware = DbSessionMiddleware(session_factory)
    handler = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await middleware(handler, object(), {})

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
