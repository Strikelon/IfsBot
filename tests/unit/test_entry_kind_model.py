"""dreams.kind: дефолт, create с типом, фильтр списка."""

from datetime import UTC, datetime

import pytest

from data.repositories import DreamRepository, UserRepository
from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION


@pytest.mark.asyncio
async def test_dream_kind_defaults_to_dream(test_session):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=1)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="тигр"
    )
    assert dream.kind == KIND_DREAM


@pytest.mark.asyncio
async def test_create_with_kind_and_list_filter(test_session):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=2)
    repo = DreamRepository(test_session)
    now = datetime.now(UTC)
    await repo.create(user_id=user.id, recorded_at=now, transcription="сон", kind=KIND_DREAM)
    sit = await repo.create(
        user_id=user.id, recorded_at=now, transcription="ссора", kind=KIND_SITUATION
    )
    conf = await repo.create(
        user_id=user.id, recorded_at=now, transcription="лень", kind=KIND_CONFLICT
    )

    dreams = await repo.list_by_user(user_id=user.id, kind=KIND_DREAM)
    situations = await repo.list_by_user(user_id=user.id, kind=KIND_SITUATION)
    conflicts = await repo.list_by_user(user_id=user.id, kind=KIND_CONFLICT)
    everything = await repo.list_by_user(user_id=user.id)

    assert [d.transcription for d in dreams] == ["сон"]
    assert [d.id for d in situations] == [sit.id]
    assert [d.id for d in conflicts] == [conf.id]
    assert len(everything) == 3  # kind=None — обратная совместимость: все обращения
