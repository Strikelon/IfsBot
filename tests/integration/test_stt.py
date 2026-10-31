from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm.stt import TranscriptionResult, transcribe_voice


@pytest.mark.asyncio
async def test_transcribe_voice_calls_openai_whisper() -> None:
    fake_response = SimpleNamespace(text="мне снился лес", duration=12.5)
    fake_client = AsyncMock()
    fake_client.audio.transcriptions.create = AsyncMock(return_value=fake_response)

    result = await transcribe_voice(
        client=fake_client,
        audio_bytes=b"\x00\x01\x02",
        filename="voice.ogg",
        language="ru",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.text == "мне снился лес"
    assert result.duration_seconds == pytest.approx(12.5)

    fake_client.audio.transcriptions.create.assert_awaited_once()
    call_kwargs = fake_client.audio.transcriptions.create.await_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["language"] == "ru"
    assert call_kwargs["response_format"] == "verbose_json"
    # file — это tuple (filename, bytes, content_type)
    file_arg = call_kwargs["file"]
    assert file_arg[0] == "voice.ogg"
    assert file_arg[1] == b"\x00\x01\x02"
    assert file_arg[2] == "audio/ogg"


@pytest.mark.asyncio
async def test_transcribe_voice_handles_missing_duration() -> None:
    """Если verbose_json не вернул duration — fallback в 0.0 (защита от падения)."""
    fake_response = SimpleNamespace(text="короткое", duration=None)
    fake_client = AsyncMock()
    fake_client.audio.transcriptions.create = AsyncMock(return_value=fake_response)

    result = await transcribe_voice(
        client=fake_client,
        audio_bytes=b"x",
        filename="voice.ogg",
        language="ru",
    )

    assert result.text == "короткое"
    assert result.duration_seconds == 0.0
