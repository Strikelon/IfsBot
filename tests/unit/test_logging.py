import json
import logging

import pytest
import structlog

from app.logging import setup_logging


def test_setup_logging_configures_structlog_json(capsys: pytest.CaptureFixture) -> None:
    setup_logging(level="INFO")

    log = structlog.get_logger("test")
    log.info("hello", user_id=42, action="ping")

    captured = capsys.readouterr()
    # structlog в JSON-режиме пишет в stdout
    line = captured.out.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["event"] == "hello"
    assert parsed["user_id"] == 42
    assert parsed["action"] == "ping"
    assert parsed["level"] == "info"


def test_setup_logging_respects_level(capsys: pytest.CaptureFixture) -> None:
    setup_logging(level="WARNING")

    log = structlog.get_logger("test")
    log.info("should_be_filtered")
    log.warning("should_appear")

    out = capsys.readouterr().out
    assert "should_be_filtered" not in out
    assert "should_appear" in out


@pytest.fixture(autouse=True)
def _reset_structlog():
    yield
    structlog.reset_defaults()
    # Очистить root logger handlers, чтобы тесты не влияли друг на друга
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
