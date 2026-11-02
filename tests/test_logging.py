import io
import json
import logging
from collections.abc import Iterator
from datetime import datetime

import pytest

from urbangraph.logging import configure_logging


@pytest.fixture
def logger_name() -> Iterator[str]:
    name = "urbangraph.tests.json"
    logger = logging.getLogger(name)
    previous = (logger.level, logger.propagate, list(logger.handlers))
    yield name
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(previous[0])
    logger.propagate = previous[1]
    for handler in previous[2]:
        logger.addHandler(handler)


def test_valid_json(logger_name: str) -> None:
    stream = io.StringIO()
    logger = configure_logging(logger_name=logger_name, stream=stream)
    logger.info('sensor %s: "ready"\nnext', "α", extra={"secret": "excluded"})
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {"timestamp", "level", "logger", "message"}
    offset = datetime.fromisoformat(payload["timestamp"]).utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    assert payload["level"] == "INFO"
    assert payload["logger"] == logger_name
    assert payload["message"] == 'sensor α: "ready"\nnext'


def test_reconfiguration_and_level(logger_name: str) -> None:
    first, second = io.StringIO(), io.StringIO()
    logger = configure_logging(logger_name=logger_name, stream=first)
    caller_handler = logging.NullHandler()
    logger.addHandler(caller_handler)
    root_handlers = list(logging.getLogger().handlers)
    assert configure_logging(logger_name=logger_name, stream=second, level="WARNING") is logger
    logger.info("hidden")
    logger.warning("visible")
    assert first.getvalue() == ""
    assert len(second.getvalue().splitlines()) == 1
    assert json.loads(second.getvalue())["message"] == "visible"
    assert caller_handler in logger.handlers
    assert len(logger.handlers) == 2
    assert not logger.propagate
    assert logging.getLogger().handlers == root_handlers


@pytest.mark.parametrize("name", ["", "root"])
def test_root_configuration_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="named logger"):
        configure_logging(logger_name=name)
