"""Tests for STEP 01 configuration, hashing, and structured logging utilities."""

import json
import logging
from pathlib import Path

import pytest

from src.utils.config import ConfigurationError, load_config, resolve_project_path
from src.utils.hashing import sha256_file
from src.utils.logging import JsonFormatter, configure_logging


def test_sha256_file_known_value(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"vantara")

    assert sha256_file(path) == "c65ac45d03de756c8156903e7dee558ddf3cdb920eee7ffd78e80dc815801d35"


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        load_config(path)


def test_load_config_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_resolve_project_path_uses_explicit_root(tmp_path: Path) -> None:
    assert (
        resolve_project_path("data/raw.xlsx", project_root=tmp_path)
        == (tmp_path / "data" / "raw.xlsx").resolve()
    )


def test_json_formatter_emits_context() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="loaded",
        args=(),
        exc_info=None,
    )
    record.event = "sample"
    record.rows = 4

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "loaded"
    assert payload["event"] == "sample"
    assert payload["rows"] == 4


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("NOT_A_LEVEL")
