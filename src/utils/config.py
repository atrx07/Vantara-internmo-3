"""YAML configuration loading and project-relative path resolution."""

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a required project configuration value is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and require a mapping at its root."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def resolve_project_path(value: str | Path, *, project_root: str | Path | None = None) -> Path:
    """Resolve a configured path relative to the repository root or current directory."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    base = Path(project_root) if project_root is not None else Path.cwd()
    return (base / candidate).resolve()
