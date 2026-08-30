"""Tests for pipeline.constants -- the env-overridable knob parsing (#135)."""

from __future__ import annotations

import pytest

from pipeline.constants import _json_env


def test_json_env_returns_parsed_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("METADATA_FILTER_STATUS_KEYWORDS", raising=False)
    assert _json_env("METADATA_FILTER_STATUS_KEYWORDS", '{"cancelled": "Cancelled"}') == {
        "cancelled": "Cancelled"
    }


def test_json_env_returns_parsed_default_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METADATA_FILTER_STATUS_KEYWORDS", "")
    assert _json_env("METADATA_FILTER_STATUS_KEYWORDS", "{}") == {}


def test_json_env_parses_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METADATA_FILTER_DOCTYPE_KEYWORDS", '{"Agreement": ["agreement", "mou"]}')
    parsed = _json_env("METADATA_FILTER_DOCTYPE_KEYWORDS", "{}")
    assert parsed == {"Agreement": ["agreement", "mou"]}


def test_json_env_fails_fast_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METADATA_FILTER_STATUS_KEYWORDS", "{not valid")
    with pytest.raises(ValueError, match="METADATA_FILTER_STATUS_KEYWORDS is not valid JSON"):
        _json_env("METADATA_FILTER_STATUS_KEYWORDS", "{}")


def test_shipped_defaults_are_behaviour_preserving() -> None:
    """The move from hardcoded constants to config must not change the shipped vocab."""
    from pipeline.constants import (
        METADATA_FILTER_DOCTYPE_KEYWORDS,
        METADATA_FILTER_STATUS_KEYWORDS,
    )

    assert METADATA_FILTER_DOCTYPE_KEYWORDS == {
        "Contract": ["contract"],
        "Terms and Conditions": ["terms and conditions"],
    }
    assert METADATA_FILTER_STATUS_KEYWORDS == {
        "cancelled": "Cancelled",
        "active": "Active",
        "unsigned": "Unsigned",
    }
