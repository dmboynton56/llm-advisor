"""Tests for infra/gcp/trigger-github-workflow Cloud Function dispatch."""

from __future__ import annotations

import os
import sys
import types
import urllib.error
from pathlib import Path
from unittest import mock

TRIGGER_DIR = (
    Path(__file__).resolve().parents[2] / "infra" / "gcp" / "trigger-github-workflow"
)
if str(TRIGGER_DIR) not in sys.path:
    sys.path.insert(0, str(TRIGGER_DIR))

functions_framework = types.ModuleType("functions_framework")
functions_framework.http = lambda function: function
sys.modules.setdefault("functions_framework", functions_framework)

import main  # noqa: E402


def test_success_returns_204():
    response = mock.MagicMock()
    response.__enter__.return_value.status = 204

    with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "token"}), mock.patch(
        "main.urllib.request.urlopen", return_value=response
    ):
        status, message = main._dispatch("live")

    assert status == 204
    assert "Triggered live" in message


def test_github_503_returns_502():
    error = urllib.error.HTTPError(
        "https://api.github.com", 503, "Service Unavailable", {}, None
    )
    error.read = mock.Mock(return_value=b"temporarily unavailable")

    with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "token"}), mock.patch(
        "main.urllib.request.urlopen", side_effect=error
    ):
        status, message = main._dispatch("live")

    assert status == 502
    assert "temporarily unavailable" in message


def test_missing_token_returns_500():
    with mock.patch.dict(os.environ, {}, clear=True):
        status, message = main._dispatch("live")

    assert status == 500
    assert message == "GITHUB_TOKEN is not configured."


def test_unexpected_exception_returns_502():
    error = urllib.error.URLError("connection reset")

    with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "token"}), mock.patch(
        "main.urllib.request.urlopen", side_effect=error
    ):
        status, message = main._dispatch("live")

    assert status == 502
    assert message == "GitHub API request failed."


def test_unknown_workflow_returns_400():
    status, message = main._dispatch("unknown")

    assert status == 400
    assert "Unknown workflow" in message
