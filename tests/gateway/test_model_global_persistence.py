from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_key_for_source = lambda _source: "session-key"
    runner._evict_cached_agent = lambda _session_key: None
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_gateway_global_switch_clears_stale_base_url(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "old-model",
                    "provider": "openrouter",
                    "base_url": "https://packy.example.com/v1",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: yaml.safe_load(config_path.read_text(encoding="utf-8")) or {},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            new_model="gpt-5.4",
            target_provider="openai",
            api_key="sk-openai",
            base_url="",
            api_mode="chat_completions",
            warning_message="",
            provider_label="OpenAI",
            model_info=None,
        ),
    )

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/model gpt-5.4 --global"))

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert result is not None
    assert saved["model"]["default"] == "gpt-5.4"
    assert saved["model"]["provider"] == "openai"
    assert "base_url" not in saved["model"]
