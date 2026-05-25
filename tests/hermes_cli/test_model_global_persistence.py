from __future__ import annotations

import yaml

from hermes_cli.model_switch import ModelSwitchResult


class _StubCLI:
    agent = None
    model = "old-model"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = ""
    _explicit_base_url = ""
    api_mode = "chat_completions"
    _pending_model_switch_note = ""


def test_model_global_switch_persists_base_url(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  default: old-model\n"
        "  provider: openrouter\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *args, **kwargs: None)

    result = ModelSwitchResult(
        success=True,
        new_model="gpt-5.4",
        target_provider="custom:packy",
        provider_changed=True,
        api_key="sk-packy",
        base_url="https://packy.example.com/v1",
        api_mode="chat_completions",
    )

    cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, True)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["default"] == "gpt-5.4"
    assert saved["model"]["provider"] == "custom:packy"
    assert saved["model"]["base_url"] == "https://packy.example.com/v1"
