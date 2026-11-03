from __future__ import annotations

from atlas_ai.config.settings import load_settings


def test_load_settings_respects_explicit_env_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "isolated_store"))
    monkeypatch.setenv("TIMEFRAME", "M1")
    monkeypatch.setenv("BACKTEST_INITIAL_BALANCE", "100")
    monkeypatch.setenv("HISTORICAL_TRAINING_STAGE_TIMEOUT_SECONDS", "12")

    settings = load_settings()

    assert settings.storage_dir == str((tmp_path / "isolated_store").resolve())
    assert settings.timeframe == "M1"
    assert settings.backtest_initial_balance == 100.0
    assert settings.historical_training_stage_timeout_seconds == 12.0
