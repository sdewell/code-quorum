import pytest

from quorum import hosts as _hosts_mod
from quorum import model_config as _mc
from quorum import orchestration as _orchestration_mod


@pytest.fixture(autouse=True)
def _reset_version_caches():
    """claude/codex/opencode/agy `<binary> --version` results are cached in
    quorum.model_config.probe_cli_version (functools.cache, shared by every
    seat's own _x_version wrapper) so a council run spawns the probe at most
    once per binary per process. Reset between tests so a monkeypatched or
    fake version from one test can't leak into the next."""
    _mc.probe_cli_version.cache_clear()
    yield
    _mc.probe_cli_version.cache_clear()


@pytest.fixture(autouse=True)
def _reset_model_config_warn_once():
    """Both model_config warnings (env-masks-recorded, version-drift) dedupe
    in module-level sets keyed by seat. Seat factories and agents in
    quorum.agents.{codex,opencode,gemini_cli} also trigger them, not just
    quorum.model_config's own tests -- reset here (suite-wide) so one test's
    warning can't silence another test's assertion regardless of which module
    triggered it."""
    _mc._env_masks_recorded_warned.clear()
    _mc._version_drift_warned.clear()
    _orchestration_mod._sdk_backend_gemini_record_warned.clear()
    yield
    _mc._env_masks_recorded_warned.clear()
    _mc._version_drift_warned.clear()
    _orchestration_mod._sdk_backend_gemini_record_warned.clear()


@pytest.fixture(autouse=True)
def _isolate_model_config_path(tmp_path, monkeypatch):
    """Point model_config.CONFIG_PATH at a per-test tmp path (absent by
    default) so the suite is hermetic against a REAL
    ~/.config/code-quorum/models.toml. mc.CONFIG_PATH is computed once at
    import time from the real `~`, so without this, the day S actually runs
    `quorum setup-models` for real, every test that resolves a seat's
    default model without its own CONFIG_PATH patch (e.g.
    test_codex_seat_pins_gpt56_terra_medium_by_default) would start reading
    -- and in the version-drift path, spawning subprocesses against -- that
    real file instead of the fixtures it's supposed to run against. Tests
    that need a specific recorded choice monkeypatch mc.CONFIG_PATH again on
    top of this; monkeypatch layering means the later setattr just wins,
    same as today."""
    monkeypatch.setattr(_mc, "CONFIG_PATH", tmp_path / "models.toml")


@pytest.fixture(autouse=True)
def _clear_council_env(monkeypatch):
    """A developer shell may export the seat-model knobs the README recommends
    in ~/.zshrc.local: CODE_QUORUM_GEMINI_BACKEND / CODE_QUORUM_GEMINI_MODEL
    (conserve AI Pro quota), CODE_QUORUM_OPENCODE_MODEL (pin the DeepSeek seat
    model across phases), and CODE_QUORUM_CODEX_MODEL / CODE_QUORUM_CODEX_EFFORT
    (pin the codex seat). Any leaking in would make default-backend /
    default-model / per-phase-model assertions environment-dependent, so clear
    them suite-wide. Factory tests opt back in via setenv."""
    monkeypatch.delenv("CODE_QUORUM_GEMINI_BACKEND", raising=False)
    monkeypatch.delenv("CODE_QUORUM_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("CODE_QUORUM_OPENCODE_MODEL", raising=False)
    monkeypatch.delenv("CODE_QUORUM_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CODE_QUORUM_CODEX_EFFORT", raising=False)
    monkeypatch.delenv("CODE_QUORUM_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("CODE_QUORUM_CLAUDE_EFFORT", raising=False)
    monkeypatch.delenv("CODE_QUORUM_HOST", raising=False)
    _hosts_mod._runtime_host = None
    yield
    _hosts_mod._runtime_host = None
