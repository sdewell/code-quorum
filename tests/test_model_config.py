import logging
from datetime import date
from pathlib import Path

import pytest

from quorum import model_config as mc
from quorum.model_config import (
    ModelConfigError,
    load_config,
    record_choice,
    recorded_choice,
    resolve_model,
)


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """The env-masks-recorded warning is deduped in a module-level set keyed
    by seat -- reset it so one test's warning doesn't silence the next."""
    mc._env_masks_recorded_warned.clear()
    yield
    mc._env_masks_recorded_warned.clear()


def _cfg(tmp_path: Path) -> Path:
    return tmp_path / "models.toml"


# Captured at import time, before conftest.py's autouse _isolate_model_config_path
# fixture monkeypatches mc.CONFIG_PATH for every test (suite hermeticity) --
# this is the real, unpatched default the module computes at import, which is
# what the test below is actually verifying the shape of.
_REAL_CONFIG_PATH = mc.CONFIG_PATH


# --- CONFIG_PATH ----------------------------------------------------------


def test_config_path_is_absolute_under_dot_config() -> None:
    assert _REAL_CONFIG_PATH.is_absolute()
    assert str(_REAL_CONFIG_PATH).endswith(".config/code-quorum/models.toml")


# --- load_config ------------------------------------------------------------


def test_load_config_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert load_config(_cfg(tmp_path)) == {}


def test_load_config_malformed_toml_raises_with_rerun_hint(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text("this is not [[ valid toml =\n", encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert str(path) in message
    assert "quorum setup-models" in message


def test_load_config_missing_model_key_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text('[gemini]\neffort = "medium"\n', encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "model" in message
    assert "quorum setup-models" in message


def test_load_config_claude_requires_model(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[claude]\ncli_version = "2.1.232"\n', encoding="utf-8")

    with pytest.raises(ModelConfigError, match="missing `model`"):
        load_config(path)


def test_load_config_disabled_seat_does_not_require_model(tmp_path: Path) -> None:
    """`quorum setup-models` records a disabled seat as `disabled = "true"`
    with no `model` field. The loader must accept that shape for a known seat."""
    path = tmp_path / "models.toml"
    path.write_text(
        '[gemini]\ndisabled = "true"\nchosen = "2026-08-18"\n', encoding="utf-8"
    )

    config = load_config(path)
    assert config["gemini"] == {"disabled": "true", "chosen": "2026-08-18"}


def test_load_config_bad_codex_effort_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\nmodel = "gpt-5.6-terra"\neffort = "turbo"\n', encoding="utf-8"
    )
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "turbo" in message
    assert "quorum setup-models" in message


def test_load_config_bad_claude_effort_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text(
        '[claude]\nmodel = "claude-opus-5"\neffort = "turbo"\n',
        encoding="utf-8",
    )
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "turbo" in message
    assert "quorum setup-models" in message


def test_load_config_scalar_known_seat_value_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text('codex = "x"\n', encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "codex" in message
    assert "quorum setup-models" in message


def test_load_config_non_string_model_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text("[codex]\nmodel = 1\n", encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "model" in message
    assert "quorum setup-models" in message


def test_load_config_non_string_effort_raises(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text('[codex]\nmodel = "gpt-5.6-terra"\neffort = 1\n', encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "effort" in message
    assert "quorum setup-models" in message


def test_load_config_non_string_metadata_field_raises(tmp_path: Path) -> None:
    # Unlike effort (which also fails the pre-existing enum-membership check),
    # cli_version has no other validation -- this is the field that actually
    # proves the general string-type check, not a coincidence of the enum
    # check rejecting anything that isn't a valid effort string.
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\nmodel = "gpt-5.6-terra"\ncli_version = 150\n', encoding="utf-8"
    )
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "cli_version" in message
    assert "quorum setup-models" in message


def test_load_config_unknown_seat_scalar_value_raises(tmp_path: Path) -> None:
    # A non-table value can't round-trip through the hand-rolled serializer
    # regardless of whether the seat name is recognized, so it's malformed
    # for ANY seat name.
    path = _cfg(tmp_path)
    path.write_text('future_seat = "x"\n', encoding="utf-8")
    with pytest.raises(ModelConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "future_seat" in message
    assert "quorum setup-models" in message


def test_load_config_unknown_seat_tolerated_when_it_has_model(
    tmp_path: Path,
) -> None:
    path = _cfg(tmp_path)
    path.write_text('[future_seat]\nmodel = "some-model"\n', encoding="utf-8")
    assert load_config(path) == {"future_seat": {"model": "some-model"}}


def test_load_config_valid_multi_seat_file(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\nmodel = "gpt-5.6-terra"\neffort = "high"\n\n'
        '[gemini]\nmodel = "gemini-3-pro"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["codex"] == {"model": "gpt-5.6-terra", "effort": "high"}
    assert config["gemini"] == {"model": "gemini-3-pro"}


# --- recorded_choice ---------------------------------------------------------


def test_recorded_choice_none_when_absent(tmp_path: Path) -> None:
    assert recorded_choice("codex", _cfg(tmp_path)) is None


def test_recorded_choice_returns_seat_table(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    path.write_text('[codex]\nmodel = "gpt-5.6-terra"\n', encoding="utf-8")
    assert recorded_choice("codex", path) == {"model": "gpt-5.6-terra"}


# --- resolve_model: resolution order ----------------------------------------


def test_resolve_model_per_run_wins_over_all(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "recorded-model"}, path)
    model, source = resolve_model(
        "codex",
        per_run="per-run-model",
        env_value="env-model",
        shipped="shipped-model",
        path=path,
    )
    assert (model, source) == ("per-run-model", "per-run")


def test_resolve_model_env_wins_over_recorded_and_shipped(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "recorded-model"}, path)
    model, source = resolve_model(
        "codex",
        per_run=None,
        env_value="env-model",
        shipped="shipped-model",
        path=path,
    )
    assert (model, source) == ("env-model", "env")


def test_resolve_model_recorded_wins_over_shipped(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "recorded-model"}, path)
    model, source = resolve_model(
        "codex",
        per_run=None,
        env_value=None,
        shipped="shipped-model",
        path=path,
    )
    assert (model, source) == ("recorded-model", "recorded")


def test_resolve_model_falls_back_to_shipped(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    model, source = resolve_model(
        "codex",
        per_run=None,
        env_value=None,
        shipped="shipped-model",
        path=path,
    )
    assert (model, source) == ("shipped-model", "shipped")


def test_resolve_model_absent_file_returns_shipped_with_no_warning(
    tmp_path: Path, caplog
) -> None:
    path = _cfg(tmp_path)
    assert not path.exists()
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        model, source = resolve_model(
            "codex",
            per_run=None,
            env_value="env-model",
            shipped="shipped-model",
            path=path,
        )
    assert (model, source) == ("env-model", "env")
    assert caplog.records == []


# --- resolve_model: env-masks-recorded warning ------------------------------


def test_resolve_model_env_masks_recorded_warns_exactly_once(
    tmp_path: Path, caplog
) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "recorded-model"}, path)
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        resolve_model(
            "codex",
            per_run=None,
            env_value="env-model",
            shipped="shipped-model",
            path=path,
        )
        resolve_model(
            "codex",
            per_run=None,
            env_value="env-model",
            shipped="shipped-model",
            path=path,
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "env-model" in warnings[0].message
    assert "recorded-model" in warnings[0].message
    assert "CODE_QUORUM_CODEX_MODEL" in warnings[0].message


def test_resolve_model_no_warning_when_env_equals_recorded(
    tmp_path: Path, caplog
) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "same-model"}, path)
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        resolve_model(
            "codex",
            per_run=None,
            env_value="same-model",
            shipped="shipped-model",
            path=path,
        )
    assert caplog.records == []


def test_resolve_model_no_warning_when_no_recorded_choice(
    tmp_path: Path, caplog
) -> None:
    path = _cfg(tmp_path)
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        resolve_model(
            "codex",
            per_run=None,
            env_value="env-model",
            shipped="shipped-model",
            path=path,
        )
    assert caplog.records == []


def test_resolve_model_no_warning_when_per_run_wins(tmp_path: Path, caplog) -> None:
    # per_run already wins the resolution outright -- warning about env
    # masking the recorded choice would be actively misleading here, since
    # neither env nor the recorded choice is what actually got used.
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "recorded-model"}, path)
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        model, source = resolve_model(
            "codex",
            per_run="per-run-model",
            env_value="env-model",
            shipped="shipped-model",
            path=path,
        )
    assert (model, source) == ("per-run-model", "per-run")
    assert caplog.records == []


def test_resolve_model_warn_once_is_per_seat(tmp_path: Path, caplog) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "codex-recorded"}, path)
    record_choice("gemini", {"model": "gemini-recorded"}, path)
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        resolve_model(
            "codex",
            per_run=None,
            env_value="codex-env",
            shipped="shipped-model",
            path=path,
        )
        resolve_model(
            "gemini",
            per_run=None,
            env_value="gemini-env",
            shipped="shipped-model",
            path=path,
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("CODE_QUORUM_CODEX_MODEL" in r.message for r in warnings)
    assert any("CODE_QUORUM_GEMINI_MODEL" in r.message for r in warnings)


# --- record_choice -----------------------------------------------------------


def test_record_choice_round_trip(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "gpt-5.6-terra", "effort": "high"}, path)
    config = load_config(path)
    assert config["codex"]["model"] == "gpt-5.6-terra"
    assert config["codex"]["effort"] == "high"


def test_record_choice_stamps_todays_date(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "gpt-5.6-terra"}, path)
    config = load_config(path)
    assert config["codex"]["chosen"] == date.today().isoformat()


def test_record_choice_atomic_no_leftover_tmp_file(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "gpt-5.6-terra"}, path)
    entries = sorted(p.name for p in tmp_path.iterdir())
    assert entries == ["models.toml"]
    # Target is complete: a fresh load_config parses it without error and
    # everything written round-trips.
    assert load_config(path)["codex"]["model"] == "gpt-5.6-terra"


def test_record_choice_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "models.toml"
    assert not path.parent.exists()
    record_choice("codex", {"model": "gpt-5.6-terra"}, path)
    assert path.exists()
    assert load_config(path)["codex"]["model"] == "gpt-5.6-terra"


def test_record_choice_preserves_other_seats(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "codex-model"}, path)
    record_choice("gemini", {"model": "gemini-model"}, path)
    config = load_config(path)
    assert config["codex"]["model"] == "codex-model"
    assert config["gemini"]["model"] == "gemini-model"


def test_record_choice_preserves_unknown_table_seat_on_rewrite(
    tmp_path: Path,
) -> None:
    # Regression: an unknown all-string table (with or without `model`)
    # survives a record_choice rewrite byte-intact -- since load_config now
    # accepts it, it stays in `config` and gets re-serialized like any
    # other table.
    path = _cfg(tmp_path)
    path.write_text(
        '[future_seat]\nmodel = "some-model"\nextra = "value"\n', encoding="utf-8"
    )
    record_choice("codex", {"model": "codex-model"}, path)
    config = load_config(path)
    assert config["future_seat"] == {"model": "some-model", "extra": "value"}
    assert config["codex"]["model"] == "codex-model"


def test_record_choice_repairs_malformed_file(tmp_path: Path) -> None:
    # A malformed existing file no longer blocks record_choice -- it's
    # treated as empty, so a successful record repairs the corruption
    # instead of raising (record_choice unconditionally tolerates a
    # malformed file now; nothing relies on the old raising default).
    path = _cfg(tmp_path)
    path.write_text('future_seat = "x"\n', encoding="utf-8")
    record_choice("codex", {"model": "codex-model"}, path)
    config = load_config(path)
    assert config == {
        "codex": {"model": "codex-model", "chosen": date.today().isoformat()}
    }


def test_record_choice_value_with_special_chars_round_trips(tmp_path: Path) -> None:
    # Exercises the json.dumps-based escaping (_serialize) for every
    # character class TOML basic strings forbid unescaped: C0 controls,
    # DEL, and the mandatory backslash/quote escapes.
    path = _cfg(tmp_path)
    value = 'line1\nline2\ttabbed\x7f\x00 back\\slash "quote"'
    record_choice("codex", {"model": "gpt-5.6-terra", "cli_version": value}, path)
    config = load_config(path)
    assert config["codex"]["cli_version"] == value


def test_load_config_skips_effort_validation_on_disabled_seat(
    tmp_path: Path,
) -> None:
    # A disabled table skips the `model` requirement. It must also ignore a
    # stale invalid `effort`; otherwise that value aborts the whole config load.
    path = _cfg(tmp_path)
    path.write_text(
        '[codex]\ndisabled = "true"\neffort = "warp9"\n'
        '[claude]\nmodel = "claude-fable-5"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["claude"]["model"] == "claude-fable-5"
    assert config["codex"]["disabled"] == "true"


def test_record_choice_non_bmp_value_round_trips(tmp_path: Path) -> None:
    # json.dumps with its ensure_ascii=True default encodes non-BMP characters
    # as \uXXXX surrogate pairs, which TOML rejects. Non-ASCII must be written
    # literally as UTF-8 so the next load_config call can parse it.
    path = _cfg(tmp_path)
    value = "gpt-🚀 café"
    record_choice("codex", {"model": value}, path)
    assert load_config(path)["codex"]["model"] == value


def test_record_choice_rejects_non_bare_key(tmp_path: Path) -> None:
    # _serialize writes keys bare, so a key outside TOML's bare-key alphabet
    # must fail at write time instead of producing an invalid file. The guard
    # belongs on the writer, where the corruption would originate.
    path = _cfg(tmp_path)
    with pytest.raises(ModelConfigError, match="bare TOML key"):
        record_choice("codex", {"cli version": "1.0"}, path)
    with pytest.raises(ModelConfigError, match="bare TOML key"):
        record_choice("bad seat", {"model": "x"}, path)
    assert not path.exists()


def test_record_choice_replaces_prior_table_for_same_seat(tmp_path: Path) -> None:
    path = _cfg(tmp_path)
    record_choice("codex", {"model": "old-model", "effort": "high"}, path)
    record_choice("codex", {"model": "new-model"}, path)
    config = load_config(path)
    assert config["codex"]["model"] == "new-model"
    assert "effort" not in config["codex"]


# --- warn_version_drift -------------------------------------------------------
# The same version-drift warning (installed CLI != recorded cli_version) fires
# from codex.py / opencode.py / gemini_cli.py -- centralized
# here so the dedup + message logic is written and tested once.


def test_warn_version_drift_warns_once_per_seat(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        mc.warn_version_drift("codex", "0.200.0", "0.100.0")
        mc.warn_version_drift("codex", "0.200.0", "0.100.0")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "0.200.0" in warnings[0].message  # installed
    assert "0.100.0" in warnings[0].message  # recorded
    assert "quorum setup-models" in warnings[0].message


def test_warn_version_drift_dedup_is_per_seat(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="quorum.model_config"):
        mc.warn_version_drift("codex", "0.200.0", "0.100.0")
        mc.warn_version_drift("gemini", "1.2.0", "1.1.0")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("codex" in r.message for r in warnings)
    assert any("gemini" in r.message for r in warnings)
