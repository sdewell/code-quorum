import pytest

from quorum import doctor
from quorum import model_config as mc


def _patch_binaries(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}",
    )


def test_claude_host_checks_direct_seatbelt_without_helper(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor,
        "check_sandbox_available",
        lambda: calls.append("sandbox") or None,
    )
    monkeypatch.setattr(
        doctor,
        "check_subscription_auth",
        lambda: calls.append("auth") or None,
    )
    monkeypatch.setattr(
        doctor,
        "helper_compatibility_error",
        lambda: calls.append("helper") or None,
    )
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)

    results = doctor.run_doctor("claude")

    assert all(result.ok for result in results)
    assert calls == ["sandbox"]
    assert {result.name for result in results if result.name.startswith("binary:")} == {
        "binary:agy",
        "binary:codex",
        "binary:opencode",
        "binary:uv",
    }


def test_codex_host_checks_subscription_and_helper_without_nested_seatbelt(
    monkeypatch,
) -> None:
    _patch_binaries(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor,
        "check_sandbox_available",
        lambda: calls.append("sandbox") or None,
    )
    monkeypatch.setattr(
        doctor,
        "check_subscription_auth",
        lambda: calls.append("auth") or None,
    )
    monkeypatch.setattr(
        doctor,
        "helper_compatibility_error",
        lambda: calls.append("helper") or None,
    )
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)

    results = doctor.run_doctor("codex")

    assert all(result.ok for result in results)
    assert calls == ["auth", "helper"]
    assert {result.name for result in results if result.name.startswith("binary:")} == {
        "binary:agy",
        "binary:claude",
        "binary:opencode",
        "binary:uv",
    }


def test_both_reports_each_failed_boundary(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(
        doctor, "check_sandbox_available", lambda: "seatbelt unavailable"
    )
    monkeypatch.setattr(
        doctor, "check_subscription_auth", lambda: "not subscription auth"
    )
    monkeypatch.setattr(
        doctor,
        "helper_compatibility_error",
        lambda: "shared seat helper protocol version is incompatible",
    )
    monkeypatch.setattr(
        doctor, "verify_read_only_config", lambda: "agy settings unsafe"
    )

    results = doctor.run_doctor("both")

    errors = {result.name: result.detail for result in results if not result.ok}
    assert "seatbelt" in errors
    assert "claude-subscription" in errors
    assert "seat-helper" in errors
    assert "agy-config" in errors


def test_codex_host_reports_incompatible_helper_protocol(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    monkeypatch.setattr(doctor, "check_subscription_auth", lambda: None)
    monkeypatch.setattr(
        doctor,
        "helper_compatibility_error",
        lambda: "shared seat helper protocol version 0 is incompatible",
    )

    results = doctor.run_doctor("codex")

    helper = next(result for result in results if result.name == "seat-helper")
    assert not helper.ok
    assert "version 0 is incompatible" in helper.detail


def test_unknown_doctor_host_rejected() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        doctor.run_doctor("gemini")


def test_disabled_seat_skips_binary_check_and_reports_ok(monkeypatch) -> None:
    """A disabled seat must be reported as ok without probing for its binary.
    A user who disables Gemini must not get a doctor failure for missing agy."""
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    mc.record_choice("gemini", {"disabled": "true"})

    results = doctor.run_doctor("claude")

    gemini = next(r for r in results if r.name == "seat:gemini")
    assert gemini.ok
    assert gemini.detail == "disabled by user"
    assert not any(r.name == "binary:agy" for r in results)
    # codex was not disabled -- still checked, and still fails (which=None).
    codex_binary = next(r for r in results if r.name == "binary:codex")
    assert not codex_binary.ok


def test_disabled_gemini_skips_agy_config_check(monkeypatch) -> None:
    """agy-config only validates the gemini seat's sandbox settings -- a
    disabled gemini seat must skip it (reported ok) WITHOUT even calling
    verify_read_only_config, so a leftover bad agy config can't fail a seat
    the user opted out of."""
    _patch_binaries(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor,
        "verify_read_only_config",
        lambda: calls.append("agy-config") or "agy settings unsafe",
    )
    mc.record_choice("gemini", {"disabled": "true"})

    results = doctor.run_doctor("claude")

    agy_config = next(r for r in results if r.name == "agy-config")
    assert agy_config.ok
    assert "gemini seat disabled by user" in agy_config.detail
    assert calls == []  # the check never ran


def test_disabled_claude_skips_subscription_check_on_codex_host(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor,
        "check_subscription_auth",
        lambda: calls.append("auth") or "not logged in",
    )
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    monkeypatch.setattr(doctor, "helper_compatibility_error", lambda: None)
    mc.record_choice("claude", {"disabled": "true"})

    results = doctor.run_doctor("codex")

    sub = next(r for r in results if r.name == "claude-subscription")
    assert sub.ok
    assert "claude seat disabled by user" in sub.detail
    assert calls == []  # the check never ran


def test_seat_helper_still_runs_when_only_one_covered_seat_disabled(
    monkeypatch,
) -> None:
    """seat-helper serves both claude and gemini -- disabling only ONE must
    NOT skip it, since the other seat still depends on it. This is the case
    most likely to regress silently if the skip condition is ever loosened
    from "all covered seats disabled" to "any covered seat disabled"."""
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    monkeypatch.setattr(doctor, "check_subscription_auth", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor, "helper_compatibility_error", lambda: calls.append("helper") or None
    )
    mc.record_choice("claude", {"disabled": "true"})  # only claude disabled

    results = doctor.run_doctor("codex")

    helper = next(r for r in results if r.name == "seat-helper")
    assert helper.ok
    assert calls == ["helper"]  # actually ran, not skipped


def test_seat_helper_skipped_when_both_covered_seats_disabled(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    monkeypatch.setattr(doctor, "check_subscription_auth", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor, "helper_compatibility_error", lambda: calls.append("helper") or None
    )
    mc.record_choice("claude", {"disabled": "true"})
    mc.record_choice("gemini", {"disabled": "true"})

    results = doctor.run_doctor("codex")

    helper = next(r for r in results if r.name == "seat-helper")
    assert helper.ok
    assert "claude and gemini seats disabled by user" in helper.detail
    assert calls == []  # the check never ran


def test_malformed_models_toml_does_not_crash_run_doctor(monkeypatch) -> None:
    """An uncaught ModelConfigError from a corrupt file must not abort the
    doctor report. A malformed file must produce one failing diagnostic while
    every other check runs and treats each seat as enabled."""
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    monkeypatch.setattr(doctor, "check_subscription_auth", lambda: None)
    monkeypatch.setattr(doctor, "check_sandbox_available", lambda: None)
    monkeypatch.setattr(doctor, "helper_compatibility_error", lambda: None)
    mc.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    mc.CONFIG_PATH.write_text("this is not valid toml [[[", encoding="utf-8")

    results = doctor.run_doctor("both")

    config = next(r for r in results if r.name == "models.toml")
    assert not config.ok
    assert "unreadable" in config.detail
    # The rest of the report still runs in full, with no seat treated as
    # disabled (a corrupt file must never silently mean "opted out").
    assert not any(r.name.startswith("seat:") for r in results)
    names = {r.name for r in results}
    assert {
        "binary:agy",
        "binary:codex",
        "binary:claude",
        "binary:opencode",
        "binary:uv",
        "agy-config",
        "seatbelt",
        "claude-subscription",
        "seat-helper",
    } <= names
    assert all(r.ok for r in results if r.name != "models.toml")


def test_disabled_gemini_skips_seatbelt_check(monkeypatch) -> None:
    """Seatbelt is Gemini-owned, like agy-config. A disabled Gemini seat must
    skip it without calling check_sandbox_available."""
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(doctor, "verify_read_only_config", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        doctor,
        "check_sandbox_available",
        lambda: calls.append("sandbox") or "seatbelt unavailable",
    )
    mc.record_choice("gemini", {"disabled": "true"})

    results = doctor.run_doctor("claude")

    seatbelt = next(r for r in results if r.name == "seatbelt")
    assert seatbelt.ok
    assert "gemini seat disabled by user" in seatbelt.detail
    assert calls == []  # the check never ran
