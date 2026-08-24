import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_verifier(
    tmp_path: Path,
    *,
    uv_exit: int,
    uv_output: str = "",
    break_update_sed: bool = False,
    agy_exit: int = 0,
    agy_output: str = "agy version 1.1.9",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    pin_file = tmp_path / "gemini_cli.py"
    pin_file.write_text(
        'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n',
        encoding="utf-8",
    )
    pin_file.chmod(0o640)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "agy",
        f"#!/bin/sh\nprintf '%b\\n' {agy_output!r}\nexit {agy_exit}\n",
    )
    _write_executable(
        fake_bin / "uv",
        f"#!/bin/sh\nprintf '%b\\n' {uv_output!r}\nexit {uv_exit}\n",
    )
    _write_executable(
        fake_bin / "sed",
        """#!/bin/sh
if [ "${CODE_QUORUM_TEST_BREAK_UPDATE_SED:-}" = "1" ] && [ "$#" -eq 2 ]; then
  case "$1" in
    s/^SEAT_VERIFIED_AGY_VERSION*) printf '%s\\n' 'not a pin assignment'; exit 0 ;;
  esac
fi
exec /usr/bin/sed "$@"
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CODE_QUORUM_AGY_PIN_FILE": str(pin_file),
        "CODE_QUORUM_TEST_BREAK_UPDATE_SED": "1" if break_update_sed else "0",
    }

    return (
        subprocess.run(
            ["bash", "scripts/verify-agy-seat.sh"],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        ),
        pin_file,
    )


def test_verifier_updates_the_pin_after_live_canaries_pass(tmp_path: Path) -> None:
    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output="5 passed, 141 deselected in 1.00s",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'SEAT_VERIFIED_AGY_VERSION = "1.1.9"' in pin_file.read_text(encoding="utf-8")
    assert pin_file.stat().st_mode & 0o777 == 0o640


def test_verifier_leaves_the_pin_when_live_canaries_fail(tmp_path: Path) -> None:
    original = 'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n'
    proc, pin_file = _run_verifier(tmp_path, uv_exit=1)

    assert proc.returncode != 0
    assert pin_file.read_text(encoding="utf-8") == original
    assert "pin left at 1.1.8" in proc.stderr


def test_verifier_refuses_to_pin_when_live_canaries_are_skipped(
    tmp_path: Path,
) -> None:
    original = 'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n'

    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output="5 skipped, 141 deselected in 0.10s",
    )

    assert proc.returncode != 0
    assert pin_file.read_text(encoding="utf-8") == original
    assert "skipped" in proc.stderr.lower()


def test_verifier_ignores_skip_text_outside_pytest_summary(tmp_path: Path) -> None:
    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output=(
            "warning: documentation example contains 1 skipped phrase\n"
            "6 passed, 141 deselected in 1.00s"
        ),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'SEAT_VERIFIED_AGY_VERSION = "1.1.9"' in pin_file.read_text(encoding="utf-8")


def test_verifier_finds_pytest_summary_before_trailing_output(tmp_path: Path) -> None:
    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output=(
            "6 passed, 141 deselected in 1.00s\nharmless plugin shutdown message"
        ),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'SEAT_VERIFIED_AGY_VERSION = "1.1.9"' in pin_file.read_text(encoding="utf-8")


def test_verifier_refuses_to_pin_when_no_live_canary_passes(tmp_path: Path) -> None:
    original = 'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n'

    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output="141 deselected in 0.10s",
    )

    assert proc.returncode != 0
    assert pin_file.read_text(encoding="utf-8") == original
    assert "no live canary passed" in proc.stderr.lower()


def test_verifier_validates_the_rewrite_before_replacing_the_pin(
    tmp_path: Path,
) -> None:
    original = 'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n'

    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output="6 passed, 141 deselected in 1.00s",
        break_update_sed=True,
    )

    assert proc.returncode != 0
    assert pin_file.read_text(encoding="utf-8") == original
    assert "refusing to replace" in proc.stderr.lower()


def test_verifier_rejects_version_text_from_a_failed_agy_command(
    tmp_path: Path,
) -> None:
    original = 'SEAT_VERIFIED_AGY_VERSION = "1.1.8"\n'

    proc, pin_file = _run_verifier(
        tmp_path,
        uv_exit=0,
        uv_output="6 passed, 141 deselected in 1.00s",
        agy_exit=1,
        agy_output="wrapper failed under Python 3.13.1",
    )

    assert proc.returncode != 0
    assert pin_file.read_text(encoding="utf-8") == original
    assert "agy --version failed" in proc.stderr.lower()
