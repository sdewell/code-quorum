import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts import build_codex_marketplace as builder

_ROOT = Path(__file__).resolve().parent.parent


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _nag_env(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    _write_executable(
        fake_bin / "agy",
        "#!/bin/sh\nprintf '%s\\n' '99.99.98'\n",
    )
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_dir),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }


def _run_nag(script: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=_nag_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=10,
    )


def _copy_nag_runtime(destination: Path) -> None:
    for path in (
        "scripts/agy-drift-nag.sh",
        "scripts/verify-agy-seat.sh",
        "quorum/agents/gemini_cli.py",
        ".claude-plugin/plugin.json",
    ):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / path, target)


def _additional_context(proc: subprocess.CompletedProcess[str]) -> str:
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert set(payload) == {"hookSpecificOutput"}
    output = payload["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart"
    return output["additionalContext"]


def test_drift_nag_uses_source_verifier_when_present(tmp_path: Path) -> None:
    proc = _run_nag(_ROOT / "scripts" / "agy-drift-nag.sh", tmp_path)

    context = _additional_context(proc)
    assert f'bash "{_ROOT}/scripts/verify-agy-seat.sh"' in context


def test_drift_nag_json_escapes_source_path(tmp_path: Path) -> None:
    # POSIX paths may contain every JSON-sensitive byte except NUL and slash.
    source = tmp_path / 'source"\\\t\r\b\f\n\x01characters'
    _copy_nag_runtime(source)
    (source / ".git").mkdir()

    proc = _run_nag(source / "scripts" / "agy-drift-nag.sh", tmp_path)

    context = _additional_context(proc)
    assert f'bash "{source}/scripts/verify-agy-seat.sh"' in context


def test_drift_nag_emits_once_per_version_pair_per_day(tmp_path: Path) -> None:
    script = _ROOT / "scripts" / "agy-drift-nag.sh"

    first = _run_nag(script, tmp_path)
    assert "agy is v99.99.98" in _additional_context(first)

    second = _run_nag(script, tmp_path)
    assert second.returncode == 0, second.stderr
    assert second.stdout == ""


def test_drift_nag_does_not_require_uv(tmp_path: Path) -> None:
    env = _nag_env(tmp_path)
    env["PATH"] = f"{tmp_path / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin"

    proc = subprocess.run(
        ["bash", str(_ROOT / "scripts" / "agy-drift-nag.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "agy is v99.99.98" in _additional_context(proc)


def test_drift_nag_does_not_stamp_failed_output(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        proc = subprocess.run(
            ["bash", str(_ROOT / "scripts" / "agy-drift-nag.sh")],
            cwd=tmp_path,
            env=_nag_env(tmp_path),
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode != 0
    assert not (tmp_path / "cache" / "code-quorum" / "agy-drift-nag").exists()


def test_drift_nag_points_codex_artifact_to_source_checkout(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)

    proc = _run_nag(plugin / "scripts" / "agy-drift-nag.sh", tmp_path)

    context = _additional_context(proc)
    assert "matching code-quorum source checkout" in context
    assert "bash scripts/verify-agy-seat.sh" in context
    assert "rebuild into a fresh --output directory and reinstall" in context
    assert str(plugin / "scripts" / "verify-agy-seat.sh") not in context


def test_drift_nag_treats_claude_cache_as_an_installed_plugin(tmp_path: Path) -> None:
    cache = tmp_path / "claude-cache"
    _copy_nag_runtime(cache)
    assert not (cache / ".git").exists()

    proc = _run_nag(cache / "scripts" / "agy-drift-nag.sh", tmp_path)

    context = _additional_context(proc)
    assert "matching code-quorum source checkout" in context
    assert "bash scripts/verify-agy-seat.sh" in context
    assert "normal commit/release flow" in context
    assert "reinstall the Claude plugin" in context
    assert "Codex marketplace artifact" not in context
    assert f'bash "{cache}/scripts/verify-agy-seat.sh"' not in context
