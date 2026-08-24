import os
import shutil
import subprocess
from pathlib import Path

from scripts import build_codex_marketplace as builder

_ROOT = Path(__file__).resolve().parent.parent


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_nag(script: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "agy",
        "#!/bin/sh\nprintf '%s\\n' '99.99.98'\n",
    )
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_drift_nag_uses_source_verifier_when_present(tmp_path: Path) -> None:
    proc = _run_nag(_ROOT / "scripts" / "agy-drift-nag.sh", tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert f'bash "{_ROOT}/scripts/verify-agy-seat.sh"' in proc.stdout


def test_drift_nag_points_codex_artifact_to_source_checkout(tmp_path: Path) -> None:
    plugin = builder.build_marketplace(tmp_path / "marketplace", source_root=_ROOT)

    proc = _run_nag(plugin / "scripts" / "agy-drift-nag.sh", tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "matching code-quorum source checkout" in proc.stdout
    assert "bash scripts/verify-agy-seat.sh" in proc.stdout
    assert "rebuild into a fresh --output directory and reinstall" in proc.stdout
    assert str(plugin / "scripts" / "verify-agy-seat.sh") not in proc.stdout


def test_drift_nag_treats_claude_cache_as_an_installed_plugin(tmp_path: Path) -> None:
    cache = tmp_path / "claude-cache"
    for path in (
        "scripts/agy-drift-nag.sh",
        "scripts/verify-agy-seat.sh",
        "quorum/agents/gemini_cli.py",
        ".claude-plugin/plugin.json",
    ):
        destination = cache / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / path, destination)
    assert not (cache / ".git").exists()

    proc = _run_nag(cache / "scripts" / "agy-drift-nag.sh", tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "matching code-quorum source checkout" in proc.stdout
    assert "bash scripts/verify-agy-seat.sh" in proc.stdout
    assert "normal commit/release flow" in proc.stdout
    assert "reinstall the Claude plugin" in proc.stdout
    assert "Codex marketplace artifact" not in proc.stdout
    assert f'bash "{cache}/scripts/verify-agy-seat.sh"' not in proc.stdout
