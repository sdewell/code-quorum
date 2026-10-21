import asyncio
import inspect
import logging
import os
import pwd
import subprocess
import sys
import threading

import pytest

from quorum.agents import opencode
from quorum.agents.base import allowlisted_seat_subprocess_env
from tests import conftest


@pytest.fixture
def keychain_cli(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="quorum")
    security = tmp_path / "security"
    security.write_text("#!/bin/sh\nprintf 'sk-keychain-test\\n'\n")
    security.chmod(0o755)
    monkeypatch.setattr(opencode, "_KEYCHAIN_SECURITY", str(security))
    monkeypatch.setattr(opencode, "_platform", lambda: "darwin")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return security


def test_keychain_isolation_fails_if_its_dependency_is_missing(tmp_path, monkeypatch):
    monkeypatch.delattr(opencode, "_KEYCHAIN_SECURITY")
    with pytest.raises(AttributeError):
        inspect.unwrap(conftest._isolate_openrouter_keychain)(tmp_path, monkeypatch)


def test_keychain_fallback_uses_only_the_dedicated_current_user_item(
    keychain_cli, monkeypatch, caplog
):
    calls = []
    real_run = subprocess.run

    def capture(args, **kwargs):
        calls.append((args, kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture)

    assert opencode._resolve_openrouter_key(environ={}) == "sk-keychain-test"
    [(args, options)] = calls
    assert args == [
        str(keychain_cli),
        "find-generic-password",
        "-a",
        pwd.getpwuid(os.getuid()).pw_name,
        "-s",
        "code-quorum-openrouter",
        "-w",
    ]
    assert options["stdin"] is subprocess.DEVNULL
    assert options["capture_output"] is True
    assert 0 < options["timeout"] <= 5
    assert "sk-keychain-test" not in caplog.text
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "sk-keychain-test" not in repr(allowlisted_seat_subprocess_env())


def test_environment_key_wins_without_accessing_keychain(keychain_cli, monkeypatch):
    def unexpected_lookup(*args, **kwargs):
        pytest.fail("an environment key must bypass Keychain access")

    monkeypatch.setattr(subprocess, "run", unexpected_lookup)
    assert (
        opencode._resolve_openrouter_key(
            environ={"OPENROUTER_API_KEY": "  sk-environment  "}
        )
        == "sk-environment"
    )


@pytest.mark.parametrize(
    "outcome", ["missing", "denied", "empty", "timeout", "non-macos"]
)
def test_unavailable_keychain_leaves_the_seat_unconfigured(
    outcome, keychain_cli, monkeypatch, caplog
):
    if outcome == "missing":
        keychain_cli.unlink()
    elif outcome == "denied":
        keychain_cli.write_text(
            "#!/bin/sh\nprintf 'sk-must-not-leak'\n"
            "printf 'sk-must-not-leak' >&2\nexit 1\n"
        )
    elif outcome == "empty":
        keychain_cli.write_text("#!/bin/sh\nprintf '   \\n'\n")
    elif outcome == "non-macos":
        actual_platform = sys.platform
        monkeypatch.setattr(opencode, "_platform", lambda: "linux")
        assert sys.platform == actual_platform
    else:

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 5, output="sk-must-not-leak")

        monkeypatch.setattr(subprocess, "run", timeout)

    assert opencode._resolve_openrouter_key(environ={}) is None
    assert "sk-must-not-leak" not in caplog.text


@pytest.mark.asyncio
async def test_seat_runs_with_keychain_key_without_persisting_it(
    keychain_cli, tmp_path, monkeypatch, caplog
):
    binary = tmp_path / "opencode"
    binary.write_text(
        "#!/bin/sh\n"
        '[ "$OPENROUTER_API_KEY" = "sk-keychain-test" ] || exit 2\n'
        'printf \'%s\\n\' \'{"type":"text","part":{"text":"fixture answer"}}\'\n'
    )
    binary.chmod(0o755)
    sandbox = tmp_path / "sandbox"
    real_prepare = opencode._ensure_sandbox_home
    monkeypatch.setattr(
        opencode,
        "_ensure_sandbox_home",
        lambda **kwargs: real_prepare(sandbox=sandbox, **kwargs),
    )
    monkeypatch.setattr(opencode, "_DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    result = await opencode.OpenCodeAgent(
        binary=str(binary), model="fixture/model"
    ).run("answer briefly", str(tmp_path))

    assert result.returncode == 0, result.error
    assert result.output == "fixture answer"
    assert "sk-keychain-test" not in repr(result) + caplog.text
    assert "OPENROUTER_API_KEY" not in os.environ
    for path in sandbox.rglob("*"):
        if path.is_file():
            assert b"sk-keychain-test" not in path.read_bytes()


@pytest.mark.asyncio
async def test_keychain_lookup_does_not_block_other_seats(keychain_cli, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def resolve():
        entered.set()
        if not release.wait(timeout=2):
            raise AssertionError("lookup blocked the event loop")
        return None

    monkeypatch.setattr(opencode, "_resolve_openrouter_key", resolve)
    task = asyncio.create_task(opencode.OpenCodeAgent().run("x", "/tmp"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
    finally:
        release.set()
        await task
