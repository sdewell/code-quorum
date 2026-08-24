import pytest

from quorum.agents import CodexAgent
from quorum.agents.seat_helper import SeatHelperAgent
from quorum.hosts import DEFAULT_HOST, resolve_host
from quorum.orchestration import select_agents
from quorum_mcp import server


def test_default_host_preserves_existing_claude_behavior(monkeypatch) -> None:
    monkeypatch.delenv("CODE_QUORUM_HOST", raising=False)

    profile = resolve_host()

    assert DEFAULT_HOST == "claude"
    assert profile.name == "claude"
    assert profile.default_agents == ("codex", "gemini", "opencode")


def test_codex_host_defaults_to_claude_gemini_opencode() -> None:
    chosen = select_agents(None, host="codex").agents

    assert tuple(agent.name for agent in chosen) == ("claude", "gemini", "opencode")
    assert isinstance(chosen[0], SeatHelperAgent)
    assert not any(isinstance(agent, CodexAgent) for agent in chosen)


def test_host_default_roles_are_symmetric() -> None:
    claude_host = {
        agent.name: agent.role for agent in select_agents(None, host="claude").agents
    }
    codex_host = {
        agent.name: agent.role for agent in select_agents(None, host="codex").agents
    }

    assert claude_host == {
        "codex": "skeptic",
        "gemini": "architect",
        "opencode": "neutral",
    }
    assert codex_host == {
        "claude": "skeptic",
        "gemini": "architect",
        "opencode": "neutral",
    }


@pytest.mark.parametrize(("host", "seat"), [("claude", "claude"), ("codex", "codex")])
def test_host_rejects_its_own_subprocess_seat(host: str, seat: str) -> None:
    with pytest.raises(ValueError, match="orchestrator"):
        select_agents([seat], host=host)


def test_resolve_host_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="known hosts"):
        resolve_host("gemini")


def test_mcp_main_pins_runtime_host(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(server.mcp, "run", lambda: called.append(True))

    server.main(["--host", "codex"])

    assert resolve_host().name == "codex"
    assert called == [True]
