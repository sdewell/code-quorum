from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HOST = "claude"
HOST_ENV_VAR = "CODE_QUORUM_HOST"


@dataclass(frozen=True)
class HostProfile:
    name: str
    default_agents: tuple[str, ...]
    default_roles: dict[str, str]


HOST_PROFILES = {
    "claude": HostProfile(
        name="claude",
        default_agents=("codex", "gemini", "opencode"),
        default_roles={
            "codex": "skeptic",
            "gemini": "architect",
            "opencode": "neutral",
        },
    ),
    "codex": HostProfile(
        name="codex",
        default_agents=("claude", "gemini", "opencode"),
        default_roles={
            "claude": "skeptic",
            "gemini": "architect",
            "opencode": "neutral",
        },
    ),
}

_runtime_host: str | None = None


def configure_runtime_host(name: str) -> HostProfile:
    global _runtime_host
    profile = resolve_host(name)
    _runtime_host = profile.name
    return profile


def resolve_host(name: str | None = None) -> HostProfile:
    raw = name or _runtime_host or os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    normalized = raw.strip().lower()
    try:
        return HOST_PROFILES[normalized]
    except KeyError as exc:
        known = ", ".join(HOST_PROFILES)
        raise ValueError(f"unknown host {normalized!r}; known hosts: {known}") from exc
