#!/bin/sh

set -eu

if [ "$#" -ne 1 ]; then
    echo "code-quorum: Codex MCP launcher requires the plugin version" >&2
    exit 64
fi
plugin_version="$1"

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
plugin_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
helper_plist="${HOME:?code-quorum: HOME is required}/Library/LaunchAgents/com.code-quorum.seat-helper.plist"

if [ -f "$helper_plist" ]; then
    if ! runtime_root="$(/usr/libexec/PlistBuddy -c 'Print :WorkingDirectory' "$helper_plist" 2>/dev/null)" || [ -z "$runtime_root" ]; then
        echo "code-quorum: helper plist has no valid WorkingDirectory; reinstall the seat helper" >&2
        exit 78
    fi
elif [ -x "$plugin_root/.venv/bin/quorum-mcp" ]; then
    runtime_root="$plugin_root"
else
    echo "code-quorum: no prepared Codex runtime; run 'uv run quorum install-seat-helper-launchagent' from the code-quorum checkout" >&2
    exit 78
fi

case "$runtime_root" in
    /*) ;;
    *)
        echo "code-quorum: helper WorkingDirectory is not absolute: $runtime_root" >&2
        exit 78
        ;;
esac

quorum_mcp="$runtime_root/.venv/bin/quorum-mcp"
runtime_python="$runtime_root/.venv/bin/python"
if [ ! -x "$quorum_mcp" ] || [ ! -x "$runtime_python" ]; then
    echo "code-quorum: stable runtime is not prepared at $runtime_root; run 'uv sync' there, then reinstall the seat helper" >&2
    exit 78
fi

if ! runtime_version="$(
    CDPATH= cd -- "$runtime_root" &&
        "$runtime_python" -c 'from quorum import __version__; print(__version__)'
)"; then
    echo "code-quorum: stable runtime version could not be read at $runtime_root; run 'uv sync' there, then reinstall the seat helper" >&2
    exit 78
fi
if [ "$runtime_version" != "$plugin_version" ]; then
    echo "code-quorum: stable runtime version $runtime_version does not match plugin $plugin_version; update the checkout and reinstall the seat helper" >&2
    exit 78
fi

exec "$plugin_root/scripts/run_in_plugin_root_with_user_path.sh" "$quorum_mcp" --host codex
