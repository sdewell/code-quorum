#!/bin/sh

set -eu

# Bundled MCP and hook commands use paths relative to the plugin root.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
plugin_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
cd "$plugin_root"

user_path=""
add_user_path() {
    directory="$1"
    if [ ! -d "$directory" ]; then
        return
    fi
    case ":${PATH:-}:" in
        *:"$directory":*) ;;
        *) user_path="${user_path:+$user_path:}$directory" ;;
    esac
}

if [ -n "${HOME:-}" ]; then
    add_user_path "$HOME/.local/bin"
    add_user_path "$HOME/.opencode/bin"
fi
add_user_path /opt/homebrew/bin
add_user_path /usr/local/bin

if [ -n "$user_path" ]; then
    PATH="$user_path${PATH:+:$PATH}"
    export PATH
fi

if [ "$#" -eq 0 ]; then
    echo "code-quorum: user-path runner requires a command" >&2
    exit 64
fi
if ! command -v "$1" >/dev/null 2>&1; then
    echo "code-quorum: $1 not found in standard macOS user/Homebrew directories or PATH" >&2
    exit 127
fi

exec "$@"
