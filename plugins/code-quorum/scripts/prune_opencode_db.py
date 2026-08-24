"""Cap the size of opencode's sandbox database by pruning old COUNCIL sessions.

The opencode seat runs in a sandbox HOME (`~/.cache/code-quorum/opencode-sandbox`)
and opencode keeps every session's messages, parts, and event log in one SQLite
database there. Nothing bounds it. Measured 2026-07-29 after a couple of months
of use: **516 MB**, of which `part` was 316 MB and `event` 173 MB. The seat
already prunes its npm cache; this closes the other unbounded write.

Deletes ONLY sessions whose agent is `council` (what this seat runs as), oldest
first, so a database shared with hand-run opencode work never loses the user's
own sessions. Under the cap it is a single stat and an immediate exit.

Two traps this had to handle, both verified against the real schema:

1. **`PRAGMA foreign_keys` is OFF by default in SQLite.** `session` owns
   `message` -> `part` by `ON DELETE CASCADE`, but without the pragma those
   cascades do not fire and deleting sessions would orphan the 316 MB of parts
   that are the actual problem. It is turned on explicitly.
2. **`event` does NOT cascade from `session`.** Its FK points at
   `event_sequence`, not `session`, even though `event.aggregate_id` holds a
   session id -- so a third of the database is unreachable from a session
   delete. Events are removed explicitly by aggregate id.

Concurrency (shared-dir-io): a council run may be using this database right now,
and VACUUM needs an exclusive lock. So this refuses to run while any opencode
process is alive, sets a bounded busy timeout, and treats a locked database as
"try again next time" rather than something to force.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

# Mirrors _SANDBOX_HOME in quorum/agents/opencode.py. Duplicated deliberately:
# this runs from a SessionStart hook via `uv run --no-project`, so it cannot
# import the package. Keep the two in step -- if the seat's sandbox home moves,
# this silently prunes nothing (it reports a missing database, which is the
# visible failure mode, not a silent one).
DB_PATH = (
    Path.home()
    / ".cache"
    / "code-quorum"
    / "opencode-sandbox"
    / ".local"
    / "share"
    / "opencode"
    / "opencode.db"
)

# The agent name the seat runs opencode as (COUNCIL_AGENT_MD -> agents/council.md,
# invoked with `--agent council`). The scoping key: everything else in this
# database belongs to the user.
COUNCIL_AGENT = "council"

# Newest council sessions never eligible, whatever the arithmetic says. The seat
# recovers a dropped answer by reading it back out of this database
# (_recover_text_from_db in the opencode adapter), so the freshest sessions are
# not dead weight -- they are the fallback for the run that just happened. Cheap
# insurance: a few rounds' worth across seats, costing at most a few MB.
KEEP_RECENT_SESSIONS = 10

DEFAULT_CAP_MB = 500.0
# Prune down to below the cap, not to it, so this does not re-fire (and re-VACUUM
# a half-gigabyte file) on every single session start once the cap is reached.
DEFAULT_TARGET_MB = 400.0
BUSY_TIMEOUT_MS = 5_000

MB = 1024 * 1024


def footprint_bytes(db: Path) -> int:
    """On-disk bytes for the database INCLUDING its WAL sidecars.

    opencode runs this database in WAL mode, so committed-but-uncheckpointed
    pages live in `-wal`, not in the main file. Measuring the main file alone
    would under-report the real footprint and could skip a prune that is
    genuinely due. (Observed here the WAL is under 1 MB, so this is about not
    depending on that staying true rather than about bytes we are missing now.)
    """
    total = 0
    for path in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        try:
            total += path.stat().st_size
        except OSError:
            pass  # sidecars exist only while the database is open
    return total


def opencode_is_running() -> bool:
    """True when any opencode process is alive. VACUUM takes an exclusive lock,
    so pruning under a live council run could stall or fail that run's seat.

    Absolute path on purpose: this is a gate on a destructive action, so it must
    not be satisfiable by a `pgrep` earlier on PATH. macOS-only assumption, which
    matches where this plugin runs; anywhere else the probe raises, and the
    handler below fails safe by reporting "busy" so nothing is ever pruned blind.
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/pgrep", "-x", "opencode"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # cannot tell -> assume busy, never prune blind
    return proc.returncode == 0


def session_costs(conn: sqlite3.Connection) -> list[tuple[str, int, int]]:
    """(session_id, time_created, approx_bytes) for council sessions, oldest
    first. Bytes sum the three tables that hold the volume, so the caller can
    stop deleting as soon as enough has been reclaimed."""
    rows = conn.execute(
        """
        SELECT s.id,
               s.time_created,
               COALESCE(p.b, 0) + COALESCE(m.b, 0) + COALESCE(e.b, 0) AS bytes
          FROM session s
          LEFT JOIN (SELECT session_id, SUM(LENGTH(data)) b FROM part
                      GROUP BY session_id) p ON p.session_id = s.id
          LEFT JOIN (SELECT session_id, SUM(LENGTH(data)) b FROM message
                      GROUP BY session_id) m ON m.session_id = s.id
          LEFT JOIN (SELECT aggregate_id, SUM(LENGTH(data)) b FROM event
                      GROUP BY aggregate_id) e ON e.aggregate_id = s.id
         WHERE s.agent = ?
         ORDER BY s.time_created ASC
        """,
        (COUNCIL_AGENT,),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def prune(
    db: Path,
    cap_mb: float,
    target_mb: float,
    dry_run: bool = False,
    quiet: bool = False,
) -> int:
    def say(msg: str) -> None:
        if not quiet:
            print(f"prune-opencode-db: {msg}")

    if not db.exists():
        say(f"no database at {db} -- nothing to prune.")
        return 0

    size = footprint_bytes(db)
    if size <= cap_mb * MB:
        say(f"{size / MB:.0f} MB is within the {cap_mb:.0f} MB cap.")
        return 0

    if opencode_is_running():
        say("opencode is running -- skipping (VACUUM needs an exclusive lock).")
        return 0

    must_free = size - int(target_mb * MB)
    try:
        # Inside the try on purpose: this runs from a SessionStart hook, and an
        # unopenable database (permissions, corruption, a locked file) must read
        # as "not today" rather than a traceback on every session start.
        conn = sqlite3.connect(db, timeout=BUSY_TIMEOUT_MS / 1000)
    except sqlite3.Error as exc:
        say(f"cannot open {db} ({exc}) -- leaving it alone.")
        return 0
    try:
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        # Load-bearing: without this the message/part cascades below do not fire
        # and the bytes we are trying to reclaim would be orphaned, not freed.
        conn.execute("PRAGMA foreign_keys = ON")

        costs = session_costs(conn)
        if not costs:
            say(
                f"{size / MB:.0f} MB is over the cap but no {COUNCIL_AGENT!r} "
                "sessions exist -- the rest is not ours to delete."
            )
            return 0

        # costs is oldest-first, so dropping the tail protects the newest.
        eligible = costs[: max(0, len(costs) - KEEP_RECENT_SESSIONS)]
        if not eligible:
            say(
                f"{size / MB:.0f} MB is over the cap but all {len(costs)} council "
                f"sessions are within the {KEEP_RECENT_SESSIONS} most recent, "
                "which are never pruned -- leaving them."
            )
            return 0

        doomed: list[str] = []
        freed = 0
        for sid, _created, nbytes in eligible:
            if freed >= must_free:
                break
            doomed.append(sid)
            freed += nbytes

        say(
            f"{size / MB:.0f} MB over the {cap_mb:.0f} MB cap; dropping "
            f"{len(doomed)} of {len(costs)} council sessions "
            f"(~{freed / MB:.0f} MB) to reach ~{target_mb:.0f} MB."
        )
        if freed < must_free:
            say(
                "note: every council session was selected and that is still not "
                "enough -- the remainder is non-council data, left untouched."
            )
        if dry_run:
            return 0

        marks = ",".join("?" for _ in doomed)
        # event has no FK to session (its FK is to event_sequence), so its rows
        # must go explicitly or a third of the database survives the delete.
        conn.execute(f"DELETE FROM event WHERE aggregate_id IN ({marks})", doomed)
        conn.execute(
            f"DELETE FROM event_sequence WHERE aggregate_id IN ({marks})", doomed
        )
        # message + part follow by cascade (foreign_keys is ON above).
        conn.execute(f"DELETE FROM session WHERE id IN ({marks})", doomed)
        conn.commit()
        # Reclaim: deleted pages stay in the file as free space otherwise, so the
        # size cap would never actually be met.
        conn.execute("VACUUM")
        # Fold the WAL back into the main file and truncate it, so the size this
        # reports is the size a later run will measure -- otherwise the bytes we
        # just freed can sit in the sidecar and the next run sees no improvement.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        # Locked/busy is the expected racy case, not a failure worth shouting
        # about -- the next session start tries again. Wider than
        # OperationalError deliberately: a malformed database raises
        # DatabaseError, and a housekeeping hook that tracebacks on every
        # session start would be worse than one that skips a cleanup.
        say(f"could not prune ({exc}) -- leaving it alone, will retry next time.")
        return 0
    finally:
        conn.close()

    after = footprint_bytes(db)
    say(f"done: {size / MB:.0f} MB -> {after / MB:.0f} MB.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--cap-mb", type=float, default=DEFAULT_CAP_MB)
    ap.add_argument("--target-mb", type=float, default=DEFAULT_TARGET_MB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.target_mb > args.cap_mb:
        ap.error("--target-mb must be <= --cap-mb")
    return prune(
        args.db,
        cap_mb=args.cap_mb,
        target_mb=args.target_mb,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
