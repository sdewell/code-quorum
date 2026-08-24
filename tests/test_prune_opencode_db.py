"""Tests for scripts/prune_opencode_db.py — the opencode sandbox DB size cap.

This deletes rows from a database the user's own opencode sessions may also live
in, so the tests are about what it must NEVER do as much as what it does: never
touch a non-council session, never leave the orphaned rows that hold the bulk of
the bytes, never act while opencode is running.

The schema here mirrors the real one where it matters, including the trap the
pruner exists to handle: `event` carries a session id in `aggregate_id` but its
foreign key points at `event_sequence`, NOT at `session`, so events do not
cascade from a session delete.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPT = _REPO_ROOT / "scripts" / "prune_opencode_db.py"
_spec = importlib.util.spec_from_file_location("prune_opencode_db", _SCRIPT)
assert _spec is not None and _spec.loader is not None
prune_opencode_db = importlib.util.module_from_spec(_spec)
sys.modules["prune_opencode_db"] = prune_opencode_db
_spec.loader.exec_module(prune_opencode_db)

pdb = prune_opencode_db

_SCHEMA = """
CREATE TABLE session (
  id TEXT PRIMARY KEY, agent TEXT, time_created INTEGER NOT NULL
);
CREATE TABLE message (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  data TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE
);
CREATE TABLE part (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  data TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE
);
CREATE TABLE event_sequence (aggregate_id TEXT PRIMARY KEY, seq INTEGER NOT NULL);
CREATE TABLE event (
  id TEXT PRIMARY KEY,
  aggregate_id TEXT NOT NULL,
  data TEXT NOT NULL,
  FOREIGN KEY (aggregate_id) REFERENCES event_sequence(aggregate_id) ON DELETE CASCADE
);
"""


def _build_db(path: Path, sessions: list[tuple[str, str, int, int]]) -> None:
    """sessions: (id, agent, time_created, payload_kb)."""
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    for sid, agent, created, kb in sessions:
        conn.execute(
            "INSERT INTO session (id, agent, time_created) VALUES (?,?,?)",
            (sid, agent, created),
        )
        blob = "x" * (kb * 1024)
        conn.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?,?,?)",
            (f"msg_{sid}", sid, blob),
        )
        conn.execute(
            "INSERT INTO part (id, session_id, message_id, data) VALUES (?,?,?,?)",
            (f"prt_{sid}", sid, f"msg_{sid}", blob),
        )
        conn.execute(
            "INSERT INTO event_sequence (aggregate_id, seq) VALUES (?,1)", (sid,)
        )
        conn.execute(
            "INSERT INTO event (id, aggregate_id, data) VALUES (?,?,?)",
            (f"evt_{sid}", sid, blob),
        )
    conn.commit()
    conn.close()


def _ids(path: Path, table: str, col: str = "id") -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute(f"SELECT {col} FROM {table}")}
    finally:
        conn.close()


def _not_running(monkeypatch) -> None:
    monkeypatch.setattr(pdb, "opencode_is_running", lambda: False)


def test_under_the_cap_is_a_no_op(tmp_path, monkeypatch) -> None:
    db = tmp_path / "oc.db"
    _build_db(db, [("ses_a", "council", 1, 64)])
    _not_running(monkeypatch)
    before = db.stat().st_size
    assert pdb.prune(db, cap_mb=100.0, target_mb=80.0, quiet=True) == 0
    assert db.stat().st_size == before
    assert _ids(db, "session") == {"ses_a"}


def test_a_missing_database_is_not_an_error(tmp_path, monkeypatch) -> None:
    # A fresh install has no sandbox DB yet; the hook must not fail the session.
    _not_running(monkeypatch)
    assert pdb.prune(tmp_path / "nope.db", cap_mb=1.0, target_mb=0.5, quiet=True) == 0


def test_it_refuses_to_prune_while_opencode_is_running(tmp_path, monkeypatch) -> None:
    # VACUUM takes an exclusive lock; doing that under a live council run would
    # stall or fail that run's seat.
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i}", "council", i, 256) for i in range(4)])
    monkeypatch.setattr(pdb, "opencode_is_running", lambda: True)
    assert pdb.prune(db, cap_mb=0.1, target_mb=0.05, quiet=True) == 0
    assert len(_ids(db, "session")) == 4, "nothing may be deleted while it runs"


def test_it_deletes_oldest_council_sessions_and_reclaims_space(
    tmp_path, monkeypatch
) -> None:
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i}", "council", i, 256) for i in range(20)])
    _not_running(monkeypatch)
    before = db.stat().st_size
    pdb.prune(db, cap_mb=1.0, target_mb=0.5, quiet=True)
    survivors = _ids(db, "session")
    assert survivors, "it must not empty the database"
    # Oldest go first, so the newest session always survives.
    assert "ses_19" in survivors
    assert "ses_0" not in survivors
    assert db.stat().st_size < before, "VACUUM must actually reclaim the pages"


def test_it_never_touches_a_non_council_session(tmp_path, monkeypatch) -> None:
    # The sandbox DB may be shared with hand-run opencode work. Those sessions
    # are the user's; only what this seat created is ours to delete.
    db = tmp_path / "oc.db"
    _build_db(
        db,
        [("ses_mine", "build", 1, 512)]
        + [(f"ses_c{i}", "council", i + 2, 256) for i in range(4)],
    )
    _not_running(monkeypatch)
    pdb.prune(db, cap_mb=0.5, target_mb=0.25, quiet=True)
    assert "ses_mine" in _ids(db, "session")
    assert _ids(db, "message") >= {"msg_ses_mine"}


def test_it_leaves_no_orphans_behind(tmp_path, monkeypatch) -> None:
    # The whole point is reclaiming bytes. message/part come away by cascade only
    # if PRAGMA foreign_keys is ON (it is OFF by default), and `event` does not
    # cascade from session at all -- so both are easy to silently orphan, which
    # would free nothing while reporting success.
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i}", "council", i, 256) for i in range(20)])
    _not_running(monkeypatch)
    pdb.prune(db, cap_mb=1.0, target_mb=0.5, quiet=True)
    live = _ids(db, "session")
    assert {p.removeprefix("prt_") for p in _ids(db, "part")} <= live
    assert {m.removeprefix("msg_") for m in _ids(db, "message")} <= live
    assert {e.removeprefix("evt_") for e in _ids(db, "event")} <= live
    assert _ids(db, "event_sequence", "aggregate_id") <= live


def test_it_stops_at_all_council_sessions_when_that_is_not_enough(
    tmp_path, monkeypatch
) -> None:
    # If the overflow is non-council data, pruning everything of ours still will
    # not reach the target -- and that is where it must stop, not keep going.
    db = tmp_path / "oc.db"
    _build_db(db, [("ses_big", "build", 1, 1024), ("ses_c", "council", 2, 32)])
    _not_running(monkeypatch)
    pdb.prune(db, cap_mb=0.2, target_mb=0.1, quiet=True)
    assert "ses_big" in _ids(db, "session")


def test_dry_run_changes_nothing(tmp_path, monkeypatch) -> None:
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i}", "council", i, 256) for i in range(20)])
    _not_running(monkeypatch)
    before = db.stat().st_size
    pdb.prune(db, cap_mb=1.0, target_mb=0.5, dry_run=True, quiet=True)
    assert len(_ids(db, "session")) == 20
    assert db.stat().st_size == before


def test_the_newest_sessions_are_never_pruned(tmp_path, monkeypatch) -> None:
    # The seat recovers a dropped answer by reading it back out of this database,
    # so the freshest sessions are the fallback for the run that just happened --
    # not dead weight. Cap set absurdly low so the arithmetic alone would take
    # every one of them.
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i:02d}", "council", i, 64) for i in range(14)])
    _not_running(monkeypatch)
    pdb.prune(db, cap_mb=0.05, target_mb=0.01, quiet=True)
    survivors = _ids(db, "session")
    assert len(survivors) >= pdb.KEEP_RECENT_SESSIONS
    assert "ses_13" in survivors and "ses_04" in survivors
    assert "ses_00" not in survivors


def test_db_path_still_matches_the_seat_it_prunes() -> None:
    """The script cannot import the package (it runs from a hook via
    `uv run --no-project`), so DB_PATH is a hand-copy of the seat's sandbox path.
    A silent divergence would mean the hook prunes nothing forever while printing
    a reassuring "no database" line. The test CAN import both, so it does."""
    from quorum.agents import opencode as seat

    expected = seat._SANDBOX_HOME / ".local" / "share" / "opencode" / "opencode.db"
    assert pdb.DB_PATH == expected, (
        "scripts/prune_opencode_db.py's DB_PATH has drifted from the opencode "
        "seat's sandbox home -- the SessionStart hook is pruning nothing."
    )


def test_footprint_counts_the_wal_sidecars(tmp_path) -> None:
    # WAL mode keeps committed-but-uncheckpointed pages beside the main file, so
    # measuring only the main file under-reports the real footprint. Plain files
    # here on purpose -- this is arithmetic over stat(), not sqlite behaviour.
    db = tmp_path / "oc.db"
    db.write_bytes(b"\0" * 3000)
    db.with_name("oc.db-wal").write_bytes(b"\0" * 500)
    db.with_name("oc.db-shm").write_bytes(b"\0" * 100)
    assert pdb.footprint_bytes(db) == 3600


def test_a_sidecar_alone_can_push_it_over_the_cap(tmp_path, monkeypatch) -> None:
    # The gap the sidecar accounting closes: a main file within the cap must not
    # short-circuit when the sidecar puts the real total over it.
    db = tmp_path / "oc.db"
    _build_db(db, [(f"ses_{i:02d}", "council", i, 64) for i in range(20)])
    main = db.stat().st_size
    _not_running(monkeypatch)
    monkeypatch.setattr(pdb, "footprint_bytes", lambda _p: main + 8 * pdb.MB)
    cap_mb = (main + pdb.MB) / pdb.MB  # main is comfortably under this
    pdb.prune(db, cap_mb=cap_mb, target_mb=cap_mb / 2, quiet=True)
    assert len(_ids(db, "session")) < 20, (
        "it short-circuited on the main file alone and ignored the sidecar"
    )


def test_target_above_cap_is_rejected() -> None:
    # Would invert the hysteresis and re-VACUUM on every session start.
    with __import__("pytest").raises(SystemExit):
        pdb.main(["--cap-mb", "100", "--target-mb", "200"])
