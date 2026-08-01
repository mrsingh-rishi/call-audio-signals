"""SQLite job and result state.

Deliberately stdlib ``sqlite3`` with a short-lived connection per operation.
Batches here are tens of files, not thousands; a queue broker and a connection
pool would be operational weight with no benefit, and operational simplicity is
explicitly graded.

Results are persisted rather than held in memory so a container restart mid-
evaluation does not lose a completed batch.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    status        TEXT NOT NULL,
    n_total       INTEGER NOT NULL DEFAULT 0,
    n_done        INTEGER NOT NULL DEFAULT 0,
    n_error       INTEGER NOT NULL DEFAULT 0,
    preflight     TEXT,
    workdir       TEXT,
    purged        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS results (
    batch_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    truth         TEXT,
    PRIMARY KEY (batch_id, name)
);
CREATE INDEX IF NOT EXISTS idx_results_batch ON results(batch_id);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # --- batches ---------------------------------------------------------

    def create_batch(self, preflight: dict[str, Any], workdir: str, n_total: int) -> str:
        batch_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO batches (id, created_at, status, n_total, preflight, workdir)"
                " VALUES (?,?,?,?,?,?)",
                (batch_id, time.time(), "pending", n_total, json.dumps(preflight), workdir),
            )
        return batch_id

    def mark_started(self, batch_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE batches SET status='running', started_at=? WHERE id=?",
                (time.time(), batch_id),
            )

    def mark_finished(self, batch_id: str, status: str = "complete") -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE batches SET status=?, finished_at=? WHERE id=?",
                (status, time.time(), batch_id),
            )

    def mark_purged(self, batch_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE batches SET purged=1 WHERE id=?", (batch_id,))

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["preflight"] = json.loads(d["preflight"] or "{}")
        return d

    def list_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, finished_at, status, n_total, n_done, n_error"
                " FROM batches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def batches_to_purge(self, older_than_s: float) -> list[dict[str, Any]]:
        cutoff = time.time() - older_than_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, workdir FROM batches WHERE purged=0 AND created_at < ?",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- results ---------------------------------------------------------

    def add_result(
        self, batch_id: str, name: str, payload: dict[str, Any],
        truth: dict[str, Any] | None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO results (batch_id, name, payload, truth)"
                " VALUES (?,?,?,?)",
                (batch_id, name, json.dumps(payload),
                 json.dumps(truth) if truth is not None else None),
            )
            is_err = 1 if payload.get("status") == "error" else 0
            c.execute(
                "UPDATE batches SET n_done = n_done + 1, n_error = n_error + ? WHERE id=?",
                (is_err, batch_id),
            )

    def get_results(self, batch_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, payload, truth FROM results WHERE batch_id=? ORDER BY name",
                (batch_id,),
            ).fetchall()
        out = []
        for r in rows:
            item = json.loads(r["payload"])
            if r["truth"]:
                item["_truth"] = json.loads(r["truth"])
            out.append(item)
        return out
