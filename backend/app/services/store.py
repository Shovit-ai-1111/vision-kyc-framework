"""SQLite records; tokens are hashed, video state lives only in one worker."""

from contextlib import contextmanager
import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from app.core.errors import KycError


class Store:
    def __init__(self, settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.data_dir / "cases.sqlite3"
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, token TEXT, expires REAL, created REAL, payload TEXT NOT NULL)"
            )

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def purge(self):
        with self.db() as db:
            db.execute(
                "DELETE FROM cases WHERE created < ?",
                (time.time() - self.settings.retention_hours * 3600,),
            )

    def create(self, payload):
        self.purge()
        token = secrets.token_urlsafe(32)
        now = time.time()
        payload.update(
            id=str(uuid.uuid4()),
            revision=0,
            created_at=now,
            documents=[],
            review_history=[],
            result=None,
            liveness=None,
        )
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                >= self.settings.max_cases
            ):
                raise KycError(
                    "capacity", "Case capacity reached. Delete older cases.", 503
                )
            db.execute(
                "INSERT INTO cases VALUES(?,?,?,?,?)",
                (
                    payload["id"],
                    hashlib.sha256(token.encode()).hexdigest(),
                    now + self.settings.session_ttl_seconds,
                    now,
                    json.dumps(payload),
                ),
            )
        return payload, token

    def get(self, case_id, token=None, reviewer=False):
        self.purge()
        with self.db() as db:
            row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        if not row:
            raise KycError("not_found", "Case does not exist or has expired.", 404)
        if not reviewer:
            digest = hashlib.sha256((token or "").encode()).hexdigest()
            if not secrets.compare_digest(digest, row["token"]):
                raise KycError("unauthorized", "Invalid session token.", 401)
            if row["expires"] < time.time():
                raise KycError("expired", "Session expired. Start a new case.", 410)
        return json.loads(row["payload"])

    def save(self, case):
        previous = case["revision"]
        case["revision"] += 1
        with self.db() as db:
            changed = db.execute(
                "UPDATE cases SET payload=? WHERE id=? AND json_extract(payload, '$.revision')=?",
                (json.dumps(case, allow_nan=False), case["id"], previous),
            ).rowcount
        if changed != 1:
            case["revision"] = previous
            raise KycError(
                "revision_conflict", "Case changed. Reload before editing.", 409
            )

    def list(self):
        self.purge()
        with self.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM cases ORDER BY created DESC LIMIT 200"
                )
            ]

    def delete(self, case_id):
        with self.db() as db:
            db.execute("DELETE FROM cases WHERE id=?", (case_id,))
