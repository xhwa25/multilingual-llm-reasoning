"""Crash-resilient cache, append-only reviews and conservative request accounting."""

import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


class LimitReached(Exception):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TranslationStore:
    def __init__(self, path: Path, fingerprint: str) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL, started TEXT NOT NULL,
                state TEXT NOT NULL, charge_usd TEXT NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY, digest TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
        """)
        row = self.connection.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if row and row[0] != fingerprint:
            self.connection.close()
            raise ValueError(
                "Translation inputs/provider/prompt/prices changed; use a new output_dir"
            )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO meta VALUES ('fingerprint', ?)", (fingerprint,)
            )

    def close(self) -> None:
        self.connection.close()

    def cached(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_identity(self, key: str, record: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO cache VALUES (?, ?)",
                (key, json.dumps(record, ensure_ascii=False)),
            )

    def totals(self) -> tuple[int, Decimal]:
        charges = self.connection.execute("SELECT charge_usd FROM attempts").fetchall()
        return len(charges), sum((Decimal(row[0]) for row in charges), Decimal(0))

    def previous_attempts(self, key: str) -> list[tuple[str, str | None]]:
        return self.connection.execute(
            "SELECT state, error FROM attempts WHERE cache_key=? ORDER BY id", (key,)
        ).fetchall()

    def reserve(self, key: str, charge: Decimal, *, max_requests: int, max_cost: Decimal) -> int:
        # Commit before the request: crashes retain a reservation and an attempt count.
        with self.connection:
            count, spent = self.totals()
            if count >= max_requests:
                raise LimitReached(
                    "max_requests reached (including retries and interrupted requests)"
                )
            if spent + charge > max_cost:
                raise LimitReached("max_cost_usd reservation limit reached")
            cursor = self.connection.execute(
                "INSERT INTO attempts(cache_key, started, state, charge_usd) VALUES (?, ?, ?, ?)",
                (key, utc_now(), "pending", str(charge)),
            )
            return int(cursor.lastrowid)

    def failure(self, attempt: int, error: str, *, retryable: bool) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET state=?, error=? WHERE id=?",
                ("retryable" if retryable else "permanent", error, attempt),
            )

    def success(self, attempt: int, key: str, record: dict[str, Any], actual: Decimal) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO cache VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False))
            )
            self.connection.execute(
                "UPDATE attempts SET state='completed', charge_usd=? WHERE id=?",
                (str(actual), attempt),
            )

    def import_reviews(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO reviews(digest,payload) VALUES (?,?)",
                [(key, json.dumps(row)) for key, row in events],
            )

    def reviews(self) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.connection.execute("SELECT payload FROM reviews ORDER BY id")
        ]

    def attempt_records(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute("SELECT * FROM attempts ORDER BY id")
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor]
