"""SQLite persistence; parameterized SQL and one connection per operation."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"open", "in_review", "resolved"}


class TicketStore:
    def __init__(self, path: str):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, customer TEXT NOT NULL, subject TEXT NOT NULL,
                message TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL,
                analysis TEXT, draft TEXT NOT NULL DEFAULT '')""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["analysis"] = json.loads(result["analysis"]) if result["analysis"] else None
        return result

    def list(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM tickets ORDER BY id DESC")]

    def get(self, ticket_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,)).fetchone())

    def create(self, customer, subject, message, analysis=None):
        with self.connect() as db:
            cursor = db.execute("INSERT INTO tickets(customer,subject,message,created_at,analysis,draft) VALUES(?,?,?,?,?,?)",
                (customer, subject, message, datetime.now(timezone.utc).isoformat(),
                 json.dumps(analysis) if analysis else None, analysis["draft"]["text"] if analysis else ""))
            ticket_id = cursor.lastrowid
        return self.get(ticket_id)

    def update(self, ticket_id, status, draft):
        if status not in STATUSES:
            raise ValueError("Invalid ticket status")
        with self.connect() as db:
            cursor = db.execute("UPDATE tickets SET status=?, draft=? WHERE id=?", (status, draft, ticket_id))
            if cursor.rowcount == 0:
                return None
        return self.get(ticket_id)

    def save_analysis(self, ticket_id, analysis):
        with self.connect() as db:
            db.execute("UPDATE tickets SET analysis=?, draft=? WHERE id=?", (json.dumps(analysis), analysis["draft"]["text"], ticket_id))
        return self.get(ticket_id)
