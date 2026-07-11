"""SQLite baseline using sqlmodel.

M1 scope: just enough to persist a conversation and its turns. The
memory-entry model (SPEC.md § 6.1) lands in M2 alongside the vector layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    project: str
    # v1.12 Stage H1: a human-readable session title (history sidebar). NULL on
    # rows from older databases (run_migrations adds the column); consumers fall
    # back to the first user message.
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    conversation_id: str = Field(foreign_key="conversation.id", index=True)
    role: str  # "system" | "user" | "assistant"
    content: str
    # v1.0 B4: which turn this message belongs to (None on pre-v1.0 rows —
    # those are never evicted by the compaction frontier).
    turn_index: Optional[int] = Field(default=None, index=True)
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Optional[float] = None
    created_at: datetime = Field(default_factory=_utcnow)


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:  # pragma: no cover
    try:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass


def run_migrations(engine) -> list[str]:
    """v0.5.0 lightweight migration: add columns present in the SQLModel
    metadata but missing from an existing table (SQLite ``create_all`` only
    creates *missing tables*, never alters existing ones). New columns get
    NULL for existing rows; the model's python default applies to new rows.

    Returns the list of "table.column" additions made (for logging/tests).
    """
    from sqlalchemy import inspect as _inspect, text as _text

    added: list[str] = []
    try:
        insp = _inspect(engine)
        existing_tables = set(insp.get_table_names())
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all handles brand-new tables
            db_cols = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in db_cols:
                    continue
                try:
                    coltype = col.type.compile(dialect=engine.dialect)
                except Exception:
                    coltype = "TEXT"
                with engine.begin() as conn:
                    conn.execute(
                        _text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}')
                    )
                added.append(f"{table.name}.{col.name}")
    except Exception:
        # Migration is best-effort; never block startup.
        pass
    return added


class Storage:
    """Thin wrapper around the SQLite engine + session lifecycle."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.sqlite_path}",
            # v1.12 F3: a 30s busy timeout so a concurrent writer (e.g. a second
            # session reopening the same profile) waits for the lock instead of
            # failing immediately with "database is locked".
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self.engine)
        run_migrations(self.engine)  # v0.5.0: add any newly-introduced columns

    def session(self) -> Session:
        return Session(self.engine)

    # --- conversation helpers ---

    def create_conversation(self, project: str) -> Conversation:
        conv = Conversation(project=project)
        with self.session() as s:
            s.add(conv)
            s.commit()
            s.refresh(conv)
        return conv

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float | None = None,
        turn_index: int | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            turn_index=turn_index,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
        with self.session() as s:
            s.add(msg)
            s.commit()
            s.refresh(msg)
        return msg

    def list_messages(self, conversation_id: str) -> list[Message]:
        with self.session() as s:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
            )
            return list(s.exec(stmt))

    # --- session management (v0.4.0) -------------------------------------

    def list_conversations(self, project: str | None = None) -> list[Conversation]:
        """Return every persisted conversation, optionally filtered by project."""
        with self.session() as s:
            stmt = select(Conversation)
            if project:
                stmt = stmt.where(Conversation.project == project)
            stmt = stmt.order_by(Conversation.created_at)
            return list(s.exec(stmt))

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.session() as s:
            return s.get(Conversation, conversation_id)

    def set_conversation_title(self, conversation_id: str, title: str) -> bool:
        """v1.12 Stage H1: set/replace a conversation's human-readable title.
        Returns False for an unknown id. The title is stored trimmed and capped
        (120 chars) so a runaway caller can't bloat the row."""
        with self.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                return False
            conv.title = (title or "").strip()[:120] or None
            s.add(conv)
            s.commit()
            return True

    def count_messages(self, conversation_id: str) -> int:
        # v1.12 H1 audit: SQL COUNT — the old select-all materialised EVERY
        # message row (full content) just to len() it, an N+1 full scan when the
        # history sidebar polls per turn.
        from sqlalchemy import func

        with self.session() as s:
            stmt = (
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conversation_id)
            )
            return int(s.exec(stmt).one())

    def first_user_message(self, conversation_id: str) -> str | None:
        """v1.12 H1: the first user message's content (LIMIT 1) — the
        deterministic title fallback, without scanning the conversation."""
        with self.session() as s:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id, Message.role == "user")
                .order_by(Message.created_at, Message.id)
                .limit(1)
            )
            m = s.exec(stmt).first()
            return m.content if m else None

    def latest_message_at(self, conversation_id: str) -> datetime | None:
        msgs = self.list_messages(conversation_id)
        if not msgs:
            return None
        return msgs[-1].created_at

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete a conversation and ALL its messages.

        Returns the number of messages removed. NOTE: callers should also
        delete the conversation's memory entries via :meth:`MemoryStore`;
        this method only handles the storage tables.
        """
        with self.session() as s:
            # Delete messages first to honour FK constraint.
            msgs = list(s.exec(select(Message).where(Message.conversation_id == conversation_id)))
            for m in msgs:
                s.delete(m)
            conv = s.get(Conversation, conversation_id)
            if conv is not None:
                s.delete(conv)
            s.commit()
            return len(msgs)
