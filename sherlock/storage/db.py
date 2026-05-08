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
    created_at: datetime = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    conversation_id: str = Field(foreign_key="conversation.id", index=True)
    role: str  # "system" | "user" | "assistant"
    content: str
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


class Storage:
    """Thin wrapper around the SQLite engine + session lifecycle."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.sqlite_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(self.engine)

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
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
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
