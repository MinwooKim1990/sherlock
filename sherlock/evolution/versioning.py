"""Companion-prompt version persistence + rollback (SPEC §5.3 minimum surface)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Session, SQLModel, select


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class CompanionPrompt(SQLModel, table=True):
    """One row per saved companion-prompt revision."""

    __tablename__ = "companion_prompt"

    id: str = Field(default_factory=_new_id, primary_key=True)
    project: str = Field(index=True)
    role: str = Field(index=True)  # "llm2" | "llm3"
    version: int = Field(index=True)
    content: str
    rationale: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    is_active: bool = Field(default=True, index=True)


class PromptVersionStore:
    def __init__(self, engine) -> None:
        self._engine = engine
        SQLModel.metadata.create_all(self._engine)

    def save(
        self, *, project: str, role: str, content: str, rationale: str = ""
    ) -> CompanionPrompt:
        with Session(self._engine) as s:
            # Deactivate prior active for the same role+project.
            stmt = select(CompanionPrompt).where(
                CompanionPrompt.project == project,
                CompanionPrompt.role == role,
                CompanionPrompt.is_active == True,  # noqa: E712
            )
            for row in s.exec(stmt):
                row.is_active = False
                s.add(row)
            stmt2 = select(CompanionPrompt).where(
                CompanionPrompt.project == project,
                CompanionPrompt.role == role,
            )
            existing = list(s.exec(stmt2))
            next_version = (max((r.version for r in existing), default=0)) + 1
            row = CompanionPrompt(
                project=project,
                role=role,
                version=next_version,
                content=content,
                rationale=rationale,
                is_active=True,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            return row

    def active(self, *, project: str, role: str) -> Optional[CompanionPrompt]:
        with Session(self._engine) as s:
            stmt = select(CompanionPrompt).where(
                CompanionPrompt.project == project,
                CompanionPrompt.role == role,
                CompanionPrompt.is_active == True,  # noqa: E712
            )
            return next(iter(s.exec(stmt)), None)

    def list(self, *, project: str, role: str) -> list[CompanionPrompt]:
        with Session(self._engine) as s:
            stmt = (
                select(CompanionPrompt)
                .where(CompanionPrompt.project == project, CompanionPrompt.role == role)
                .order_by(CompanionPrompt.version.desc())
            )
            return list(s.exec(stmt))

    def rollback(self, *, project: str, role: str, version: int) -> Optional[CompanionPrompt]:
        with Session(self._engine) as s:
            stmt = select(CompanionPrompt).where(
                CompanionPrompt.project == project,
                CompanionPrompt.role == role,
            )
            rows = list(s.exec(stmt))
            target = next((r for r in rows if r.version == version), None)
            if target is None:
                return None
            for r in rows:
                r.is_active = r.id == target.id
                s.add(r)
            s.commit()
            return target
