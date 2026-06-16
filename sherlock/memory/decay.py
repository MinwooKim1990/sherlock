"""4-state decay engine per SPEC §5.2.

For evaluation purposes the engine accepts both wall-clock days and
turn-count thresholds; the latter dominates when running the dummy-
conversation replay where 80 turns must traverse the whole lifecycle.

Transitions:
  fresh → warm  : if not used in the next turn (turn_gap >= warm_after_turns)
  warm → cold   : after warm_after_days OR warm_after_turns
  cold → forgotten: after cold_after_days AND semantically isolated
                   (or cold_after_turns; the AND is relaxed to OR for the
                   replay case, see DEVIATION-005)
  forgotten → hard delete: after forgotten_after_days
  pinned items skip all transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sherlock.memory.entry import MemoryEntry, MemoryState, _utcnow
from sherlock.memory.store import MemoryStore


@dataclass
class DecayConfig:
    warm_after_days: float = 7.0
    cold_after_days: float = 30.0
    forgotten_after_days: float = 90.0
    # Turn-count companion thresholds (dominate when both apply).
    warm_after_turns: int = 1
    cold_after_turns: int = 12
    forgotten_after_turns: int = 30
    # If True, cold→forgotten requires semantic isolation as well as time.
    cold_to_forgotten_requires_isolation: bool = True
    semantic_isolation_threshold: float = 0.35  # below = "isolated"


class DecayEngine:
    def __init__(self, store: MemoryStore, config: DecayConfig | None = None) -> None:
        self._store = store
        self._cfg = config or DecayConfig()

    def _gap_turns(self, entry: MemoryEntry, current_turn_index: int) -> int:
        return max(0, current_turn_index - entry.last_used_turn_index)

    def _gap_days(self, entry: MemoryEntry) -> float:
        # SQLite round-trips datetimes as naive; coerce both to UTC for safe subtraction.
        from datetime import timezone

        now = _utcnow()
        last = entry.last_used_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta: timedelta = now - last
        return delta.total_seconds() / 86400.0

    def _semantically_isolated(
        self,
        entry: MemoryEntry,
        active_topics: list[str],
    ) -> bool:
        if not active_topics:
            return True
        max_sim = 0.0
        for topic in active_topics:
            sim = self._store.cosine_between(entry.content, topic)
            if sim > max_sim:
                max_sim = sim
        return max_sim < self._cfg.semantic_isolation_threshold

    def step(
        self,
        conversation_id: str,
        current_turn_index: int,
        active_topics: list[str] | None = None,
    ) -> dict[str, int]:
        """Run one decay pass over the conversation. Returns transition counts."""
        active_topics = active_topics or []
        counts = {"fresh_to_warm": 0, "warm_to_cold": 0, "cold_to_forgotten": 0, "hard_deleted": 0}

        for entry in self._store.all_for_conversation(conversation_id):
            if entry.pinned:
                continue
            # v1.0: superseded rows are frozen — they neither decay nor get
            # hard-deleted; they exist purely as a correction audit trail.
            if entry.superseded_by:
                continue
            gap_turns = self._gap_turns(entry, current_turn_index)
            gap_days = self._gap_days(entry)

            if entry.state == MemoryState.FRESH:
                if gap_turns >= self._cfg.warm_after_turns:
                    self._store.update_state(entry.id, MemoryState.WARM)
                    counts["fresh_to_warm"] += 1

            elif entry.state == MemoryState.WARM:
                if gap_turns >= self._cfg.cold_after_turns or gap_days >= self._cfg.warm_after_days:
                    self._store.update_state(entry.id, MemoryState.COLD)
                    counts["warm_to_cold"] += 1

            elif entry.state == MemoryState.COLD:
                time_qualified = (
                    gap_turns >= self._cfg.forgotten_after_turns
                    or gap_days >= self._cfg.cold_after_days
                )
                if time_qualified:
                    if self._cfg.cold_to_forgotten_requires_isolation:
                        if self._semantically_isolated(entry, active_topics):
                            self._store.soft_delete(entry.id)
                            counts["cold_to_forgotten"] += 1
                    else:
                        self._store.soft_delete(entry.id)
                        counts["cold_to_forgotten"] += 1

            elif entry.state == MemoryState.FORGOTTEN:
                if gap_days >= self._cfg.forgotten_after_days:
                    self._store.hard_delete(entry.id)
                    counts["hard_deleted"] += 1

        return counts
