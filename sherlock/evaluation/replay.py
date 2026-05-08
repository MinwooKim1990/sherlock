"""Replay a dummy_conversation.md file through a Sherlock instance, turn by turn.

The dummy file format is fixed by EVALUATION_PROTOCOL.md §1.3:
  ### Turn N
  **User:** ...
  **Assistant:** ...

We feed each `**User:**` text into `agent.chat()` and ignore the dummy's
own assistant reply (Sherlock generates its own). The point of the replay
is to populate the memory store so the post-replay summary+inference
output can be scored against the gold standard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sherlock.agent import Sherlock


_TURN_HEADER_RE = re.compile(r"^###\s+Turn\s+(\d+)\s*$", re.MULTILINE)
_USER_RE = re.compile(r"^\*\*User:\*\*\s*(.*?)(?=^\*\*Assistant:\*\*|^###\s+Turn|\Z)", re.MULTILINE | re.DOTALL)


@dataclass
class ReplayTurn:
    turn_number: int
    user: str


def parse_dummy_conversation(path: str | Path) -> list[ReplayTurn]:
    text = Path(path).read_text(encoding="utf-8")
    # Cut off everything before "## Conversation" so the persona summary
    # at the top doesn't leak in as a turn.
    if "## Conversation" in text:
        text = text.split("## Conversation", 1)[1]
    # Cut off "## Notes for evaluators" tail.
    if "## Notes for evaluators" in text:
        text = text.split("## Notes for evaluators", 1)[0]

    turns: list[ReplayTurn] = []
    # Walk turn headers.
    parts = re.split(r"^###\s+Turn\s+(\d+)\s*$", text, flags=re.MULTILINE)
    # parts pattern: [pre, num, body, num, body, ...]
    for i in range(1, len(parts), 2):
        try:
            num = int(parts[i])
        except (ValueError, IndexError):
            continue
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Extract the **User:** segment (everything up to **Assistant:** or end).
        m = re.search(
            r"\*\*User:\*\*\s*(.*?)(?=\*\*Assistant:\*\*|\Z)",
            body,
            flags=re.DOTALL,
        )
        if not m:
            continue
        user_text = m.group(1).strip()
        if user_text:
            turns.append(ReplayTurn(turn_number=num, user=user_text))
    return turns


def replay_dummy_conversation(
    agent: Sherlock,
    dummy_path: str | Path,
    *,
    max_turns: int | None = None,
    progress_callback=None,
) -> list[ReplayTurn]:
    """Drive the agent through every user turn in the dummy. Returns the parsed turns."""
    turns = parse_dummy_conversation(dummy_path)
    if max_turns is not None:
        turns = turns[:max_turns]
    # Tell the agent the total run length so its safety-net force-fire on
    # the last turn can trigger when LLM-1 never asked for any companion.
    agent._replay_total_turns = len(turns)
    for i, t in enumerate(turns):
        try:
            agent.chat(t.user)
        except Exception as exc:
            if progress_callback:
                progress_callback(i, t, error=str(exc))
            continue
        if progress_callback:
            progress_callback(i, t, error=None)
    return turns
