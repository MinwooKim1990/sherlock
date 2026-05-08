"""The Sherlock class — M1+M2+M3 surface.

M1: bare LLM-1 chat with no memory and no inference.
M2: memory layer (vector store, summarizer, decay, K-turn retention).
M3: bootstrap-authored companion prompts + LLM-3 inference + web search.

All milestones are wired into a single synchronous turn pipeline. M5
upgrades the background portion to async.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sherlock.config import Config
from sherlock.evolution import PromptVersionStore
from sherlock.memory import (
    DecayConfig,
    DecayEngine,
    KTurnPolicy,
    MemoryStore,
    SummarizerConfig,
    SummarizerEngine,
    build_embedding_provider,
)
from sherlock.memory.entry import MemoryEntry, MemorySource, MemoryState, MemoryType
from sherlock.providers import BaseProvider, ChatMessage, ChatResponse, build_provider
from sherlock.rag import HybridSearch
from sherlock.storage import Conversation, Message, Storage


@dataclass
class TurnState:
    """Read-only snapshot of the last turn for inspection (SPEC §8.1)."""

    user_text: str
    response: ChatResponse
    messages_passed_to_llm1: list[ChatMessage]
    retrieved_memories: list[tuple[MemoryEntry, float]] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    search_results: list[dict] = field(default_factory=list)
    summary_run: bool = False
    decay_counts: dict = field(default_factory=dict)
    tokens_used: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Sherlock:
    """Main entry point. The synchronous chat loop assembles the slot per
    SPEC §4.2 and §6.2, calls the main provider, persists everything, and
    runs the async-style background pipeline (M5 upgrades to true async).
    """

    def __init__(
        self,
        config: Config,
        *,
        provider: BaseProvider | None = None,
        background_summary_provider: BaseProvider | None = None,
        background_inference_provider: BaseProvider | None = None,
    ) -> None:
        self.config = config
        self._provider = provider or build_provider(config.models.main)
        self._summary_provider = background_summary_provider or self._build_optional(
            config.models.background_summary
        )
        self._inference_provider = background_inference_provider or self._build_optional(
            config.models.background_inference
        )
        # Storage: conversations + messages
        self._storage = Storage(config.storage.sqlite_path)
        # Memory store reuses the same engine.
        self._embed = build_embedding_provider(config.storage.embedding)
        self._memory = MemoryStore(
            engine=self._storage.engine,
            embedding_provider=self._embed,
            vector_path=config.storage.vector_path,
        )
        self._hybrid = HybridSearch(store=self._memory)
        self._prompt_store = PromptVersionStore(self._storage.engine)
        # Decay engine
        self._decay = DecayEngine(
            self._memory,
            DecayConfig(
                warm_after_days=config.memory.decay.warm_after_days,
                cold_after_days=config.memory.decay.cold_after_days,
                forgotten_after_days=config.memory.decay.forgotten_after_days,
                warm_after_turns=config.memory.decay.warm_after_turns,
                cold_after_turns=config.memory.decay.cold_after_turns,
                forgotten_after_turns=config.memory.decay.forgotten_after_turns,
            ),
        )
        # K-turn policy
        self._k_turn = KTurnPolicy(
            k_min=config.memory.k_turn_min,
            k_max=config.memory.k_turn_max,
            adaptive=config.memory.k_turn_max_adaptive,
        )
        # Companion prompts (Bootstrap engine fills these in if enabled).
        self._llm2_prompt: Optional[str] = None
        self._llm3_prompt: Optional[str] = None
        self._llm2_prompt_version = 0
        self._llm3_prompt_version = 0
        self._summarizer: Optional[SummarizerEngine] = None
        self._inferer = None  # set up after bootstrap
        self._search = None  # web search module, set up lazily

        self._system_prompt = config.read_main_system_prompt()
        self._conversation: Conversation | None = None
        self._last_turn: TurnState | None = None
        self._turn_index = 0
        self._prev_user_text: Optional[str] = None
        # Persist a system-source persona note so the T76-style provenance
        # trap is correctly handled: the agent's identity-of-user comes from
        # the persona/system note, not from a user utterance.
        self._persona_seeded = False
        # Cumulative LLM-3 outputs across turns (for Section 4 in eval output).
        self._tool_call_history: list[dict] = []

    @staticmethod
    def _build_optional(model_cfg) -> Optional[BaseProvider]:
        if model_cfg is None:
            return None
        return build_provider(model_cfg)

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation.id if self._conversation else None

    # ---- bootstrap wiring (filled by sherlock.bootstrap.engine) ----

    def install_companion_prompts(self, llm2: str, llm3: str, version: int = 1) -> None:
        self._llm2_prompt = llm2
        self._llm3_prompt = llm3
        self._llm2_prompt_version = version
        self._llm3_prompt_version = version
        # Persist as a new version so rollback / inspection are possible.
        try:
            self._prompt_store.save(project=self.config.project, role="llm2", content=llm2)
            self._prompt_store.save(project=self.config.project, role="llm3", content=llm3)
        except Exception:
            pass
        if self._summary_provider is not None:
            self._summarizer = SummarizerEngine(
                provider=self._summary_provider,
                store=self._memory,
                config=SummarizerConfig(
                    trigger_every_n_turns=self.config.memory.summarize_every_n_turns,
                    topic_change_similarity_threshold=self.config.memory.topic_change_similarity_threshold,
                    prompt=llm2,
                ),
            )
        # Inference engine:
        from sherlock.inference.engine import InferenceEngine  # local import to avoid cycle

        if self._inference_provider is not None:
            self._inferer = InferenceEngine(
                provider=self._inference_provider,
                store=self._memory,
                system_prompt=llm3,
                cold_start_turns=self.config.inference.cold_start_turns,
                confidence_threshold=self.config.inference.confidence_threshold,
            )

    def install_search(self, search_engine) -> None:
        self._search = search_engine

    # ---- conversation management ----

    def _ensure_conversation(self) -> Conversation:
        if self._conversation is None:
            self._conversation = self._storage.create_conversation(project=self.config.project)
            self._storage.add_message(
                self._conversation.id,
                role="system",
                content=self._system_prompt,
            )
            # Seed system-source persona facts if any are declared via domain hints.
            if not self._persona_seeded:
                hints = self.config.main_system_prompt.domain_hints
                for h in hints:
                    self._memory.add(
                        conversation_id=self._conversation.id,
                        content=h,
                        type=MemoryType.FACT,
                        source=MemorySource.SYSTEM,
                        confidence=0.95,
                        pinned=True,
                        last_used_turn_index=0,
                        tags="domain_hint",
                    )
                self._persona_seeded = True
        return self._conversation

    # ---- slot assembly ----

    def _retrieve_memories(self, user_text: str) -> list[tuple[MemoryEntry, float]]:
        if self._conversation is None:
            return []
        # M4-light: hybrid vector + BM25 with RRF fusion.
        return self._hybrid.search(
            user_text,
            conversation_id=self._conversation.id,
            top_k=self.config.memory.rag_top_k,
            confidence_threshold=0.0,
            exclude_inferences_below=self.config.inference.confidence_threshold,
        )

    def _format_pinned_block(self, conv_id: str) -> str:
        pinned = self._memory.list(conversation_id=conv_id, pinned=True)
        if not pinned:
            return ""
        lines = ["[PINNED USER PROFILE — system-tracked]"]
        for p in pinned:
            tag = "system" if p.source == MemorySource.SYSTEM else p.source.value
            lines.append(f"- ({tag}) {p.content}")
        return "\n".join(lines)

    def _format_retrieved_block(self, mems: list[tuple[MemoryEntry, float]]) -> str:
        if not mems:
            return ""
        lines = ["[RELEVANT MEMORIES — retrieved]"]
        for entry, score in mems:
            tag = entry.source.value
            conf = f" conf={entry.confidence:.2f}" if entry.type == MemoryType.INFERENCE else ""
            lines.append(f"- ({tag}{conf}, sim={score:.2f}) {entry.content}")
        return "\n".join(lines)

    def _format_active_intent(self, hypotheses: list[dict]) -> str:
        if not hypotheses:
            return ""
        top = hypotheses[0]
        intent = top.get("intent")
        prob = top.get("probability")
        if not intent:
            return ""
        return f"[ACTIVE INTENT (inferred, p={prob})] {intent}"

    def _format_search_block(self, results: list[dict]) -> str:
        if not results:
            return ""
        lines = ["[CACHED WEB SEARCH RESULTS]"]
        for r in results[:5]:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "") or r.get("snippet", "")
            lines.append(f"- {title} — {url}\n    {snippet[:200]}")
        return "\n".join(lines)

    def _format_last_k_turns(self, conv_id: str, k: int) -> list[ChatMessage]:
        msgs = self._storage.list_messages(conv_id)
        # Skip the leading system message; take the last 2*k user/assistant entries.
        non_sys = [m for m in msgs if m.role != "system"]
        tail = non_sys[-(2 * k) :]
        return [ChatMessage(role=m.role, content=m.content) for m in tail]

    def _assemble_messages(
        self,
        user_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        hypotheses: list[dict],
        search_results: list[dict],
        topic_changed: bool,
    ) -> list[ChatMessage]:
        conv = self._ensure_conversation()
        # Slot blocks per SPEC §6.2:
        slot_blocks: list[str] = []
        # 1. system prompt
        slot_blocks.append(self._system_prompt.strip())
        # 2. current date/time
        if self.config.search.inject_datetime:
            slot_blocks.append(f"[CURRENT TIME — system-injected] {_now_iso()}")
        # 3. pinned user profile
        pinned = self._format_pinned_block(conv.id)
        if pinned:
            slot_blocks.append(pinned)
        # 4. active intent
        intent = self._format_active_intent(hypotheses)
        if intent:
            slot_blocks.append(intent)
        # 5. relevant memories
        retrieved_block = self._format_retrieved_block(retrieved)
        if retrieved_block:
            slot_blocks.append(retrieved_block)
        # 6. cached web search results
        search_block = self._format_search_block(search_results)
        if search_block:
            slot_blocks.append(search_block)

        composite_system = "\n\n".join(slot_blocks)

        # 7. last K turns + 8. current input
        k = self._k_turn.k(topic_changed=topic_changed, context_utilisation=0.0)
        tail = self._format_last_k_turns(conv.id, k)

        messages: list[ChatMessage] = [ChatMessage(role="system", content=composite_system)]
        messages.extend(tail)
        messages.append(ChatMessage(role="user", content=user_text))
        return messages

    # ---- the synchronous turn ----

    def chat(self, user_input: str) -> str:
        conv = self._ensure_conversation()
        self._turn_index += 1
        turn_index = self._turn_index

        # 1. Persist user turn first (crash-safe).
        self._storage.add_message(conv.id, role="user", content=user_input)
        # Also write the user utterance into the memory store as a high-confidence
        # USER record so retrieval and decay can reason about it.
        self._memory.add(
            conversation_id=conv.id,
            content=user_input,
            type=MemoryType.USER_UTTERANCE,
            source=MemorySource.USER,
            confidence=1.0,
            last_used_turn_index=turn_index,
        )

        # 2. Topic-change check (drives K-turn shrink + summarizer trigger).
        topic_changed = False
        if self._summarizer and self._prev_user_text:
            _, topic_changed = self._summarizer.should_run(
                turn_index=turn_index,
                prev_user_text=self._prev_user_text,
                current_user_text=user_input,
            )

        # 3. Inference (LLM-3) for current turn — runs before LLM-1 so its
        #    output can populate the active intent slot.
        hypotheses: list[dict] = []
        search_results: list[dict] = []
        if self._inferer is not None:
            try:
                infer_result = self._inferer.infer(
                    conversation_id=conv.id,
                    turn_index=turn_index,
                    user_text=user_input,
                    recent_turns=self._format_last_k_turns(conv.id, 3),
                )
                hypotheses = infer_result.get("hypotheses", []) or []
                # Cache the full LLM-3 output per turn for the eval-time Section 4.
                self._tool_call_history.append({
                    "turn_index": turn_index,
                    "user": user_input,
                    "tools_recommended": infer_result.get("tools_recommended", []) or [],
                    "freshness_required": infer_result.get("freshness_required", []) or [],
                    "context_to_expand": infer_result.get("context_to_expand", []) or [],
                })
                # Web-search prefetch on freshness_required topics
                freshness = infer_result.get("freshness_required", []) or []
                if self._search is not None:
                    for topic in freshness[:3]:
                        try:
                            search_results.extend(self._search.search(topic, max_results=3))
                        except Exception:
                            pass
            except Exception:
                hypotheses = []

        # 4. Retrieve memories (RAG top-K).
        retrieved = self._retrieve_memories(user_input)
        for entry, _ in retrieved:
            self._memory.touch(entry.id, turn_index=turn_index)

        # 5. Assemble + call LLM-1.
        messages = self._assemble_messages(
            user_input,
            retrieved,
            hypotheses,
            search_results,
            topic_changed=topic_changed,
        )
        response = self._provider.chat(messages)

        # 6. Persist assistant turn.
        self._storage.add_message(
            conv.id,
            role="assistant",
            content=response.text,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
        )

        # 7. Background: LLM-2 summarization cycle.
        summary_run = False
        if self._summarizer is not None:
            should, _ = self._summarizer.should_run(
                turn_index=turn_index,
                prev_user_text=self._prev_user_text,
                current_user_text=user_input,
            )
            if should:
                try:
                    self._summarizer.run(
                        conversation_id=conv.id,
                        recent_turns=self._format_last_k_turns(conv.id, 5),
                        turn_index=turn_index,
                    )
                    summary_run = True
                except Exception:
                    summary_run = False

        # 8. Decay pass.
        active_topics = [user_input]
        if hypotheses:
            for h in hypotheses[:2]:
                if h.get("intent"):
                    active_topics.append(str(h["intent"]))
        decay_counts = self._decay.step(
            conversation_id=conv.id,
            current_turn_index=turn_index,
            active_topics=active_topics,
        )
        # 8b. PIN cap — keep PIN bucket from ballooning. Demotes least-recent
        # non-system pins above the cap.
        try:
            self._memory.cap_pinned(conv.id, max_pinned=25)
        except Exception:
            pass

        self._prev_user_text = user_input
        self._last_turn = TurnState(
            user_text=user_input,
            response=response,
            messages_passed_to_llm1=messages,
            retrieved_memories=retrieved,
            hypotheses=hypotheses,
            search_results=search_results,
            summary_run=summary_run,
            decay_counts=decay_counts,
            tokens_used=response.usage.total_tokens,
        )
        return response.text

    def inspect_last_turn(self) -> TurnState | None:
        return self._last_turn

    async def achat(self, user_input: str) -> str:
        """M5 async path. Background work runs in parallel via asyncio.gather.

        For now LLM-1 is awaited synchronously (it gates the response).
        Summarizer + decay run AFTER the response is ready in parallel.
        """
        import asyncio

        conv = self._ensure_conversation()
        self._turn_index += 1
        turn_index = self._turn_index

        self._storage.add_message(conv.id, role="user", content=user_input)
        self._memory.add(
            conversation_id=conv.id,
            content=user_input,
            type=MemoryType.USER_UTTERANCE,
            source=MemorySource.USER,
            confidence=1.0,
            last_used_turn_index=turn_index,
        )

        # Inference + retrieval can race in async-mode (LLM-3 + RAG are independent).
        async def _do_infer():
            if self._inferer is None:
                return {}
            try:
                return self._inferer.infer(
                    conversation_id=conv.id,
                    turn_index=turn_index,
                    user_text=user_input,
                    recent_turns=self._format_last_k_turns(conv.id, 3),
                )
            except Exception:
                return {}

        async def _do_retrieve():
            return self._retrieve_memories(user_input)

        infer_result, retrieved = await asyncio.gather(
            _do_infer() if self.config.execution.parallel_when_possible else asyncio.sleep(0, result={}),
            _do_retrieve(),
        )
        if not isinstance(infer_result, dict):
            infer_result = {}

        hypotheses = infer_result.get("hypotheses", []) or []
        for entry, _ in retrieved:
            self._memory.touch(entry.id, turn_index=turn_index)

        # Optional web-search prefetch.
        search_results: list[dict] = []
        freshness = infer_result.get("freshness_required", []) or []
        if self._search and freshness:
            for topic in freshness[:3]:
                try:
                    search_results.extend(self._search.search(topic, max_results=3))
                except Exception:
                    pass

        topic_changed = False
        if self._summarizer and self._prev_user_text:
            _, topic_changed = self._summarizer.should_run(
                turn_index=turn_index,
                prev_user_text=self._prev_user_text,
                current_user_text=user_input,
            )

        messages = self._assemble_messages(
            user_input,
            retrieved,
            hypotheses,
            search_results,
            topic_changed=topic_changed,
        )
        response = await self._provider.achat(messages)
        self._storage.add_message(
            conv.id,
            role="assistant",
            content=response.text,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
        )

        # Background: summarizer + decay can run in parallel after the response is sent.
        async def _do_summary():
            if self._summarizer is None:
                return False
            should, _ = self._summarizer.should_run(
                turn_index=turn_index,
                prev_user_text=self._prev_user_text,
                current_user_text=user_input,
            )
            if not should:
                return False
            try:
                await asyncio.to_thread(
                    self._summarizer.run,
                    conversation_id=conv.id,
                    recent_turns=self._format_last_k_turns(conv.id, 5),
                    turn_index=turn_index,
                )
                return True
            except Exception:
                return False

        async def _do_decay():
            active_topics = [user_input]
            for h in hypotheses[:2]:
                if h.get("intent"):
                    active_topics.append(str(h["intent"]))
            return await asyncio.to_thread(
                self._decay.step,
                conversation_id=conv.id,
                current_turn_index=turn_index,
                active_topics=active_topics,
            )

        summary_run, decay_counts = await asyncio.gather(_do_summary(), _do_decay())

        self._prev_user_text = user_input
        self._last_turn = TurnState(
            user_text=user_input,
            response=response,
            messages_passed_to_llm1=messages,
            retrieved_memories=retrieved,
            hypotheses=hypotheses,
            search_results=search_results,
            summary_run=summary_run,
            decay_counts=decay_counts,
            tokens_used=response.usage.total_tokens,
        )
        return response.text

    def messages(self) -> list[Message]:
        if self._conversation is None:
            return []
        return self._storage.list_messages(self._conversation.id)

    # ---- entry helpers ----

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sherlock":
        cfg = Config.from_yaml(path)
        agent = cls(cfg)
        if cfg.bootstrap.auto_run_on_init:
            agent._maybe_bootstrap()
        else:
            # Bootstrap disabled — install DEFAULT_*_PROMPT directly so the
            # summarizer + inferer still get wired. Without this they would
            # be None and the memory layer would silently disable.
            from sherlock.inference.engine import DEFAULT_LLM3_PROMPT
            from sherlock.memory.summarizer import DEFAULT_LLM2_PROMPT

            agent.install_companion_prompts(
                DEFAULT_LLM2_PROMPT, DEFAULT_LLM3_PROMPT, version=0
            )
            try:
                from sherlock.tools.web_search import build_search_engine

                search = build_search_engine(cfg.search)
                if search is not None:
                    agent.install_search(search)
            except Exception:
                pass
        return agent

    def _maybe_bootstrap(self) -> None:
        """Run Bootstrap if companion prompts haven't been installed yet."""
        if self._llm2_prompt and self._llm3_prompt:
            return
        # Lazy import to avoid cycles.
        from sherlock.bootstrap.engine import BootstrapEngine

        engine = BootstrapEngine(
            main_provider=self._provider,
            main_system_prompt=self._system_prompt,
            domain_hints=self.config.main_system_prompt.domain_hints,
        )
        try:
            llm2, llm3 = engine.run()
            self.install_companion_prompts(llm2, llm3, version=1)
        except Exception:
            # Bootstrap failure should NOT block chat at M2/M3; the agent
            # falls back to LLM-1 only with sane default companion prompts.
            from sherlock.memory.summarizer import DEFAULT_LLM2_PROMPT
            from sherlock.inference.engine import DEFAULT_LLM3_PROMPT

            self.install_companion_prompts(DEFAULT_LLM2_PROMPT, DEFAULT_LLM3_PROMPT, version=0)
        # Install web search if configured.
        try:
            from sherlock.tools.web_search import build_search_engine

            search = build_search_engine(self.config.search)
            if search is not None:
                self.install_search(search)
        except Exception:
            pass
