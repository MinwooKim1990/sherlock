"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => (s == null ? "" : String(s)).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmt = (o) => { try { return JSON.stringify(o, null, 2); } catch { return String(o); } };
// v1.3: markdown for ASSISTANT bubbles only (sherlock + baseline replies).
// marked + DOMPurify come from CDN <script> tags; offline we fall back to the
// escaped plain text with <br> so nothing ever renders unsanitized.
const mdRender = (text) => {
  const s = text == null ? "" : String(text);
  if (window.marked && window.DOMPurify) {
    try { return DOMPurify.sanitize(marked.parse(s, { breaks: true })); } catch { /* fall through */ }
  }
  return esc(s).replace(/\n/g, "<br>");
};

const ACTOR = {
  llm1: { stripe: "bg-blue-500", text: "text-blue-700", soft: "bg-blue-50", n: "LLM-1" },
  llm2: { stripe: "bg-green-500", text: "text-green-700", soft: "bg-green-50", n: "LLM-2" },
  llm3: { stripe: "bg-purple-500", text: "text-purple-700", soft: "bg-purple-50", n: "LLM-3" },
  memory: { stripe: "bg-amber-500", text: "text-amber-700", soft: "bg-amber-50", n: "memory" },
  decay: { stripe: "bg-slate-400", text: "text-slate-600", soft: "bg-slate-50", n: "decay" },
  carry: { stripe: "bg-rose-500", text: "text-rose-700", soft: "bg-rose-50", n: "carry" },
  slot: { stripe: "bg-indigo-500", text: "text-indigo-700", soft: "bg-indigo-50", n: "slot" },
  tool: { stripe: "bg-teal-500", text: "text-teal-700", soft: "bg-teal-50", n: "tool" },
  system: { stripe: "bg-slate-400", text: "text-slate-600", soft: "bg-slate-50", n: "system" },
};
const STATE_CHIP = {
  fresh: "bg-green-100 text-green-700", warm: "bg-yellow-100 text-yellow-700",
  cold: "bg-blue-100 text-blue-700", forgotten: "bg-slate-200 text-slate-500 line-through",
};

const S = { prov: {}, sid: null, ws: null, llmio: {}, research: {} };

/* ---------------- setup: multi-provider connect ---------------- */
// S.prov = { gemini: {creds:{api_key}, models:[...]}, openai: {...}, anthropic: {...}, local: {creds:{base_url,api_key}, models:[...]} }
const PROV_LABEL = { gemini: "Gemini", openai: "OpenAI", anthropic: "Anthropic", local: "Local" };

document.querySelectorAll(".connectBtn").forEach((btn) => {
  btn.onclick = async () => {
    const p = btn.dataset.prov;
    const creds = p === "local"
      ? { base_url: $("url-local").value.trim(), api_key: $("key-local").value.trim() }
      : { api_key: $("key-" + p).value.trim() };
    if (p === "local" ? !creds.base_url : !creds.api_key) { $("st-" + p).textContent = p === "local" ? "URL?" : "key?"; return; }
    $("st-" + p).textContent = "…";
    try {
      const r = await fetch("/api/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p, ...creds }) });
      const j = await r.json();
      if (j.error || !j.models.length) { $("st-" + p).textContent = "✗"; $("modelStatus").textContent = `✗ ${PROV_LABEL[p]}: ` + (j.error || "no models"); return; }
      S.prov[p] = { creds, models: j.models };
      $("st-" + p).textContent = `✓ ${j.models.length}`;
      $("modelStatus").textContent = `✓ ${PROV_LABEL[p]} connected — ${j.models.length} models`;
      rebuildRoleSelects();
      $("startBtn").disabled = false;
    } catch (e) { $("st-" + p).textContent = "✗"; $("modelStatus").textContent = "✗ " + e; }
  };
});

// Sensible default per role: a mid-size model for LLM-1, small/cheap for companions.
const ROLE_PREF = {
  main: [/gemini-.*flash(?!-lite)(?!.*8b)/i, /gpt-4o(?!-mini)/i, /claude-sonnet/i, /gpt-4\./i],
  summary: [/flash-lite|flash-8b/i, /gpt-4o-mini|gpt-4\.1-mini|gpt-4\.1-nano/i, /claude-haiku/i, /mini|small|lite/i],
  inference: [/flash-lite|flash-8b/i, /gpt-4o-mini|gpt-4\.1-mini|gpt-4\.1-nano/i, /claude-haiku/i, /mini|small|lite/i],
};
function allModelOptions() {
  // [{value:"prov::id", label, prov}]
  const out = [];
  for (const [p, info] of Object.entries(S.prov)) info.models.forEach((m) => out.push({ value: `${p}::${m.id}`, label: m.id, prov: p }));
  return out;
}
function fillSelect(el, current) {
  el.innerHTML = ""; el.disabled = false;
  for (const [p, info] of Object.entries(S.prov)) {
    const og = document.createElement("optgroup"); og.label = PROV_LABEL[p] || p;
    info.models.forEach((m) => { const o = h("option", "", esc(m.id)); o.value = `${p}::${m.id}`; og.appendChild(o); });
    el.appendChild(og);
  }
  if (current && [...el.options].some((o) => o.value === current)) el.value = current;
}
function pickDefault(role) {
  const opts = allModelOptions();
  for (const re of ROLE_PREF[role] || []) { const hit = opts.find((o) => re.test(o.label)); if (hit) return hit.value; }
  return opts.length ? opts[0].value : "";
}
function rebuildRoleSelects() {
  for (const [sel, role] of [["modelMain", "main"], ["modelSummary", "summary"], ["modelInference", "inference"]]) {
    const el = $(sel);
    const keep = el.value && [...el.options].some((o) => o.value === el.value) ? el.value : null;
    fillSelect(el, keep);
    if (!keep) el.value = pickDefault(role);
  }
}
const parseSpec = (v) => { const i = (v || "").indexOf("::"); return i < 0 ? null : { provider: v.slice(0, i), model: v.slice(i + 2) }; };

$("startBtn").onclick = async () => {
  const models = { main: parseSpec($("modelMain").value), summary: parseSpec($("modelSummary").value), inference: parseSpec($("modelInference").value) };
  if (!models.main) { $("startStatus").textContent = "✗ connect a provider and pick models first"; return; }
  const providers = {};
  for (const [p, info] of Object.entries(S.prov)) providers[p] = info.creds;
  const settings = {
    embedding: "local", background: $("optBackground").checked,
    redact_secrets: $("optRedact").checked,
    search_engine: $("searchEngine").value,
    search_api_key: $("searchKey").value.trim() || null,
    force_companions: $("optForce").checked,
  };
  $("startStatus").textContent = "building agent (first run downloads the embedder)…";
  $("startBtn").disabled = true;
  try {
    const r = await fetch("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ providers, models, system_prompt: $("systemPrompt").value, settings }) });
    const j = await r.json();
    if (j.error) { $("startStatus").textContent = "✗ " + j.error; $("startBtn").disabled = false; return; }
    S.sid = j.session_id;
    mirrorLive();
    connectWS();
    initPanels();
    $("setup").classList.add("hidden");
    $("main").classList.remove("hidden"); $("main").classList.add("flex");
  } catch (e) { $("startStatus").textContent = "✗ " + e; $("startBtn").disabled = false; }
};

function mirrorLive() {
  for (const [sel, src, role] of [["liveMain", "modelMain", "main"], ["liveSummary", "modelSummary", "summary"], ["liveInference", "modelInference", "inference"]]) {
    const el = $(sel);
    fillSelect(el, $(src).value);
    el.onchange = async () => {
      await fetch("/api/select_models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, models: { [role]: parseSpec(el.value) } }) });
    };
  }
}

// Show the API-key field only for keyed search engines.
$("searchEngine").onchange = () => {
  const keyed = ["brave", "tavily", "valyu"].includes($("searchEngine").value);
  $("searchKey").classList.toggle("hidden", !keyed);
};

$("newSession").onclick = () => location.reload();

// Session export: the browser downloads the markdown (Content-Disposition).
$("exportBtn").onclick = () => { if (S.sid) window.open(`/api/export?session_id=${encodeURIComponent(S.sid)}`, "_blank"); };

/* ---------------- websocket ---------------- */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  S.ws = new WebSocket(`${proto}://${location.host}/ws/${S.sid}`);
  S.ws.onopen = () => { $("wsStatus").textContent = "ws: live"; $("wsStatus").className = "ml-auto text-green-400"; };
  S.ws.onclose = () => {
    if (S.closed) { $("wsStatus").textContent = "ws: closed"; $("wsStatus").className = "ml-auto text-red-400"; return; }
    // The server-side queue keeps buffering events while we're gone — just retry.
    $("wsStatus").textContent = "ws: reconnecting…"; $("wsStatus").className = "ml-auto text-amber-400";
    setTimeout(connectWS, 2000);
  };
  S.ws.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
}

/* ---------------- chat ---------------- */
$("send").onclick = sendMsg;
$("msg").addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return; // Enter during IME composition (Korean etc.)
  if (e.key === "Enter") sendMsg();
});
async function sendMsg() {
  const text = $("msg").value.trim(); if (!text || !S.sid) return;
  const mode = $("chatMode") ? $("chatMode").value : "sherlock";
  $("msg").value = "";
  addBubble("user", text);
  // "both": mirror the user message at the top of the MIDDLE column too, so
  // the sherlock + baseline replies align per turn.
  if (mode === "both") addBubble("user", text, $("chatB"));
  setThinking(true, mode === "single" ? "Single LLM is thinking…" : undefined);
  S.llmio = {};
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, message: text, mode, baseline_search: $("baselineSearch") ? $("baselineSearch").checked : true }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { addBubble("system", "✗ " + (j.error || r.status)); setThinking(false); return; }
    if (j.latency_ms != null) {
      // The sherlock bubble itself arrives via the turn.completed WS event
      // (which also stashes its token counts) — here we add the meta line.
      S.lastSherlockLatency = j.latency_ms;
      const t = S.lastTurnTokens || {};
      addMetaLine(`⏱ ${j.latency_ms}ms · ${t.i || 0}/${t.o || 0} tok`);
    }
    if (j.baseline) renderBaseline(j.baseline, mode);
    if (mode === "single") setThinking(false); // no agent turn → no turn.done event
  }
  catch (e) { addBubble("system", "✗ " + e); setThinking(false); }
}
function addBubble(role, text, target) {
  const box = target || $("chat");
  const wrap = h("div", "flex " + (role === "user" ? "justify-end" : "justify-start"));
  const cls = role === "user" ? "bg-blue-600 text-white" : role === "assistant" ? "bg-white border" : "bg-amber-100 text-amber-800 text-xs";
  // Assistant replies render as sanitized markdown; user/system bubbles stay
  // escaped plain text.
  if (role === "assistant") wrap.appendChild(h("div", `max-w-[85%] px-3 py-2 rounded-2xl text-sm prose-md ${cls}`, mdRender(text)));
  else wrap.appendChild(h("div", `max-w-[85%] px-3 py-2 rounded-2xl text-sm whitespace-pre-wrap ${cls}`, esc(text)));
  box.appendChild(wrap); box.scrollTop = box.scrollHeight;
}
function addMetaLine(text, target) {
  const box = target || $("chat");
  const wrap = h("div", "flex justify-start");
  wrap.appendChild(h("div", "text-[10px] text-slate-400 px-3 -mt-2", esc(text)));
  box.appendChild(wrap); box.scrollTop = box.scrollHeight;
}
// A/B baseline: a clearly-labelled bare-model bubble + its own meta line.
// In "both" mode it lives in the MIDDLE column; in "single" mode the middle
// column is hidden, so it stays in the left chat flow.
function renderBaseline(b, mode) {
  const target = mode === "both" ? $("chatB") : $("chat");
  const wrap = h("div", "flex justify-start");
  const box = h("div", "max-w-[85%] px-3 py-2 rounded-2xl text-sm bg-slate-50 border border-slate-300");
  box.appendChild(h("div", "text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-1", "Single LLM"));
  if (b.error) box.appendChild(h("div", "whitespace-pre-wrap text-red-600", esc("✗ " + b.error)));
  else box.appendChild(h("div", "prose-md", mdRender(b.text || "")));
  wrap.appendChild(box);
  target.appendChild(wrap);
  addMetaLine(`⏱ ${b.latency_ms}ms · ${b.prompt_tokens || 0}/${b.completion_tokens || 0} tok`, target);
  BASE.i += b.prompt_tokens || 0; BASE.o += b.completion_tokens || 0;
  renderTokBar();
}
function setThinking(on, text) {
  $("thinking").classList.toggle("hidden", !on);
  $("thinking").classList.toggle("flex", on);
  $("thinkingLabel").textContent = text || "Sherlock is thinking…";
}

/* ---------------- layout: compare column + draggable resizers ---------------- */
// The middle (Single-LLM) column exists ONLY for mode === "both"; hidden, the
// layout collapses to the classic two-column look.
function updateCompareLayout() {
  const bw = $("baselineSearchWrap");
  if (bw) { const m = $("chatMode").value; bw.classList.toggle("hidden", m === "sherlock"); bw.classList.toggle("flex", m !== "sherlock"); }
  const both = $("chatMode").value === "both";
  $("colSingle").classList.toggle("hidden", !both);
  $("colSingle").classList.toggle("flex", both);
  $("rsA").classList.toggle("hidden", !both);
}
$("chatMode").addEventListener("change", updateCompareLayout);
updateCompareLayout();

// Vanilla drag: each 6px handle resizes the nearest VISIBLE column to its
// left (flex-basis in px, min 240px per visible column — including the
// right panel, which keeps flex:1 and absorbs the remainder).
const COL_MIN = 240;
function prevVisibleCol(bar) {
  let el = bar.previousElementSibling;
  while (el && (el.classList.contains("col-resizer") || el.classList.contains("hidden"))) el = el.previousElementSibling;
  return el;
}
document.querySelectorAll(".col-resizer").forEach((bar) => {
  bar.addEventListener("mousedown", (e) => {
    const col = prevVisibleCol(bar);
    if (!col) return;
    e.preventDefault();
    bar.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const startX = e.clientX;
    const startW = col.getBoundingClientRect().width;
    const container = $("cols");
    const onMove = (ev) => {
      const kids = [...container.children].filter((c) => !c.classList.contains("hidden"));
      const others = kids.filter((c) => c !== col && !c.classList.contains("col-resizer")).length;
      const handles = kids.filter((c) => c.classList.contains("col-resizer")).length;
      const max = container.getBoundingClientRect().width - others * COL_MIN - handles * 6;
      const w = Math.min(Math.max(COL_MIN, startW + (ev.clientX - startX)), Math.max(COL_MIN, max));
      col.style.flex = `0 0 ${w}px`;
    };
    const onUp = () => {
      bar.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
});

/* ---------------- event routing ---------------- */
function handleEvent(ev) {
  appendFlow(ev);
  const d = ev.data || {};
  switch (ev.type) {
    case "turn.completed":
      S.lastTurnTokens = { i: d.prompt_tokens || 0, o: d.completion_tokens || 0 };
      if (d.error) addBubble("system", "⚠ " + (d.response_text || "provider error — check the LLM I/O panel"));
      else addBubble("assistant", d.response_text || "");
      break;
    case "slot.assembled": renderSlot(d); break;
    case "llm.call": renderLLMIO(d); countTokens(d); break;
    case "infer.done": renderInference(d); break;
    case "compact.done": renderCompaction(d); break;
    case "memory.snapshot": renderMemory(d.rows || []); break;
    case "decay.done": S.lastDecay = d; break;
    case "carry.snapshot": renderCarry(d); break;
    case "carry.stored": renderCarry(d); break;
    case "tool.start": setThinking(true, `🔧 ${d.kind}: ${trim(d.payload, 40)}…`); break;
    case "tool.done": setThinking(true, d.ok ? `🔧 ${d.kind} done${d.result_count != null ? " · " + d.result_count + " results" : ""}` : `🔧 ${d.kind} failed: ${trim(d.error, 50)}`); break;
    case "background.start": setThinking(true); break;
    case "background.end": case "turn.done": setThinking(false); break;
    case "deep_research.approval_needed": onDRApprovalNeeded(d); break;
    case "deep_research.approved": case "deep_research.start": onDRStart(d); break;
    case "deep_research.plan": S.research.plan_languages = d.languages || []; S.research.plan_queries = d.queries || []; renderResearch(); break;
    case "deep_research.strategy": S.research.strategy = d; renderResearch(); break;
    case "deep_research.clarified": break; // ack bubble arrives via turn.completed
    case "deep_research.tokens": S.research.tokens = d; renderResearch(); break;
    case "deep_research.round": onDRRound(d); break;
    case "deep_research.synthesizing":
      S.research.status = "synthesising"; S.research.stop_reason = d.stop_reason; S.research.rounds_total = d.rounds;
      addBubble("system", `🔬 ${d.rounds} rounds done (${d.stop_reason}) — writing the answer…`);
      setThinking(true, "🔬 synthesising the final answer…"); renderResearch(); break;
    case "deep_research.input_folded": onDRFolded(d); break;
    case "deep_research.queued": addBubble("system", "📨 queued — will fold in at the next research checkpoint"); break;
    case "deep_research.documents": S.research.docs = d.docs || []; S.research.stop_reason = d.stop_reason; S.research.rounds_total = d.rounds; renderResearch(); break;
    case "deep_research.done": onDRDone(d); break;
    case "deep_research.cancelled": hideDRBanner(); S.research.status = "cancelled"; renderResearch(); break;
    case "deep_research.failed":
      S.research.status = "failed";
      addBubble("system", "🔬 ✗ research failed: " + (d.error || "unknown error"));
      setThinking(false); renderResearch(); break;
    case "deep_research.inbox_discarded":
      addBubble("system", `📨 ${d.count} message(s) arrived too late to fold in — they're in the chat, just not in this research`);
      break;
  }
}

/* ---------------- cumulative token bar ---------------- */
const TOK = { main: { i: 0, o: 0, n: 0, c: 0 }, summary: { i: 0, o: 0, n: 0, c: 0 }, inference: { i: 0, o: 0, n: 0, c: 0 } };
const BASE = { i: 0, o: 0 }; // cumulative bare-model (A/B baseline) tokens
const kfmt = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n));
function renderTokBar() {
  const total = TOK.main.i + TOK.main.o + TOK.summary.i + TOK.summary.o + TOK.inference.i + TOK.inference.o;
  const cached = TOK.main.c + TOK.summary.c + TOK.inference.c;
  $("tokBar").textContent =
    `🪙 L1 ${kfmt(TOK.main.i)}/${kfmt(TOK.main.o)} · L2 ${kfmt(TOK.summary.i)}/${kfmt(TOK.summary.o)} · L3 ${kfmt(TOK.inference.i)}/${kfmt(TOK.inference.o)} · Σ ${kfmt(total)}${cached ? ` · ⚡cached ${kfmt(cached)}` : ""}${BASE.i + BASE.o ? ` · single ${kfmt(BASE.i)}/${kfmt(BASE.o)}` : ""}`;
}
function countTokens(d) {
  const t = TOK[d.role]; if (!t) return;
  t.i += d.prompt_tokens || 0; t.o += d.completion_tokens || 0; t.n += 1; t.c += d.cache_read_tokens || 0;
  renderTokBar();
}

/* ---------------- DEEP RESEARCH (v0.7) ---------------- */
function hideDRBanner() { $("drBanner").classList.add("hidden"); }
function onDRApprovalNeeded(d) {
  S.research = { topic: d.topic || "", plan: d.plan || "", status: "awaiting approval", rounds: [], docs: [], folded: [], answer: "" };
  $("drBannerPlan").textContent = d.plan || d.topic || "";
  $("drBanner").classList.remove("hidden");
  renderResearch(); flashTab("research");
}
function onDRStart(d) {
  hideDRBanner();
  if (!S.research.topic) S.research = { topic: d.topic || "", rounds: [], docs: [], folded: [], answer: "" };
  S.research.status = "researching"; S.research.topic = d.topic || S.research.topic;
  if (d.plan) S.research.plan = d.plan;
  setThinking(true, "🔬 deep research running…"); renderResearch(); flashTab("research");
}
function onDRRound(d) {
  (S.research.rounds = S.research.rounds || []).push(d);
  S.research.status = `round ${d.round}`;
  const head = d.key_finding || d.summary || `${d.hits} hits`;
  const newsrc = d.new_sources != null ? `${d.new_sources} new src · ` : "";
  // Per-round 1–2 line summary in the chat so progress is unmissable.
  addBubble("system", `🔬 R${d.round} · ${newsrc}${trim(head, 100)}`);
  setThinking(true, `🔬 round ${d.round}… ${trim(head, 40)}`);
  renderResearch(); flashTab("research");
}
function onDRFolded(d) {
  (S.research.folded = S.research.folded || []).push(d);
  renderResearch();
}
function onDRDone(d) {
  S.research.status = "done"; S.research.answer = d.answer || "";
  addBubble("assistant", d.answer || "");
  setThinking(false); renderResearch(); flashTab("research");
}

/* ---------------- FLOW timeline ---------------- */
const SUMMARY = {
  "turn.start": (d) => `▶ user: ${trim(d.user_text, 80)}`,
  "memory.retrieved": (d) => `retrieved ${d.hits.length} memories` + (d.hits[0] ? ` · top: ${trim(d.hits[0].content, 40)} (${d.hits[0].score})` : ""),
  "slot.assembled": (d) => `slot built · sys ${d.system_tokens} tok · K-turn ${d.k_turn_turns} turns · ${d.retrieved_count} RAG`,
  "llm.call": (d) => `${d.role} (${d.model}) · ${d.total_tokens} tok · ${d.latency_ms}ms` + (d.error ? ` · ✗ ${trim(d.error,40)}` : ` · ${trim(d.response_text, 50)}`),
  "turn.completed": (d) => (d.error ? "⚠ provider error" : `✓ main reply · ${d.tokens_used} tok · companions: ${(d.companions_requested || []).join(",") || "none"}`),
  "tool.start": (d) => `🔧 ${d.kind}: ${trim(d.payload, 50)}`,
  "tool.done": (d) => `🔧 ${d.kind} ${d.ok ? "✓" : "✗"}` + (d.result_count != null ? ` ${d.result_count} results` : "") + (d.error ? ` · ${trim(d.error, 40)}` : ""),
  "background.start": () => "background companions started",
  "baseline.reply": (d) => `⚖ single-LLM baseline · ${d.latency_ms}ms · ${d.prompt_tokens || 0}/${d.completion_tokens || 0} tok` + (d.error ? ` · ✗ ${trim(d.error, 40)}` : ` · ${trim(d.text, 50)}`),
  "sherlock.latency": (d) => `⏱ sherlock turn · ${d.latency_ms}ms`,
  "infer.done": (d) => `LLM-3 · ${(d.hypotheses || []).length} hypotheses` + (d.hypotheses && d.hypotheses[0] ? ` · top: ${trim(d.hypotheses[0].intent, 45)} (${d.hypotheses[0].probability})` : ""),
  "freshness.done": (d) => `freshness searches: ${(d.searches || []).map((s) => s.topic + "(" + s.hits + ")").join(", ")}`,
  "compact.done": (d) => `LLM-2 · summary + ${(d.facts || []).length} facts · ${(d.predicted_directions || []).length} predictions`,
  "decay.done": (d) => `decay · f→w ${d.fresh_to_warm || 0} · w→c ${d.warm_to_cold || 0} · c→forgotten ${d.cold_to_forgotten || 0}`,
  "carry.stored": (d) => `carry-forward · ${(d.hypotheses || []).length} hypotheses → next turn`,
  "background.end": (d) => "background done" + (d.ok === false ? " (error)" : ""),
  "memory.snapshot": (d) => `memory snapshot · ${(d.rows || []).length} entries`,
  "carry.snapshot": (d) => `pending · ${(d.hypotheses || []).length} hypotheses, ${(d.search_results || []).length} search`,
  "deep_research.plan": (d) => `🌐 search plan · langs: ${(d.languages || []).join("/")} · ${(d.queries || []).length} keyword queries`,
  "deep_research.tokens": (d) => `🪙 tokens · ${d.calls} calls · in ${d.in} / out ${d.out}`,
  "deep_research.approval_needed": (d) => `🔬 approval needed · ${trim(d.topic, 50)}`,
  "deep_research.approved": (d) => `🔬 approved · ${trim(d.topic, 50)}`,
  "deep_research.start": (d) => `🔬 research started · ${trim(d.topic, 50)}`,
  "deep_research.round": (d) => `🔬 round ${d.round} · ${d.hits} hits · ${d.new_sources != null ? d.new_sources + " new · " : ""}${d.meta_source} · ${trim(d.key_finding || d.summary, 40)}`,
  "deep_research.synthesizing": (d) => `🔬 synthesising · ${d.rounds} rounds · stop: ${d.stop_reason}`,
  "deep_research.input_folded": (d) => `📨 folded ${d.count} queued message(s)`,
  "deep_research.queued": (d) => `📨 queued mid-research · ${trim(d.text, 45)}`,
  "deep_research.documents": (d) => `📑 ${(d.docs || []).length} research documents`,
  "deep_research.done": (d) => `🔬 research done · ${trim(d.answer, 55)}`,
  "deep_research.cancelled": (d) => `🔬 research cancelled`,
  "deep_research.strategy": (d) => `📋 strategy · ${trim(d.objective, 50)} · ${(d.sub_topics || []).length} sub-topics${(d.clarifying_questions || []).length ? " · ❓" + (d.clarifying_questions || []).length : ""}`,
  "deep_research.clarified": (d) => `📋 clarification folded · ${trim(d.text, 45)}`,
  "deep_research.failed": (d) => `🔬 ✗ research failed · ${trim(d.error, 60)}`,
  "deep_research.inbox_discarded": (d) => `📨 ${d.count} late message(s) not folded (kept in chat)`,
};
const trim = (s, n) => { s = (s || "").replace(/\s+/g, " "); return s.length > n ? s.slice(0, n) + "…" : s; };
let lastTurn = null;
function appendFlow(ev) {
  const a = ACTOR[ev.actor] || ACTOR.system;
  if (ev.turn !== lastTurn && ev.type === "turn.start") {
    lastTurn = ev.turn;
    $("tab-flow").appendChild(h("div", "text-[11px] font-bold text-slate-400 mt-3 mb-1 uppercase tracking-wide", `Turn ${ev.turn}`));
  }
  const card = h("div", "flow-card flex gap-2 mb-1.5");
  card.appendChild(h("div", `w-1 rounded ${a.stripe}`));
  const body = h("div", "flex-1 min-w-0");
  const sumFn = SUMMARY[ev.type];
  const head = h("div", "flex items-baseline gap-2");
  head.appendChild(h("span", `text-[10px] font-bold ${a.text} uppercase`, a.n));
  head.appendChild(h("span", "text-xs text-slate-700 truncate", esc(sumFn ? sumFn(ev.data || {}) : ev.type)));
  body.appendChild(head);
  const det = h("details", "");
  det.appendChild(h("summary", "text-[10px] text-slate-400 hover:text-slate-600", "raw"));
  det.appendChild(h("pre", "mono text-[10px] bg-slate-900 text-slate-100 rounded p-2 mt-1 overflow-x-auto scroll max-h-52", esc(fmt(ev.data))));
  body.appendChild(det);
  card.appendChild(body);
  $("tab-flow").appendChild(card);
  const box = $("tab-flow").parentElement;
  box.scrollTop = box.scrollHeight;
  flashTab(tabForType(ev.type));
}
function tabForType(t) {
  if (t === "slot.assembled") return "slot";
  if (t === "llm.call") return "llmio";
  if (t === "infer.done") return "infer";
  if (t === "compact.done") return "compact";
  if (t.startsWith("memory.")) return "memory";
  if (t.startsWith("carry.")) return "carry";
  if (t.startsWith("deep_research")) return "research";
  return null;
}

/* ---------------- SLOT ---------------- */
function renderSlot(d) {
  const root = $("tab-slot"); root.innerHTML = "";
  root.appendChild(h("div", "text-sm font-bold mb-2", "🧱 Context slot assembled for LLM-1"));
  const stats = h("div", "flex flex-wrap gap-2 mb-3");
  const badge = (t) => h("span", "text-[11px] bg-white border rounded px-2 py-1 mono", t);
  stats.appendChild(badge(`system: ${d.system_tokens} tok`));
  stats.appendChild(badge(`K-turn: ${d.k_turn_turns} turns / ${d.k_turn_tokens} tok`));
  stats.appendChild(badge(`RAG hits: ${d.retrieved_count}`));
  stats.appendChild(badge(`active-intent: ${(d.active_intent || []).length}`));
  stats.appendChild(badge(`search: ${(d.search_block || []).length}`));
  root.appendChild(stats);
  if (d.slot_budget && Object.keys(d.slot_budget).length) {
    const bg = h("details", "mb-2 bg-white border rounded p-2");
    bg.appendChild(h("summary", "text-xs font-semibold text-indigo-700", "token budget (per block)"));
    bg.appendChild(h("pre", "mono text-[10px] mt-1 overflow-x-auto scroll", esc(fmt(d.slot_budget))));
    root.appendChild(bg);
  }
  // TIER-highlighted system prompt
  const pre = h("pre", "mono text-[11px] bg-white border rounded p-3 whitespace-pre-wrap leading-relaxed");
  pre.innerHTML = esc(d.system_prompt || "").replace(/(TIER\s*\d[^\n]*)/g, '<span class="font-bold text-indigo-600">$1</span>');
  root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-2 mb-1", "Assembled system prompt"));
  root.appendChild(pre);
  if ((d.tail || []).length) {
    const t = h("details", "mt-2 bg-white border rounded p-2");
    t.appendChild(h("summary", "text-xs font-semibold", `K-turn tail (${d.tail.length} messages)`));
    d.tail.forEach((m) => t.appendChild(h("div", "text-[11px] border-l-2 pl-2 my-1 " + (m.role === "user" ? "border-blue-300" : "border-slate-300"), `<b>${esc(m.role)}:</b> ${esc(trim(m.content, 200))}`)));
    root.appendChild(t);
  }
}

/* ---------------- LLM I/O ---------------- */
function renderLLMIO(d) {
  // Keep EVERY call this turn (tool rounds, deep-research internals), newest last.
  (S.llmio[d.role] = S.llmio[d.role] || []).push(d);
  const root = $("tab-llmio"); root.innerHTML = "";
  for (const role of ["main", "summary", "inference"]) {
    const calls = S.llmio[role] || [];
    const v = calls.length ? calls[calls.length - 1] : null;
    const actor = role === "main" ? ACTOR.llm1 : role === "summary" ? ACTOR.llm2 : ACTOR.llm3;
    const card = h("div", `border rounded-lg overflow-hidden ${v ? "" : "opacity-40"}`);
    card.appendChild(h("div", `px-3 py-1.5 text-xs font-bold text-white ${actor.stripe}`, `${actor.n} · ${role}` + (v ? ` · ${esc(v.model)} · ${v.total_tokens} tok · ${v.latency_ms}ms` : " · (not called this turn)")));
    if (v) {
      const b = h("div", "p-2 space-y-1 bg-white");
      if (v.error) b.appendChild(h("div", "text-xs text-red-600", esc(v.error)));
      const sp = h("details", ""); sp.appendChild(h("summary", "text-[11px] text-slate-500", "system prompt")); sp.appendChild(h("pre", "mono text-[10px] bg-slate-50 rounded p-2 mt-1 max-h-40 overflow-auto scroll whitespace-pre-wrap", esc(v.system_prompt))); b.appendChild(sp);
      const mm = h("details", ""); mm.appendChild(h("summary", "text-[11px] text-slate-500", `messages (${(v.messages || []).length})`)); mm.appendChild(h("pre", "mono text-[10px] bg-slate-50 rounded p-2 mt-1 max-h-40 overflow-auto scroll", esc(fmt(v.messages)))); b.appendChild(mm);
      b.appendChild(h("div", "text-[11px] font-semibold text-slate-500 mt-1", "response"));
      b.appendChild(h("div", "text-xs whitespace-pre-wrap bg-slate-50 rounded p-2 mono", esc(v.response_text)));
      if (calls.length > 1) {
        const prev = calls.slice(0, -1);
        const pd = h("details", "mt-1 border-t pt-1");
        pd.appendChild(h("summary", "text-[11px] text-slate-500", `previous calls this turn (${prev.length})`));
        prev.forEach((c, i) => {
          const row = h("div", "border rounded p-1.5 mt-1 bg-slate-50");
          row.appendChild(h("div", "text-[11px] text-slate-600 mono", `#${i + 1} · ${esc(c.model)} · ${c.total_tokens} tok · ${c.latency_ms}ms`));
          if (c.error) row.appendChild(h("div", "text-[11px] text-red-600", esc(c.error)));
          row.appendChild(h("div", "text-[11px] text-slate-700 whitespace-pre-wrap mono mt-0.5", esc(trim(c.response_text, 300))));
          const cm = h("details", ""); cm.appendChild(h("summary", "text-[10px] text-slate-400", `messages (${(c.messages || []).length})`)); cm.appendChild(h("pre", "mono text-[10px] bg-white rounded p-2 mt-1 max-h-40 overflow-auto scroll", esc(fmt(c.messages)))); row.appendChild(cm);
          pd.appendChild(row);
        });
        b.appendChild(pd);
      }
      card.appendChild(b);
    }
    root.appendChild(card);
  }
}

/* ---------------- INFERENCE (LLM-3) ---------------- */
function renderInference(d) {
  const root = $("tab-infer"); root.innerHTML = "";
  root.appendChild(h("div", "text-sm font-bold mb-1 text-purple-700", "🧠 LLM-3 inference"));
  if (d.really_asking) {
    const ra = h("div", "text-[11px] bg-purple-50 border border-purple-200 rounded px-2 py-1 mb-1");
    let inner = `🎯 really asking: <b>${esc(d.really_asking)}</b>`;
    if ((d.implied_chain || []).length) inner += `<br>chain: ${(d.implied_chain || []).map(esc).join(" → ")}`;
    (d.anticipated_next || []).forEach((nx) => { if (nx && nx.question) inner += `<br>next: ${esc(nx.question)}${nx.answer_hint ? " — " + esc(nx.answer_hint) : ""}`; });
    ra.innerHTML = inner;
    root.appendChild(ra);
  }
  root.appendChild(h("div", "text-xs text-slate-500 mb-2", `overall confidence ${d.confidence_overall ?? "?"} · tools: ${(d.tools_recommended || []).join(", ") || "none"} · freshness: ${(d.freshness_required || []).join(", ") || "none"}`));
  (d.hypotheses || []).forEach((hyp, i) => {
    const p = Math.round((hyp.probability || 0) * 100);
    const card = h("div", "border rounded-lg p-2 mb-2 bg-white");
    const top = h("div", "flex items-center gap-2");
    top.appendChild(h("span", "text-xs font-bold text-slate-400", "#" + (i + 1)));
    top.appendChild(h("span", "text-sm font-semibold flex-1", esc(hyp.intent)));
    top.appendChild(h("span", "text-[10px] px-2 py-0.5 rounded-full bg-purple-100 text-purple-700", esc(hyp.reasoning_type || "")));
    card.appendChild(top);
    const bar = h("div", "h-2 bg-slate-100 rounded mt-1 overflow-hidden");
    bar.appendChild(h("div", "h-full bg-purple-500", "")).style.width = p + "%";
    card.appendChild(bar);
    card.appendChild(h("div", "text-[11px] text-slate-500 mt-1", `${p}% · evidence: ${esc((hyp.evidence || []).join("; ") || "—")}`));
    if ((hyp.search_keywords || []).length) card.appendChild(h("div", "text-[10px] text-slate-400 mt-0.5", "search: " + esc(hyp.search_keywords.join(", "))));
    root.appendChild(card);
  });
}

/* ---------------- COMPACTION (LLM-2) ---------------- */
function renderCompaction(d) {
  const root = $("tab-compact"); root.innerHTML = "";
  root.appendChild(h("div", "text-sm font-bold mb-1 text-green-700", "🗜 LLM-2 compaction"));
  root.appendChild(h("div", "text-xs bg-white border rounded p-2 mb-2", `<b>summary:</b> ${esc(d.summary || "")} <span class="text-slate-400">· topic: ${esc(d.topic_label || "")}</span>`));
  if (d.persona_summary) root.appendChild(h("div", "text-xs bg-green-50 border border-green-200 rounded p-2 mb-2", `<b>persona:</b> ${esc(d.persona_summary)}`));
  if ((d.facts || []).length) {
    root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-1 mb-1", `facts (${d.facts.length})`));
    const tbl = h("div", "border rounded overflow-hidden bg-white");
    d.facts.forEach((f) => {
      const row = h("div", "flex items-center gap-2 px-2 py-1 border-b text-[11px]");
      row.appendChild(h("span", "flex-1", esc(f.content)));
      row.appendChild(h("span", "text-slate-400", esc(f.source || "")));
      row.appendChild(h("span", "text-slate-400", (f.confidence ?? "") + ""));
      if (f.pin_recommended) row.appendChild(h("span", "text-amber-600", "📌"));
      tbl.appendChild(row);
    });
    root.appendChild(tbl);
  }
  if ((d.predicted_directions || []).length) {
    root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-2 mb-1", `predictions (${d.predicted_directions.length})`));
    d.predicted_directions.forEach((p) => root.appendChild(h("div", "text-[11px] bg-white border rounded px-2 py-1 mb-1", `${esc(p.direction)} <span class="text-slate-400">(${p.confidence})</span>`)));
  }
}

/* ---------------- MEMORY ---------------- */
function renderMemory(rows) {
  const root = $("tab-memory"); root.innerHTML = "";
  const head = h("div", "flex items-center gap-2 mb-2");
  head.appendChild(h("div", "text-sm font-bold text-amber-700", `🗃 Memory (${rows.length})`));
  if (S.lastDecay) head.appendChild(h("span", "text-[11px] text-slate-500", `last decay → warm ${S.lastDecay.fresh_to_warm || 0}, cold ${S.lastDecay.warm_to_cold || 0}, forgotten ${S.lastDecay.cold_to_forgotten || 0}`));
  root.appendChild(head);
  const counts = {};
  rows.forEach((r) => (counts[r.state] = (counts[r.state] || 0) + 1));
  const legend = h("div", "flex gap-1 mb-2");
  Object.keys(STATE_CHIP).forEach((st) => legend.appendChild(h("span", `text-[10px] px-2 py-0.5 rounded-full ${STATE_CHIP[st]}`, `${st} ${counts[st] || 0}`)));
  root.appendChild(legend);
  const tbl = h("div", "border rounded-lg overflow-hidden bg-white");
  rows.forEach((r) => {
    const row = h("div", "flex items-center gap-2 px-2 py-1.5 border-b text-[11px]");
    row.appendChild(h("span", `px-1.5 py-0.5 rounded text-[9px] font-bold ${STATE_CHIP[r.state] || ""}`, r.state));
    if (r.pinned) row.appendChild(h("span", "", "📌"));
    const c = h("span", "flex-1 truncate", esc(r.content)); c.title = r.content; row.appendChild(c);
    row.appendChild(h("span", "text-slate-400 mono", esc(r.type)));
    row.appendChild(h("span", "text-slate-400 mono", esc(r.source)));
    row.appendChild(h("span", "text-slate-400", `c${r.confidence}`));
    row.appendChild(h("span", "text-slate-300", `×${r.use_count}`));
    tbl.appendChild(row);
  });
  root.appendChild(tbl);
}

/* ---------------- CARRY-FORWARD ---------------- */
function renderCarry(d) {
  const root = $("tab-carry"); root.innerHTML = "";
  root.appendChild(h("div", "text-sm font-bold mb-1 text-rose-700", "↪ Carry-forward → next turn's slot"));
  root.appendChild(h("div", "text-xs text-slate-500 mb-2", "LLM-3's hypotheses + freshness from this turn seed the NEXT turn's TIER-3 (active-intent) block — this is how the loop closes."));
  const hyp = d.hypotheses || [];
  if (!hyp.length) root.appendChild(h("div", "text-xs text-slate-400 italic", "no pending hypotheses (none requested this turn)"));
  hyp.forEach((x) => root.appendChild(h("div", "text-xs bg-rose-50 border border-rose-200 rounded px-2 py-1 mb-1", `${esc(x.intent || x.direction || JSON.stringify(x))} <span class="text-slate-400">(${x.probability ?? x.confidence ?? ""})</span>`)));
  const sr = d.search_results || [];
  if (sr.length) { root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-2 mb-1", `freshness results (${sr.length})`)); sr.forEach((s) => root.appendChild(h("div", "text-[11px] bg-white border rounded px-2 py-1 mb-1", esc(trim(s.title || s.content || JSON.stringify(s), 120))))); }
}

/* ---------------- DEEP RESEARCH render ---------------- */
function renderResearch() {
  const root = $("tab-research"); root.innerHTML = "";
  const r = S.research || {};
  root.appendChild(h("div", "text-sm font-bold mb-1 text-indigo-700", "🔬 Deep research"));
  if (!r.topic) {
    root.appendChild(h("div", "text-xs text-slate-400 italic", "No research yet. LLM-1 proposes deep research when a question needs depth/breadth a few searches can't give — approve it and the rounds stream in here, each saved as a session document."));
    return;
  }
  const stopTxt = r.stop_reason ? ` · stopped: ${esc(r.stop_reason)}` : "";
  root.appendChild(h("div", "text-xs text-slate-600 mb-1", `topic: <b>${esc(r.topic)}</b> · status: <b>${esc(r.status || "")}</b>${r.rounds_total ? " · " + r.rounds_total + " rounds" : ""}${stopTxt}`));
  if (r.plan) root.appendChild(h("div", "text-[11px] text-slate-400 mb-1", esc(r.plan)));
  if (r.strategy && (r.strategy.objective || (r.strategy.sub_topics || []).length)) {
    const st = h("div", "text-[11px] bg-indigo-50 border border-indigo-200 rounded px-2 py-1 mb-1");
    let inner = `📋 <b>${esc(r.strategy.objective || "strategy")}</b>`;
    if ((r.strategy.sub_topics || []).length) inner += `<br>cover: ${(r.strategy.sub_topics || []).map(esc).join(" · ")}`;
    if ((r.strategy.clarifying_questions || []).length) inner += `<br>❓ ${(r.strategy.clarifying_questions || []).map(esc).join(" / ")}`;
    st.innerHTML = inner;
    root.appendChild(st);
  }
  // v0.8: multilingual search plan
  if ((r.plan_languages || []).length) {
    const pl = h("div", "text-[11px] mb-1");
    pl.innerHTML = `🌐 search languages: <b>${(r.plan_languages || []).map(esc).join(", ")}</b>`;
    root.appendChild(pl);
    if ((r.plan_queries || []).length) root.appendChild(h("div", "text-[10px] text-slate-400 mb-1", "keywords: " + esc((r.plan_queries || []).join(" · "))));
  }
  // v0.8: live token usage
  if (r.tokens && r.tokens.calls) {
    const t = r.tokens;
    const by = Object.entries(t.by_stage || {}).map(([k, v]) => `${k} ${v.in}/${v.out}`).join(" · ");
    root.appendChild(h("div", "text-[11px] bg-slate-50 border rounded px-2 py-1 mb-2 mono", `🪙 ${t.calls} calls · in ${t.in} / out ${t.out} tok${by ? " — " + esc(by) : ""}`));
  }
  (r.rounds || []).forEach((rd) => {
    const card = h("div", "border rounded-lg p-2 mb-1.5 bg-white");
    const top = h("div", "flex items-center gap-2");
    top.appendChild(h("span", "text-xs font-bold text-indigo-400", "R" + rd.round));
    top.appendChild(h("span", "text-xs font-semibold flex-1", esc(rd.key_finding || rd.summary || "")));
    const llm3 = rd.meta_source === "llm3-generated";
    top.appendChild(h("span", "text-[10px] px-2 py-0.5 rounded-full " + (llm3 ? "bg-purple-100 text-purple-700" : "bg-slate-100 text-slate-600"), llm3 ? "LLM-3 Qs" : "LLM-1 Qs"));
    if (rd.sufficient) top.appendChild(h("span", "text-[10px] text-green-600", "✓ enough"));
    card.appendChild(top);
    const ns = rd.new_sources != null ? ` · ${rd.new_sources} new src` : "";
    const nf = rd.new_fragments != null ? ` · ${rd.new_fragments} new frags` : "";
    const ft = rd.facts_total != null ? ` · ${rd.facts_total} facts so far` : "";
    const bl = rd.backlog ? ` · ${rd.backlog} queued frags` : "";
    const se = rd.search_errors ? ` · ⚠ ${rd.search_errors} search errors` : "";
    card.appendChild(h("div", "text-[11px] text-slate-500 mt-1", `${rd.hits} hits${ns}${nf} · ${rd.fetched || 0} pages${ft}${bl}${se}`));
    card.appendChild(h("div", "text-[11px] text-slate-500 mt-0.5", `🔎 queries: ${esc((rd.queries || []).join(", "))}`));
    if ((rd.meta_questions || []).length) {
      const mq = h("details", "mt-0.5");
      mq.appendChild(h("summary", "text-[10px] " + (llm3 ? "text-purple-600" : "text-indigo-500"), `${llm3 ? "LLM-3" : "LLM-1"} questions (${rd.meta_questions.length})`));
      rd.meta_questions.forEach((q) => mq.appendChild(h("div", "text-[10px] text-slate-500 pl-2", "• " + esc(q))));
      card.appendChild(mq);
    }
    if (rd.answers) {
      const ad = h("details", "mt-0.5");
      ad.appendChild(h("summary", "text-[10px] text-slate-500", "answer"));
      ad.appendChild(h("div", "text-[11px] text-slate-700 whitespace-pre-wrap pl-2 mt-0.5", esc(rd.answers)));
      card.appendChild(ad);
    }
    if (rd.summary && rd.summary !== rd.key_finding) card.appendChild(h("div", "text-[11px] text-slate-600 mt-0.5", esc(rd.summary)));
    root.appendChild(card);
  });
  (r.folded || []).forEach((f) => root.appendChild(h("div", "text-[11px] bg-amber-50 border border-amber-200 rounded px-2 py-1 mb-1", "📨 folded your input: " + esc((f.texts || []).join("; ")))));
  if (r.answer) {
    root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-2 mb-1", "Synthesis (read from the documents, not raw context)"));
    root.appendChild(h("div", "text-xs bg-indigo-50 border border-indigo-200 rounded p-2 whitespace-pre-wrap", esc(r.answer)));
  }
  const docs = r.docs || [];
  if (docs.length) {
    root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-3 mb-1", `📑 Session documents (${docs.length})`));
    docs.forEach((doc) => {
      const det = h("details", "bg-white border rounded mb-1");
      det.appendChild(h("summary", "text-[11px] px-2 py-1 font-semibold " + (doc.final ? "text-indigo-700" : "text-slate-600"), (doc.final ? "FINAL synthesis" : "Round " + doc.round) + " — " + esc(trim(doc.key_finding || doc.summary || "", 60))));
      const b = h("div", "px-2 py-1.5 text-[11px] space-y-1");
      if ((doc.queries || []).length) b.appendChild(h("div", "text-slate-500", "queries: " + esc(doc.queries.join(", "))));
      if (doc.meta_source) b.appendChild(h("div", "text-slate-400", "meta-question source: " + esc(doc.meta_source)));
      if ((doc.meta_questions || []).length) b.appendChild(h("div", "text-slate-400", "Qs: " + esc(doc.meta_questions.join(" | "))));
      (doc.facts || []).forEach((f) => b.appendChild(h("div", "text-slate-700", "• " + esc(typeof f === "string" ? f : (f.fact || "")))));
      if ((doc.gaps || []).length) b.appendChild(h("div", "text-amber-600", "gaps: " + esc(doc.gaps.join("; "))));
      if (doc.answers && !(doc.facts || []).length) b.appendChild(h("div", "text-slate-700 whitespace-pre-wrap", esc(doc.answers)));
      if ((doc.sources || []).length) { const s = h("div", "text-slate-500", "sources: "); doc.sources.forEach((src) => s.appendChild(h("span", "text-blue-600 mr-1", "[" + esc(trim(src.title || src.url, 40)) + "]"))); b.appendChild(s); }
      det.appendChild(b);
      root.appendChild(det);
    });
  }
}
$("drApprove").onclick = async () => {
  hideDRBanner(); setThinking(true, "🔬 deep research starting…");
  try {
    const r = await fetch("/api/deep_research/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { addBubble("system", "✗ " + (j.error || r.status)); setThinking(false); }
  }
  catch (e) { addBubble("system", "✗ " + e); setThinking(false); }
};
$("drSkip").onclick = async () => {
  hideDRBanner();
  try {
    const r = await fetch("/api/deep_research/skip", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) addBubble("system", "✗ " + (j.error || r.status));
  }
  catch (e) { addBubble("system", "✗ " + e); }
  S.research.status = "skipped"; renderResearch();
};

/* ---------------- tabs ---------------- */
const TABS = ["flow", "slot", "llmio", "infer", "compact", "memory", "carry", "research"];
document.querySelectorAll(".tabbtn").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  TABS.forEach((x) => $("tab-" + x).classList.toggle("hidden", x !== t));
  document.querySelectorAll(".tabbtn").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
}
let flashTimers = {};
function flashTab(tab) {
  if (!tab) return;
  const btn = document.querySelector(`.tabbtn[data-tab="${tab}"]`);
  if (!btn || btn.classList.contains("active")) return;
  btn.classList.add("ring-2", "ring-offset-1", "ring-amber-400");
  clearTimeout(flashTimers[tab]);
  flashTimers[tab] = setTimeout(() => btn.classList.remove("ring-2", "ring-offset-1", "ring-amber-400"), 900);
}

/* example chips → fill the input */
document.querySelectorAll(".ex-chip").forEach((b) => (b.onclick = () => { $("msg").value = b.textContent; $("msg").focus(); }));

/* empty-state hints so panels aren't blank before the first turn */
function initPanels() {
  const hint = (id, txt) => ($(id).innerHTML = `<div class="text-xs text-slate-400 italic p-6 text-center leading-relaxed">${txt}</div>`);
  hint("tab-infer", "🧠 <b>LLM-3 inference</b> appears here after you send a message.<br>Always-run reasoning is on — <i>any</i> message triggers it.");
  hint("tab-compact", "🗜 <b>LLM-2 compaction</b> (summary · facts · persona · predictions) appears here.");
  hint("tab-memory", "🗃 The <b>memory table</b> fills as facts are stored — with provenance + decay-state chips.");
  hint("tab-carry", "↪ <b>Pending hypotheses</b> that seed the next turn's slot appear here.");
  hint("tab-research", "🔬 <b>Deep research</b> — when LLM-1 proposes it and you approve, each round (search → read → meta-question Q&amp;A) streams here and is saved as a session document.");
  hint("tab-slot", "🧱 The assembled <b>LLM-1 context</b> (TIER 1–4) + token budget appears here each turn.");
  hint("tab-llmio", "💬 The exact <b>prompts + responses</b> for LLM-1 / LLM-2 / LLM-3 appear here.");
}
