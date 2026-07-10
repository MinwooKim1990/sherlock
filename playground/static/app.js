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

/* ---------------- i18n: UI language (LLM output is untouched) ---------------- */
// English is the source + fallback. The other languages are filled in below.
const LANG = {
  en: {
    subtitle: "Bring any LLM — Gemini, OpenAI, Anthropic, an open-source-model host (DeepInfra · Together · OpenRouter), or a local OpenAI-compatible server — and watch Sherlock curate its context in real time.",
    providers: "Providers", providers_hint: "— connect at least one; mix freely per role",
    role_main: "Main", role_summary: "Summarizer", role_infer: "Inferencer",
    sysprompt_label: "System prompt (LLM-1 persona)",
    companions: "Companions",
    companions_turbo: "turbo — LLM-2 + LLM-3 every turn",
    companions_cold: "cold_start — only when a signal needs them",
    companions_off: "off — single model (legacy)",
    companions_help: "💡 <b>turbo</b> runs LLM-3 inference + LLM-2 compaction every turn (the 🧠 Inference / 🗜 Compaction panels always fill). <b>cold_start</b> keeps it single-model until a real signal needs the companions; <b>off</b> is the legacy gate.",
    background: "Background", redact: "Redact secrets", verify: "Verify", websearch: "🌐 Web search",
    engine_ddg: "DuckDuckGo (free, no key)", engine_brave: "Brave (key)",
    engine_tavily: "Tavily (key)", engine_valyu: "Valyu (key)", engine_off: "Off",
    searchkey_ph: "search API key",
    websearch_help: "LLM-1 (search + fetch) and LLM-3 (freshness) both use this engine. DuckDuckGo is free but weak for news; Brave/Tavily/Valyu give far better results with a key.",
    start_session: "Start session", export: "⬇ export", new_session: "new session",
    thinking: "Sherlock is thinking…", stop: "Stop", try: "try:",
    ex1: "My daughter Yujin has a peanut allergy",
    ex2: "I'm torn about whether to quit my job",
    ex3: "I'm going to Tokyo next week — recommend a hotel",
    ex4: "Search today's weather in Seoul",
    ex5: "Do you remember what I said earlier?",
    dr_proposed: "🔬 Deep research proposed", approve: "Approve", skip: "Skip",
    dr_or_yes: "…or just reply “yes”.",
    chatmode_title: "who answers: Sherlock, the bare model, or both side-by-side",
    mode_single: "Single LLM", mode_both: "Compare (A/B)",
    baseline_search_title: "Fair comparison: the Single LLM also gets one search pass with the same engine",
    single_search: "🔎single", msg_ph: "Say something…", send: "Send",
    single_baseline: "⚖ Single LLM (baseline)",
    tab_flow: "⚡ Flow", tab_slot: "🧱 Slot", tab_llmio: "💬 LLM I/O", tab_infer: "🧠 Inference",
    tab_compact: "🗜 Compaction", tab_memory: "🗃 Memory", tab_carry: "↪ Carry", tab_research: "🔬 Research",
    // dynamic (JS) strings
    thinking_single: "Single LLM is thinking…", stopping: "stopping…",
    building_agent: "building agent (first run downloads the embedder)…",
    need_provider: "✗ connect a provider and pick models first",
    dr_running: "🔬 deep research running…", dr_synth: "🔬 synthesising the final answer…",
    dr_starting: "🔬 deep research starting…",
    bg_running: "🧩 updating memory in the background…",
    bg_on: "async", bg_off: "inline",
    async_title: "async — LLM-2/LLM-3 run in the background so the reply never waits (change live)",
    // v1.12 A5: long-term memory
    longterm: "Long-term memory", ltm_profile: "profile", ltm_profile_ph: "default",
    ltm_incognito: "incognito — read, don't save",
    longterm_help: "🧠 Long-term memory keeps durable facts (name, allergies, “remember this”) under the named <b>profile</b> across sessions and restarts. Off by default. <b>Incognito</b> keeps recalling what's stored but pauses new writes.",
    tab_ltm: "🧠 Long-term", tab_ltm_full: "Long-term memory",
    ltm_on: "LTM on", ltm_off: "LTM off", incog_off: "rec", incog_on: "incognito",
    ltm_title: "long-term (cross-session) memory — change live",
    incog_title: "incognito — recall but pause new writes",
    ltm_refresh: "Refresh", ltm_export: "Export", ltm_import: "Import", ltm_wipe: "Wipe",
    ltm_empty: "No long-term memory yet. When long-term memory is on, LLM-2 promotes durable facts (identity, allergies, explicit “remember this”, stable preferences/projects) here — shared across every session.",
    ltm_col_cat: "category", ltm_col_content: "fact", ltm_col_conf: "conf", ltm_col_created: "since", ltm_delete: "delete",
    ltm_delete_confirm: "Delete this long-term memory permanently?",
    ltm_wipe_confirm1: "Wipe ALL long-term memory? A markdown backup is written first.",
    ltm_wipe_confirm2: "Are you sure? This clears every durable fact for this profile.",
    ltm_remembered: "remembered", ltm_forgot: "forgot", ltm_wiped: "long-term wiped",
    ltm_imported: "imported to long-term", ltm_import_fail: "import failed",
    ltm_temp_note: "🧠 note: this session stores to a temporary dir — start a new session to persist under a profile",
    // v1.12 B4: LLM-4 inline visualizer
    viz: "📊 Visualizations", viz_on: "viz on", viz_off: "viz off",
    viz_title: "inline data visualizations (LLM-4) — render charts the model proposes; change live",
    viz_role: "Visualizer", viz_same_as_main: "— same as Main —",
    viz_help: "📊 When on, LLM-1 (and deep research) may drop an inline chart where a visual genuinely helps; LLM-4 renders it into a self-contained, sandboxed artifact. Off by default.",
    viz_loading: "📊 rendering visualization…",
    viz_repairing: "📊 repairing visualization… (round {0})",
    viz_repairing_runtime: "📊 repairing visualization…",
    viz_unavailable: "📊 (visualization unavailable)",
    // v1.12 P1: optional text→image model + mid-session per-role model panel
    viz_image_model: "image model",
    viz_image_help: "<code>image:</code> markers render a generated picture — needs an image-capable model + that provider's key",
    models_btn: "models", models_title: "Per-role models (applies next turn)",
    // v1.12 H2: history sidebar
    hist_title: "history", hist_new: "+ new",
    hist_empty: "no conversations yet", hist_rename: "rename",
    hist_open_fail: "couldn't open the conversation",
  },
  ko: {"subtitle":"아무 LLM이나 연결하세요 — Gemini, OpenAI, Anthropic, 오픈소스 모델 호스트(DeepInfra · Together · OpenRouter), 또는 로컬 OpenAI 호환 서버까지 — 그리고 Sherlock이 그 컨텍스트를 실시간으로 정리하는 모습을 지켜보세요.","providers":"제공자","providers_hint":"— 최소 하나는 연결하세요. 역할별로 자유롭게 조합 가능","role_main":"메인","role_summary":"요약기","role_infer":"추론기","sysprompt_label":"시스템 프롬프트 (LLM-1 페르소나)","companions":"컴패니언","companions_turbo":"turbo — 매 턴마다 LLM-2 + LLM-3","companions_cold":"cold_start — 신호가 필요할 때만","companions_off":"off — 단일 모델 (레거시)","companions_help":"💡 <b>turbo</b>는 매 턴마다 LLM-3 추론 + LLM-2 압축을 실행합니다(🧠 추론 / 🗜 압축 패널이 항상 채워짐). <b>cold_start</b>는 실제 신호로 컴패니언이 필요해질 때까지 단일 모델을 유지하고, <b>off</b>는 레거시 게이트입니다.","background":"백그라운드","redact":"비밀정보 가리기","verify":"검증","websearch":"🌐 웹 검색","engine_ddg":"DuckDuckGo (무료, 키 불필요)","engine_brave":"Brave (키 필요)","engine_tavily":"Tavily (키 필요)","engine_valyu":"Valyu (키 필요)","engine_off":"끔","searchkey_ph":"검색 API 키","websearch_help":"LLM-1(검색 + 가져오기)과 LLM-3(최신성)이 모두 이 엔진을 사용합니다. DuckDuckGo는 무료지만 뉴스에는 약하고, Brave/Tavily/Valyu는 키만 있으면 훨씬 나은 결과를 줍니다.","start_session":"세션 시작","export":"⬇ 내보내기","new_session":"새 세션","thinking":"Sherlock이 생각 중…","stop":"중지","try":"예시:","ex1":"우리 딸 유진이가 땅콩 알레르기가 있어요","ex2":"회사를 그만둘지 말지 고민이에요","ex3":"다음 주에 부산 가는데 호텔 추천해줘","ex4":"오늘 서울 날씨 검색해줘","ex5":"내가 아까 한 말 기억해?","dr_proposed":"🔬 심층 리서치 제안됨","approve":"승인","skip":"건너뛰기","dr_or_yes":"…또는 그냥 “네”라고 답하세요.","chatmode_title":"누가 답하는지: Sherlock, 순수 모델, 또는 둘을 나란히","mode_single":"단일 LLM","mode_both":"비교 (A/B)","baseline_search_title":"공정한 비교: 단일 LLM도 동일한 엔진으로 한 번 검색을 수행합니다","single_search":"🔎단일","msg_ph":"메시지를 입력하세요…","send":"전송","single_baseline":"⚖ 단일 LLM (기준)","tab_flow":"⚡ 흐름","tab_slot":"🧱 슬롯","tab_llmio":"💬 LLM 입출력","tab_infer":"🧠 추론","tab_compact":"🗜 압축","tab_memory":"🗃 메모리","tab_carry":"↪ 이월","tab_research":"🔬 리서치","thinking_single":"단일 LLM이 생각 중…","stopping":"중지하는 중…","building_agent":"에이전트 구성 중 (첫 실행 시 임베더를 다운로드합니다)…","need_provider":"✗ 먼저 제공자를 연결하고 모델을 선택하세요","dr_running":"🔬 심층 리서치 실행 중…","dr_synth":"🔬 최종 답변 종합 중…","dr_starting":"🔬 심층 리서치 시작 중…","bg_running":"🧩 백그라운드에서 메모리 갱신 중…","bg_on":"비동기","bg_off":"동기","async_title":"비동기 — LLM-2/LLM-3가 백그라운드에서 돌아 답변이 기다리지 않습니다 (실시간 변경 가능)","longterm":"장기 기억","ltm_profile":"프로필","ltm_profile_ph":"default","ltm_incognito":"시크릿 — 읽되 저장 안 함","longterm_help":"🧠 장기 기억은 이름·알레르기·“기억해 둬” 같은 지속적 사실을 지정한 <b>프로필</b>에 저장해 세션과 재시작을 넘어 유지합니다. 기본값은 꺼짐. <b>시크릿</b>은 저장된 내용은 계속 불러오되 새 쓰기는 멈춥니다.","tab_ltm":"🧠 장기기억","tab_ltm_full":"장기 기억","ltm_on":"장기 켬","ltm_off":"장기 끔","incog_off":"기록","incog_on":"시크릿","ltm_title":"장기(세션 간) 기억 — 실시간 변경","incog_title":"시크릿 — 회상은 하되 새 쓰기는 멈춤","ltm_refresh":"새로고침","ltm_export":"내보내기","ltm_import":"가져오기","ltm_wipe":"전체삭제","ltm_empty":"아직 장기 기억이 없습니다. 장기 기억이 켜지면 LLM-2가 지속적 사실(정체성·알레르기·명시적 “기억해 둬”·안정적 선호/프로젝트)을 여기로 승격하며, 모든 세션에서 공유됩니다.","ltm_col_cat":"분류","ltm_col_content":"사실","ltm_col_conf":"신뢰","ltm_col_created":"기록시점","ltm_delete":"삭제","ltm_delete_confirm":"이 장기 기억을 영구 삭제할까요?","ltm_wipe_confirm1":"모든 장기 기억을 삭제할까요? 먼저 마크다운 백업이 기록됩니다.","ltm_wipe_confirm2":"정말요? 이 프로필의 모든 지속적 사실이 지워집니다.","ltm_remembered":"기억됨","ltm_forgot":"잊음","ltm_wiped":"장기 기억 삭제됨","ltm_imported":"장기 기억으로 가져옴","ltm_import_fail":"가져오기 실패","ltm_temp_note":"🧠 참고: 이 세션은 임시 디렉터리에 저장됩니다 — 프로필에 영구 저장하려면 새 세션을 시작하세요","viz":"📊 시각화","viz_on":"시각화 켬","viz_off":"시각화 끔","viz_title":"인라인 데이터 시각화 (LLM-4) — 모델이 제안하는 차트를 렌더링합니다. 실시간 변경 가능","viz_role":"시각화기","viz_same_as_main":"— 메인과 동일 —","viz_help":"📊 켜면 LLM-1(및 심층 리서치)이 시각화가 정말 도움이 되는 자리에 인라인 차트를 넣고, LLM-4가 이를 독립적인 샌드박스 아티팩트로 렌더링합니다. 기본값은 꺼짐.","viz_loading":"📊 시각화 렌더링 중…","viz_repairing":"📊 시각화 복구 중… (라운드 {0})","viz_repairing_runtime":"📊 시각화 복구 중…","viz_unavailable":"📊 (시각화를 사용할 수 없음)","viz_image_model":"이미지 모델","viz_image_help":"<code>image:</code> 마커가 생성 이미지를 렌더링합니다. 이미지 지원 모델과 해당 제공자 키가 필요합니다","models_btn":"모델","models_title":"역할별 모델 (다음 턴부터 적용)","hist_title":"기록","hist_new":"+ 새 대화","hist_empty":"아직 대화가 없습니다","hist_rename":"이름 바꾸기","hist_open_fail":"대화를 열 수 없습니다"},
  zh: {"subtitle":"接入任意 LLM —— Gemini、OpenAI、Anthropic、开源模型托管服务（DeepInfra · Together · OpenRouter），或本地的 OpenAI 兼容服务器 —— 实时观看 Sherlock 整理它的上下文。","providers":"提供方","providers_hint":"—— 至少连接一个；每个角色可自由搭配","role_main":"主模型","role_summary":"摘要器","role_infer":"推断器","sysprompt_label":"系统提示词（LLM-1 人设）","companions":"协同模型","companions_turbo":"turbo —— 每轮都运行 LLM-2 + LLM-3","companions_cold":"cold_start —— 仅在出现信号时才启用","companions_off":"off —— 单模型（旧版）","companions_help":"💡 <b>turbo</b> 每轮都运行 LLM-3 推断 + LLM-2 压缩（🧠 推断 / 🗜 压缩 面板始终有内容）。<b>cold_start</b> 在出现真实信号需要协同模型之前一直保持单模型；<b>off</b> 是旧版门控。","background":"后台运行","redact":"隐去敏感信息","websearch":"🌐 联网搜索","engine_ddg":"DuckDuckGo（免费，无需密钥）","engine_brave":"Brave（需密钥）","engine_tavily":"Tavily（需密钥）","engine_valyu":"Valyu（需密钥）","engine_off":"关闭","searchkey_ph":"搜索 API 密钥","websearch_help":"LLM-1（搜索 + 抓取）和 LLM-3（时效性）都使用此引擎。DuckDuckGo 免费但新闻检索能力较弱；配上密钥后，Brave/Tavily/Valyu 的结果要好得多。","start_session":"开始会话","export":"⬇ 导出","new_session":"新建会话","thinking":"Sherlock 正在思考…","stop":"停止","try":"试试：","ex1":"我女儿雨欣对花生过敏","ex2":"我在纠结要不要辞职","ex3":"我下周要去东京——推荐一家酒店吧","ex4":"查一下今天上海的天气","ex5":"你还记得我之前说过什么吗？","dr_proposed":"🔬 已建议深度研究","approve":"同意","skip":"跳过","dr_or_yes":"…或者直接回复\"好\"。","chatmode_title":"谁来回答：Sherlock、原始模型，还是两者并排对比","mode_single":"单个 LLM","mode_both":"对比（A/B）","baseline_search_title":"公平对比：单个 LLM 也会用同一引擎执行一次搜索","single_search":"🔎单次","msg_ph":"说点什么…","send":"发送","single_baseline":"⚖ 单个 LLM（基准）","tab_flow":"⚡ 流程","tab_slot":"🧱 槽位","tab_llmio":"💬 LLM 输入/输出","tab_infer":"🧠 推断","tab_compact":"🗜 压缩","tab_memory":"🗃 记忆","tab_carry":"↪ 携带","tab_research":"🔬 研究","thinking_single":"单个 LLM 正在思考…","stopping":"正在停止…","building_agent":"正在构建智能体（首次运行会下载嵌入模型）…","need_provider":"✗ 请先连接提供方并选择模型","dr_running":"🔬 深度研究进行中…","dr_synth":"🔬 正在综合最终答案…","dr_starting":"🔬 深度研究启动中…"},
  ja: {"subtitle":"お好きなLLM — Gemini、OpenAI、Anthropic、オープンソースモデルのホスト（DeepInfra · Together · OpenRouter）、またはローカルのOpenAI互換サーバー — を接続すれば、Sherlockがそのコンテキストをリアルタイムで整える様子を見られます。","providers":"プロバイダー","providers_hint":"— 最低1つ接続してください。役割ごとに自由に組み合わせOK","role_main":"メイン","role_summary":"要約担当","role_infer":"推論担当","sysprompt_label":"システムプロンプト（LLM-1のペルソナ）","companions":"コンパニオン","companions_turbo":"turbo — 毎ターン LLM-2 + LLM-3 を実行","companions_cold":"cold_start — 必要な兆候があるときだけ起動","companions_off":"off — 単一モデル（レガシー）","companions_help":"💡 <b>turbo</b> は毎ターン LLM-3 の推論と LLM-2 の圧縮を実行します（🧠 推論 / 🗜 圧縮 パネルが常に埋まります）。<b>cold_start</b> は本物の兆候がコンパニオンを必要とするまで単一モデルのまま動作します。<b>off</b> はレガシーのゲートです。","background":"バックグラウンド","redact":"秘密情報を伏せる","websearch":"🌐 ウェブ検索","engine_ddg":"DuckDuckGo（無料・キー不要）","engine_brave":"Brave（キー要）","engine_tavily":"Tavily（キー要）","engine_valyu":"Valyu（キー要）","engine_off":"オフ","searchkey_ph":"検索APIキー","websearch_help":"LLM-1（検索＋取得）と LLM-3（鮮度チェック）はどちらもこのエンジンを使います。DuckDuckGo は無料ですがニュースには弱く、Brave / Tavily / Valyu はキーを使えばはるかに良い結果が得られます。","start_session":"セッション開始","export":"⬇ エクスポート","new_session":"新しいセッション","thinking":"Sherlock が考えています…","stop":"停止","try":"試してみる:","ex1":"娘の結衣がピーナッツアレルギーなんです","ex2":"仕事を辞めるべきか迷っています","ex3":"来週、京都に行くんだけどおすすめのホテルを教えて","ex4":"今日の東京の天気を調べて","ex5":"さっき私が言ったこと、覚えてる？","dr_proposed":"🔬 ディープリサーチを提案","approve":"承認","skip":"スキップ","dr_or_yes":"…または「はい」と返すだけでもOK。","chatmode_title":"誰が答えるか：Sherlock、素のモデル、または両方を並べて比較","mode_single":"単一LLM","mode_both":"比較（A/B）","baseline_search_title":"公平な比較：単一LLMも同じエンジンで1回検索を行います","single_search":"🔎単独","msg_ph":"メッセージを入力…","send":"送信","single_baseline":"⚖ 単一LLM（ベースライン）","tab_flow":"⚡ フロー","tab_slot":"🧱 スロット","tab_llmio":"💬 LLM入出力","tab_infer":"🧠 推論","tab_compact":"🗜 圧縮","tab_memory":"🗃 メモリ","tab_carry":"↪ 引き継ぎ","tab_research":"🔬 リサーチ","thinking_single":"単一LLM が考えています…","stopping":"停止中…","building_agent":"エージェントを構築中（初回はエンベッダーをダウンロードします）…","need_provider":"✗ まずプロバイダーを接続してモデルを選んでください","dr_running":"🔬 ディープリサーチを実行中…","dr_synth":"🔬 最終的な回答をまとめています…","dr_starting":"🔬 ディープリサーチを開始中…"},
  fr: {"subtitle":"Connectez n'importe quel LLM — Gemini, OpenAI, Anthropic, un hébergeur de modèles open source (DeepInfra · Together · OpenRouter) ou un serveur local compatible OpenAI — et regardez Sherlock organiser son contexte en temps réel.","providers":"Fournisseurs","providers_hint":"— connectez-en au moins un ; combinez-les librement selon le rôle","role_main":"Principal","role_summary":"Synthétiseur","role_infer":"Inférenceur","sysprompt_label":"Prompt système (persona LLM-1)","companions":"Compagnons","companions_turbo":"turbo — LLM-2 + LLM-3 à chaque tour","companions_cold":"cold_start — uniquement quand un signal les requiert","companions_off":"off — modèle unique (hérité)","companions_help":"💡 <b>turbo</b> exécute l'inférence LLM-3 + la compaction LLM-2 à chaque tour (les panneaux 🧠 Inférence / 🗜 Compaction se remplissent toujours). <b>cold_start</b> reste en modèle unique jusqu'à ce qu'un véritable signal requière les compagnons ; <b>off</b> est le mécanisme hérité.","background":"Arrière-plan","redact":"Masquer les secrets","websearch":"🌐 Recherche web","engine_ddg":"DuckDuckGo (gratuit, sans clé)","engine_brave":"Brave (clé)","engine_tavily":"Tavily (clé)","engine_valyu":"Valyu (clé)","engine_off":"Désactivé","searchkey_ph":"clé API de recherche","websearch_help":"LLM-1 (recherche + récupération) et LLM-3 (fraîcheur) utilisent tous deux ce moteur. DuckDuckGo est gratuit mais faible pour l'actualité ; Brave/Tavily/Valyu donnent de bien meilleurs résultats avec une clé.","start_session":"Démarrer la session","export":"⬇ exporter","new_session":"nouvelle session","thinking":"Sherlock réfléchit…","stop":"Arrêter","try":"essayez :","ex1":"Ma fille Camille est allergique aux arachides","ex2":"J'hésite à démissionner de mon poste","ex3":"Je pars à Lyon la semaine prochaine — conseille-moi un hôtel","ex4":"Cherche la météo d'aujourd'hui à Paris","ex5":"Tu te souviens de ce que je t'ai dit tout à l'heure ?","dr_proposed":"🔬 Recherche approfondie proposée","approve":"Approuver","skip":"Ignorer","dr_or_yes":"…ou répondez simplement « oui ».","chatmode_title":"qui répond : Sherlock, le modèle brut, ou les deux côte à côte","mode_single":"LLM unique","mode_both":"Comparer (A/B)","baseline_search_title":"Comparaison équitable : le LLM unique bénéficie aussi d'une passe de recherche avec le même moteur","single_search":"🔎unique","msg_ph":"Écrivez quelque chose…","send":"Envoyer","single_baseline":"⚖ LLM unique (référence)","tab_flow":"⚡ Flux","tab_slot":"🧱 Emplacement","tab_llmio":"💬 E/S LLM","tab_infer":"🧠 Inférence","tab_compact":"🗜 Compaction","tab_memory":"🗃 Mémoire","tab_carry":"↪ Report","tab_research":"🔬 Recherche","thinking_single":"Le LLM unique réfléchit…","stopping":"arrêt en cours…","building_agent":"construction de l'agent (le premier lancement télécharge l'encodeur)…","need_provider":"✗ connectez un fournisseur et choisissez d'abord les modèles","dr_running":"🔬 recherche approfondie en cours…","dr_synth":"🔬 synthèse de la réponse finale…","dr_starting":"🔬 démarrage de la recherche approfondie…"},
  de: {"subtitle":"Bring ein beliebiges LLM mit — Gemini, OpenAI, Anthropic, einen Host für Open-Source-Modelle (DeepInfra · Together · OpenRouter) oder einen lokalen, OpenAI-kompatiblen Server — und sieh Sherlock dabei zu, wie es dessen Kontext in Echtzeit kuratiert.","providers":"Anbieter","providers_hint":"— mindestens einen verbinden; pro Rolle frei kombinierbar","role_main":"Haupt","role_summary":"Zusammenfasser","role_infer":"Schlussfolgerer","sysprompt_label":"System-Prompt (LLM-1-Persona)","companions":"Begleiter","companions_turbo":"turbo — LLM-2 + LLM-3 in jedem Zug","companions_cold":"cold_start — nur wenn ein Signal sie erfordert","companions_off":"off — einzelnes Modell (Legacy)","companions_help":"💡 <b>turbo</b> führt in jedem Zug LLM-3-Inferenz + LLM-2-Verdichtung aus (die Panels 🧠 Inferenz / 🗜 Verdichtung füllen sich immer). <b>cold_start</b> bleibt beim Einzelmodell, bis ein echtes Signal die Begleiter erfordert; <b>off</b> ist das Legacy-Gate.","background":"Hintergrund","redact":"Geheimnisse schwärzen","websearch":"🌐 Websuche","engine_ddg":"DuckDuckGo (kostenlos, kein Schlüssel)","engine_brave":"Brave (Schlüssel)","engine_tavily":"Tavily (Schlüssel)","engine_valyu":"Valyu (Schlüssel)","engine_off":"Aus","searchkey_ph":"Such-API-Schlüssel","websearch_help":"LLM-1 (Suche + Abruf) und LLM-3 (Aktualität) nutzen beide diese Engine. DuckDuckGo ist kostenlos, aber schwach bei Nachrichten; Brave/Tavily/Valyu liefern mit einem Schlüssel deutlich bessere Ergebnisse.","start_session":"Sitzung starten","export":"⬇ exportieren","new_session":"neue Sitzung","thinking":"Sherlock denkt nach …","stop":"Stopp","try":"Probier:","ex1":"Meine Tochter Yujin hat eine Erdnussallergie","ex2":"Ich bin hin- und hergerissen, ob ich kündigen soll","ex3":"Ich fliege nächste Woche nach Tokio — empfiehl mir ein Hotel","ex4":"Suche nach dem heutigen Wetter in München","ex5":"Erinnerst du dich, was ich vorhin gesagt habe?","dr_proposed":"🔬 Deep Research vorgeschlagen","approve":"Bestätigen","skip":"Überspringen","dr_or_yes":"… oder antworte einfach mit „ja“.","chatmode_title":"wer antwortet: Sherlock, das blanke Modell oder beide nebeneinander","mode_single":"Einzelnes LLM","mode_both":"Vergleichen (A/B)","baseline_search_title":"Fairer Vergleich: Das einzelne LLM bekommt ebenfalls einen Suchdurchlauf mit derselben Engine","single_search":"🔎einzeln","msg_ph":"Sag etwas …","send":"Senden","single_baseline":"⚖ Einzelnes LLM (Baseline)","tab_flow":"⚡ Fluss","tab_slot":"🧱 Slot","tab_llmio":"💬 LLM-E/A","tab_infer":"🧠 Inferenz","tab_compact":"🗜 Verdichtung","tab_memory":"🗃 Speicher","tab_carry":"↪ Übertrag","tab_research":"🔬 Recherche","thinking_single":"Einzelnes LLM denkt nach …","stopping":"wird gestoppt …","building_agent":"Agent wird aufgebaut (beim ersten Lauf wird der Embedder heruntergeladen) …","need_provider":"✗ Verbinde zuerst einen Anbieter und wähle Modelle aus","dr_running":"🔬 Deep Research läuft …","dr_synth":"🔬 finale Antwort wird zusammengeführt …","dr_starting":"🔬 Deep Research startet …"},
  es: {"subtitle":"Conecta cualquier LLM — Gemini, OpenAI, Anthropic, un host de modelos open source (DeepInfra · Together · OpenRouter) o un servidor local compatible con OpenAI — y observa cómo Sherlock organiza su contexto en tiempo real.","providers":"Proveedores","providers_hint":"— conecta al menos uno; combínalos libremente por rol","role_main":"Principal","role_summary":"Resumidor","role_infer":"Inferenciador","sysprompt_label":"Prompt de sistema (persona de LLM-1)","companions":"Acompañantes","companions_turbo":"turbo — LLM-2 + LLM-3 en cada turno","companions_cold":"cold_start — solo cuando una señal los necesita","companions_off":"off — modelo único (heredado)","companions_help":"💡 <b>turbo</b> ejecuta la inferencia de LLM-3 + la compactación de LLM-2 en cada turno (los paneles 🧠 Inferencia / 🗜 Compactación siempre se llenan). <b>cold_start</b> lo mantiene en modelo único hasta que una señal real necesite a los acompañantes; <b>off</b> es la compuerta heredada.","background":"Segundo plano","redact":"Ocultar secretos","websearch":"🌐 Búsqueda web","engine_ddg":"DuckDuckGo (gratis, sin clave)","engine_brave":"Brave (clave)","engine_tavily":"Tavily (clave)","engine_valyu":"Valyu (clave)","engine_off":"Desactivada","searchkey_ph":"clave de API de búsqueda","websearch_help":"LLM-1 (búsqueda + recuperación) y LLM-3 (actualidad) usan este motor. DuckDuckGo es gratis pero flojo para noticias; Brave/Tavily/Valyu dan resultados mucho mejores con una clave.","start_session":"Iniciar sesión","export":"⬇ exportar","new_session":"nueva sesión","thinking":"Sherlock está pensando…","stop":"Detener","try":"prueba:","ex1":"Mi hija Lucía tiene alergia al maní","ex2":"No sé si debería renunciar a mi trabajo","ex3":"Voy a Barcelona la próxima semana — recomiéndame un hotel","ex4":"Busca el clima de hoy en Madrid","ex5":"¿Recuerdas lo que te dije antes?","dr_proposed":"🔬 Investigación a fondo propuesta","approve":"Aprobar","skip":"Omitir","dr_or_yes":"…o simplemente responde «sí».","chatmode_title":"quién responde: Sherlock, el modelo a secas, o ambos en paralelo","mode_single":"LLM único","mode_both":"Comparar (A/B)","baseline_search_title":"Comparación justa: el LLM único también recibe una pasada de búsqueda con el mismo motor","single_search":"🔎único","msg_ph":"Escribe algo…","send":"Enviar","single_baseline":"⚖ LLM único (referencia)","tab_flow":"⚡ Flujo","tab_slot":"🧱 Ranura","tab_llmio":"💬 E/S de LLM","tab_infer":"🧠 Inferencia","tab_compact":"🗜 Compactación","tab_memory":"🗃 Memoria","tab_carry":"↪ Arrastre","tab_research":"🔬 Investigación","thinking_single":"El LLM único está pensando…","stopping":"deteniendo…","building_agent":"construyendo el agente (la primera ejecución descarga el embebedor)…","need_provider":"✗ conecta un proveedor y elige modelos primero","dr_running":"🔬 investigación a fondo en curso…","dr_synth":"🔬 sintetizando la respuesta final…","dr_starting":"🔬 iniciando investigación a fondo…"},
};
let LOCALE = (() => {
  try { const s = localStorage.getItem("sherlock_lang"); if (s && LANG[s]) return s; } catch (e) {}
  const n = (navigator.language || "en").slice(0, 2).toLowerCase();
  return LANG[n] ? n : "en";
})();
function t(key, ...args) {
  const d = LANG[LOCALE] || {};
  let s = d[key] != null ? d[key] : (LANG.en[key] != null ? LANG.en[key] : key);
  args.forEach((a, i) => { s = s.split("{" + i + "}").join(a); });
  return s;
}
window.t = t;
function applyI18n() {
  const get = (k) => t(k);
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.innerHTML = get(el.getAttribute("data-i18n")); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = get(el.getAttribute("data-i18n-ph")); });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => { el.title = get(el.getAttribute("data-i18n-title")); });
  if ($("langSel")) $("langSel").value = LOCALE;
  document.documentElement.lang = LOCALE;
}
function setLang(code) {
  LOCALE = LANG[code] ? code : "en";
  try { localStorage.setItem("sherlock_lang", LOCALE); } catch (e) {}
  applyI18n();
}
if ($("langSel")) $("langSel").onchange = (e) => setLang(e.target.value);
applyI18n();

const ACTOR = {
  llm1: { stripe: "bg-blue-500", text: "text-blue-700", soft: "bg-blue-50", n: "LLM-1" },
  llm2: { stripe: "bg-green-500", text: "text-green-700", soft: "bg-green-50", n: "LLM-2" },
  llm3: { stripe: "bg-purple-500", text: "text-purple-700", soft: "bg-purple-50", n: "LLM-3" },
  llm4: { stripe: "bg-orange-500", text: "text-orange-700", soft: "bg-orange-50", n: "LLM-4" },
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

const S = { prov: {}, sid: null, ws: null, llmio: {}, research: {}, streamOpen: false,
  // v1.12 B4: LLM-4 visualizer — bubble registry (KEY→rendered-markdown element),
  // per-viz status ('pending'|'harnessing'|'ready'|'failed'), a buffer for viz.*
  // events that arrive before their slot exists, and the live iframe harnesses.
  bubbles: {}, vizBuffer: {}, vizStatus: {}, vizHarnesses: new Set() };

/* ---------------- setup: multi-provider connect ---------------- */
// S.prov = { gemini: {creds:{api_key}, models:[...]}, openai: {...}, anthropic: {...}, local: {creds:{base_url,api_key}, models:[...]} }
const PROV_LABEL = { gemini: "Gemini", openai: "OpenAI", anthropic: "Anthropic", deepinfra: "DeepInfra", together: "Together", openrouter: "OpenRouter", local: "Local" };

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
  main: [/gemini-.*flash(?!-lite)(?!.*8b)/i, /gpt-4o(?!-mini)/i, /claude-sonnet/i, /gpt-4\./i,
    /llama-3\.[13]-70b|qwen.*2\.5-72b|qwen3.*(72b|235b)|deepseek-(v3|r1)|mixtral-8x22/i],
  summary: [/flash-lite|flash-8b/i, /gpt-4o-mini|gpt-4\.1-mini|gpt-4\.1-nano/i, /claude-haiku/i,
    /(llama-3\.[12]-)?8b|qwen.*7b|mistral-7b|mini|small|lite/i],
  inference: [/flash-lite|flash-8b/i, /gpt-4o-mini|gpt-4\.1-mini|gpt-4\.1-nano/i, /claude-haiku/i,
    /(llama-3\.[12]-)?8b|qwen.*7b|mistral-7b|mini|small|lite/i],
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
// v1.12 B4: the VISUALIZER (LLM-4) model select — a leading "— same as Main —"
// option (value "") means no dedicated viz model, so the library falls back to
// the main provider. Otherwise identical to fillSelect.
function fillVizSelect(el, current) {
  if (!el) return;
  el.innerHTML = ""; el.disabled = false;
  const same = h("option", "", esc(t("viz_same_as_main"))); same.value = ""; el.appendChild(same);
  for (const [p, info] of Object.entries(S.prov)) {
    const og = document.createElement("optgroup"); og.label = PROV_LABEL[p] || p;
    info.models.forEach((m) => { const o = h("option", "", esc(m.id)); o.value = `${p}::${m.id}`; og.appendChild(o); });
    el.appendChild(og);
  }
  el.value = current && [...el.options].some((o) => o.value === current) ? current : "";
}
function rebuildRoleSelects() {
  for (const [sel, role] of [["modelMain", "main"], ["modelSummary", "summary"], ["modelInference", "inference"]]) {
    const el = $(sel);
    const keep = el.value && [...el.options].some((o) => o.value === el.value) ? el.value : null;
    fillSelect(el, keep);
    if (!keep) el.value = pickDefault(role);
  }
  const vz = $("modelViz");
  if (vz) fillVizSelect(vz, vz.value || "");  // default: — same as Main —
}
const parseSpec = (v) => { const i = (v || "").indexOf("::"); return i < 0 ? null : { provider: v.slice(0, i), model: v.slice(i + 2) }; };

$("startBtn").onclick = async () => {
  if (!currentModels().main) { $("startStatus").textContent = t("need_provider"); return; }
  $("startStatus").textContent = t("building_agent");
  $("startBtn").disabled = true;
  const ok = await createSession();
  $("startBtn").disabled = false;
  if (!ok) { $("startStatus").textContent = "✗ couldn't start the session"; return; }
  initPanels();
  $("setup").classList.add("hidden");
  $("main").classList.remove("hidden"); $("main").classList.add("flex");
};

function mirrorLive() {
  for (const [sel, src, role] of [["liveMain", "modelMain", "main"], ["liveSummary", "modelSummary", "summary"], ["liveInference", "modelInference", "inference"]]) {
    const el = $(sel);
    fillSelect(el, el.value || $(src).value); // keep a mid-session choice on re-fill
    el.onchange = async () => {
      await fetch("/api/select_models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, models: { [role]: parseSpec(el.value) } }) });
    };
  }
  // live companion-mode switch — mirrors the setup choice, changeable per turn
  const lc = $("liveCompanions");
  if (lc) {
    lc.value = lc.value || ($("companionsMode") ? $("companionsMode").value : "turbo");
    lc.onchange = async () => {
      await fetch("/api/companions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, mode: lc.value }) });
    };
  }
  // live async (background) switch — mirrors the setup checkbox, changeable anytime
  const lb = $("liveBackground");
  if (lb) {
    lb.value = $("optBackground") && !$("optBackground").checked ? "off" : "on";
    lb.onchange = async () => {
      await fetch("/api/background", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, on: lb.value === "on" }) });
    };
  }
  // live deep-research verify-tier switch — A/B the accuracy layer per research run
  const lv2 = $("liveVerify");
  if (lv2) {
    lv2.value = lv2.value || ($("verifyTier") ? $("verifyTier").value : "faithfulness");
    lv2.onchange = async () => {
      await fetch("/api/verify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, tier: lv2.value }) });
    };
  }
  // v1.12 A5: live long-term-memory on/off — mirrors the setup checkbox
  const llt = $("liveLongTerm");
  if (llt) {
    llt.value = $("optLongTerm") && $("optLongTerm").checked ? "on" : "off";
    llt.onchange = async () => {
      await fetch("/api/long_term", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, on: llt.value === "on" }) });
      // keep the setup checkbox in sync (same precedent as the verify tier)
      if ($("optLongTerm")) $("optLongTerm").checked = llt.value === "on";
      // v1.12 F4: a live ON-flip over a session that started WITHOUT a profile
      // dir writes durable facts to a throwaway tempdir — warn they won't persist.
      if (llt.value === "on" && !S.ltmProfileBound) addBubble("system", t("ltm_temp_note"));
      refreshLTM();
    };
  }
  // v1.12 A5: live incognito (pause writes) — mirrors the setup checkbox
  const linc = $("liveIncognito");
  if (linc) {
    linc.value = $("optIncognito") && $("optIncognito").checked ? "on" : "off";
    linc.onchange = async () => {
      await fetch("/api/incognito", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, on: linc.value === "on" }) });
      // keep the setup checkbox in sync (same precedent as the verify tier)
      if ($("optIncognito")) $("optIncognito").checked = linc.value === "on";
    };
  }
  // v1.12 B4: live visualizer MODEL select — mirrors the setup pick ("same as Main")
  const lvz = $("liveViz");
  if (lvz) {
    fillVizSelect(lvz, lvz.value || ($("modelViz") ? $("modelViz").value : ""));
    lvz.onchange = async () => {
      await fetch("/api/select_models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, models: { viz: parseSpec(lvz.value) } }) });
    };
  }
  // v1.12 B4: live visualizer ON/OFF toggle — mirrors the setup checkbox → /api/visualization
  const lvo = $("liveVizOn");
  if (lvo) {
    lvo.value = $("optViz") && $("optViz").checked ? "on" : "off";
    lvo.onchange = async () => {
      await fetch("/api/visualization", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, on: lvo.value === "on" }) });
      if ($("optViz")) $("optViz").checked = lvo.value === "on";
    };
  }
}

// Build the session config from the CURRENT controls — live top-bar selects (if a
// mid-session change was made) win over the setup-screen selects.
function pickVal(liveId, setupId) { const lv = $(liveId) && $(liveId).value; return lv || $(setupId).value; }
function currentModels() {
  return {
    main: parseSpec(pickVal("liveMain", "modelMain")),
    summary: parseSpec(pickVal("liveSummary", "modelSummary")),
    inference: parseSpec(pickVal("liveInference", "modelInference")),
    // v1.12 B4: viz (LLM-4) — null when "— same as Main —" (falls back to main)
    viz: parseSpec(pickVal("liveViz", "modelViz")),
  };
}
function currentSettings() {
  const mode = ($("liveCompanions") && $("liveCompanions").value) || ($("companionsMode") && $("companionsMode").value) || "turbo";
  const st = {
    embedding: "local", background: $("optBackground").checked, redact_secrets: $("optRedact").checked,
    search_engine: $("searchEngine").value, search_api_key: $("searchKey").value.trim() || null,
    companions_mode: mode, force_companions: mode === "turbo",
    deep_research_verify: ($("liveVerify") && $("liveVerify").value) || ($("verifyTier") && $("verifyTier").value) || "faithfulness",
    // v1.12 A5: long-term memory (off by default; opt-in per session)
    long_term: !!($("optLongTerm") && $("optLongTerm").checked),
    ltm_profile: ($("ltmProfile") && $("ltmProfile").value.trim()) || "default",
    ltm_incognito: !!($("optIncognito") && $("optIncognito").checked),
    // v1.12 B4: LLM-4 inline visualizer (off by default; opt-in per session)
    visualization: !!($("optViz") && $("optViz").checked),
  };
  // v1.12 P1: optional text→image model for `image:` viz markers — only sent
  // when a model id was actually typed (absent → the modality stays dormant).
  const imgModel = $("vizImageModel") && $("vizImageModel").value.trim();
  // audit: the row is HIDDEN when viz is off, but the input keeps its value —
  // never ship image_model for a session that has visualization disabled.
  if (imgModel && st.visualization) st.image_model = { provider: $("vizImageProvider").value, model: imgModel };
  return st;
}
// POST /api/session with the current config, swap the WebSocket. Returns true on
// success (S.sid is the new session). Used by BOTH Start and New session.
async function createSession() {
  const models = currentModels();
  if (!models.main) return false;
  // v1.12 F4: remember whether THIS session was built bound to a persistent
  // profile dir (long-term ON at session start). A later live ON-flip over a
  // session that started OFF only writes durable facts to a throwaway tempdir.
  S.ltmProfileBound = !!currentSettings().long_term;
  const providers = {};
  for (const [p, info] of Object.entries(S.prov)) providers[p] = info.creds;
  let j;
  try {
    const r = await fetch("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ providers, models, system_prompt: $("systemPrompt").value, settings: currentSettings() }) });
    j = await r.json().catch(() => ({ error: "bad response" }));
  } catch (e) { return false; }
  if (!j || j.error) return false;
  const old = S.ws;
  S.sid = j.session_id;            // old socket's handlers now no-op (sid mismatch)
  if (old) { try { old.close(); } catch (e) {} }
  mirrorLive(); connectWS();
  refreshHistory(); // v1.12 H2: the sidebar lists the NEW session's store
  return true;
}
// v1.12 H2: wipe ONLY the chat panes + their viz machinery — used both by a full
// session reset and by opening/starting a conversation on the SAME session (the
// WS, token counters and system panels stay).
function clearChatPane() {
  $("chat").innerHTML = ""; if ($("chatB")) $("chatB").innerHTML = "";
  // v1.12 B4: drop stale viz harnesses + registries so the fresh pane starts clean.
  // Cancel each in-flight 4s harness timer too, else a stale settle could fire
  // repairViz against a slot that no longer exists.
  try { S.vizHarnesses.forEach((hh) => { try { clearTimeout(hh.timer); } catch (e) {} try { hh.iframe.remove(); } catch (e) {} }); } catch (e) {}
  S.bubbles = {}; S.vizBuffer = {}; S.vizStatus = {}; S.vizHarnesses = new Set();
  S.stream = null; S.streamOpen = false;
  setThinking(false); hideDRBanner();
}
// Wipe the chat + panels + counters for a fresh session, without leaving the app.
function resetChatUI() {
  clearChatPane();
  if ($("tab-flow")) $("tab-flow").innerHTML = "";
  initPanels();
  for (const k of ["main", "summary", "inference", "viz"]) TOK[k] = { i: 0, o: 0, n: 0, c: 0 };
  BASE.i = 0; BASE.o = 0; renderTokBar();
  S.llmio = {}; S.research = {}; S.busy = false; lastTurn = null;
}

// Show the API-key field only for keyed search engines.
$("searchEngine").onchange = () => {
  const keyed = ["brave", "tavily", "valyu"].includes($("searchEngine").value);
  $("searchKey").classList.toggle("hidden", !keyed);
};

// v1.12 P1: the optional image-model row only makes sense with the viz toggle on.
function updateVizImageRow() {
  const row = $("vizImageRow"); if (!row) return;
  const on = !!($("optViz") && $("optViz").checked);
  row.classList.toggle("hidden", !on);
  row.classList.toggle("flex", on);
}
if ($("optViz")) $("optViz").addEventListener("change", updateVizImageRow);
updateVizImageRow();

// "new session" — fresh session in place, KEEPING the connected providers, model
// picks and settings (no trip back to the API-key screen).
$("newSession").onclick = async () => {
  const btn = $("newSession");
  btn.disabled = true; const label = btn.textContent; btn.textContent = "…";
  const ok = await createSession();
  btn.disabled = false; btn.textContent = label;
  if (ok) resetChatUI();
  else addBubble("system", "✗ couldn't start a new session");
};

// Session export: the browser downloads the markdown (Content-Disposition).
$("exportBtn").onclick = () => { if (S.sid) window.open(`/api/export?session_id=${encodeURIComponent(S.sid)}`, "_blank"); };

/* ---------------- websocket ---------------- */
function connectWS() {
  const mySid = S.sid; // the session THIS socket belongs to
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${mySid}`);
  S.ws = ws;
  ws.onopen = () => { if (S.sid === mySid) { $("wsStatus").textContent = "ws: live"; $("wsStatus").className = "ml-auto text-green-400"; } };
  ws.onclose = () => {
    if (S.sid !== mySid) return; // a new session superseded this socket → let it die
    $("wsStatus").textContent = "ws: reconnecting…"; $("wsStatus").className = "ml-auto text-amber-400";
    setTimeout(() => { if (S.sid === mySid) connectWS(); }, 2000);
  };
  ws.onmessage = (e) => { if (S.sid === mySid) try { handleEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
}

/* ---------------- chat ---------------- */
// One composer button: Send when idle, Stop (red) while a turn is generating —
// the ChatGPT/Claude/Gemini pattern. `S.busy` is driven by setThinking().
function doStop() {
  if (!S.sid) return;
  fetch("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) }).catch(() => {});
  setThinking(true, t("stopping"));
}
function setComposerBusy(on) {
  S.busy = on;
  const b = $("send");
  if (!b) return;
  b.textContent = on ? t("stop") : t("send");
  b.classList.toggle("bg-blue-600", !on); b.classList.toggle("hover:bg-blue-700", !on);
  b.classList.toggle("bg-red-500", on); b.classList.toggle("hover:bg-red-600", on);
}
$("send").onclick = () => (S.busy ? doStop() : sendMsg());
$("msg").addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return; // IME composition (Korean/JP/CN) — never submit
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!S.busy) sendMsg(); } // Enter=send, Shift+Enter=newline
});
// auto-grow the textarea up to the CSS max-height
$("msg").addEventListener("input", () => { const m = $("msg"); m.style.height = "auto"; m.style.height = Math.min(m.scrollHeight, 160) + "px"; });
async function sendMsg() {
  const text = $("msg").value.trim(); if (!text || !S.sid || S.busy) return;
  const mode = $("chatMode") ? $("chatMode").value : "sherlock";
  S.streamOpen = true; // a new user turn → streaming allowed until its turn.completed
  $("msg").value = ""; $("msg").style.height = "auto";
  addBubble("user", text);
  // "both": mirror the user message at the top of the MIDDLE column too, so
  // the sherlock + baseline replies align per turn.
  if (mode === "both") addBubble("user", text, $("chatB"));
  setComposerBusy(true); setThinking(true, mode === "single" ? t("thinking_single") : undefined);
  S.llmio = {};
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, message: text, mode, baseline_search: $("baselineSearch") ? $("baselineSearch").checked : true }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { addBubble("system", "✗ " + (j.error || r.status)); setThinking(false); setComposerBusy(false); return; }
    if (j.latency_ms != null) {
      // The sherlock bubble itself arrives via the turn.completed WS event
      // (which also stashes its token counts) — here we add the meta line.
      S.lastSherlockLatency = j.latency_ms;
      const t = S.lastTurnTokens || {};
      addMetaLine(`⏱ ${j.latency_ms}ms · ${t.i || 0}/${t.o || 0} tok`);
    }
    if (j.baseline) renderBaseline(j.baseline, mode);
    if (mode === "single") { setThinking(false); setComposerBusy(false); } // no agent turn → no turn.completed
  }
  catch (e) { addBubble("system", "✗ " + e); setThinking(false); setComposerBusy(false); }
}
// Hover-to-copy on assistant messages (copies the markdown source).
function attachCopy(bubble, text) {
  bubble.classList.add("msg-wrap");
  const btn = h("div", "msg-copy", "⧉");
  btn.title = "copy";
  btn.onclick = () => {
    try { navigator.clipboard.writeText(text); } catch (e) {}
    btn.textContent = "✓"; setTimeout(() => (btn.textContent = "⧉"), 1200);
  };
  bubble.appendChild(btn);
}
function addBubble(role, text, target) {
  const box = target || $("chat");
  const wrap = h("div", "flex " + (role === "user" ? "justify-end" : "justify-start"));
  const cls = role === "user" ? "bg-blue-600 text-white" : role === "assistant" ? "bg-white border" : "bg-amber-100 text-amber-800 text-xs";
  // Assistant replies render as sanitized markdown; user/system bubbles stay
  // escaped plain text.
  const bubble = role === "assistant"
    ? h("div", `max-w-[85%] px-3 py-2 rounded-2xl text-sm prose-md ${cls}`, mdRender(text))
    : h("div", `max-w-[85%] px-3 py-2 rounded-2xl text-sm whitespace-pre-wrap ${cls}`, esc(text));
  wrap.appendChild(bubble);
  if (role === "assistant" && text) attachCopy(bubble, text);
  box.appendChild(wrap); box.scrollTop = box.scrollHeight;
  // v1.12 B4: turn any ⟦viz:id⟧ placeholders in a rendered assistant bubble into
  // loading slots (spliced with sandboxed iframes once viz.rendered arrives).
  if (role === "assistant") spliceVizSlots(bubble);
  return bubble;
}
function addMetaLine(text, target) {
  const box = target || $("chat");
  const wrap = h("div", "flex justify-start");
  wrap.appendChild(h("div", "text-[10px] text-slate-400 px-3 -mt-2", esc(text)));
  box.appendChild(wrap); box.scrollTop = box.scrollHeight;
}

/* ---------------- streaming: live assistant bubble + 💭 thinking ---------------- */
// The main reply streams token-by-token into ONE live bubble per turn. Reasoning
// ("thinking") tokens from reasoning models fill a collapsible panel above it.
// On turn.completed the answer is replaced with the authoritative, tag-stripped
// markdown so the final text is always clean even though the stream showed raw.
function liveBubble(turn) {
  if (S.stream && S.stream.turn === turn) return S.stream;
  // A new turn supersedes any stale handle — but never leave the old bubble
  // blinking: strip its caret so a superseded partial can't linger as a zombie.
  if (S.stream) { try { S.stream.answerEl.classList.remove("stream-caret"); } catch (e) {} }
  S.stream = null;
  const box = $("chat");
  const wrap = h("div", "flex justify-start");
  const bubble = h("div", "max-w-[85%] px-3 py-2 rounded-2xl text-sm bg-white border");
  const thinkWrap = h("details", "mb-1 text-xs hidden");
  const summary = document.createElement("summary");
  summary.className = "cursor-pointer text-purple-600 select-none";
  summary.textContent = "💭 " + (window.t ? t("thinking") : "Thinking");
  const thinkEl = h("div", "mt-1 whitespace-pre-wrap text-slate-500 max-h-48 overflow-auto border-l-2 border-purple-200 pl-2");
  thinkWrap.appendChild(summary); thinkWrap.appendChild(thinkEl);
  const answerEl = h("div", "whitespace-pre-wrap stream-caret");
  bubble.appendChild(thinkWrap); bubble.appendChild(answerEl);
  wrap.appendChild(bubble); box.appendChild(wrap);
  S.stream = { turn, answer: "", reasoning: "", answerEl, thinkEl, thinkWrap, wrap, bubble };
  return S.stream;
}
function streamAnswer(turn, chunk) {
  if (!chunk || !S.streamOpen) return; // drop late deltas after turn.completed (no zombie bubble)
  const s = liveBubble(turn);
  s.answer += chunk;
  // v1.12 B4: deltas are PRE-strip, so the raw <<sherlock-viz: …>> marker (or a
  // ⟦viz:…⟧ placeholder) can flash mid-stream. Hide complete AND partial markers
  // from the live text; the finalized bubble gets proper slots. (Display only —
  // s.answer keeps the raw accumulation for the fallback path.)
  s.answerEl.textContent = filterVizForStream(s.answer);
  autoScroll();
}
// Strip viz markers/placeholders (complete + trailing-partial) from streaming text.
function filterVizForStream(text) {
  let out = String(text == null ? "" : text)
    .replace(/<<\s*sherlock-viz\s*:[\s\S]*?>>/gi, "")
    .replace(/⟦viz:[^⟧]*⟧/g, "");
  // a trailing partial placeholder ⟦… with no closing ⟧ (still streaming in)
  const lb = out.lastIndexOf("⟦");
  if (lb !== -1 && out.indexOf("⟧", lb) === -1) out = out.slice(0, lb);
  // a trailing partial marker <<… that could still become <<sherlock-viz:…>>
  const lt = out.lastIndexOf("<<");
  if (lt !== -1 && out.indexOf(">>", lt) === -1) {
    const tail = out.slice(lt).replace(/\s+/g, "").toLowerCase();
    const P = "<<sherlock-viz";
    if (tail.length <= P.length ? P.startsWith(tail) : tail.startsWith(P)) out = out.slice(0, lt);
  }
  return out;
}
function streamReasoning(turn, chunk) {
  if (!chunk || !S.streamOpen) return; // drop late deltas after turn.completed
  const s = liveBubble(turn);
  s.reasoning += chunk; s.thinkEl.textContent = s.reasoning;
  s.thinkWrap.classList.remove("hidden"); s.thinkWrap.open = true;
  autoScroll();
}
// Replace the live answer with the clean final markdown; collapse (keep) thinking.
// Returns false if there was no live bubble for this turn (caller adds one).
function finalizeStream(turn, finalText) {
  // Finalize whatever stream is live — the playground runs ONE turn at a time, so
  // a turn-number drift between llm.delta (session.turn) and turn.completed (agent
  // turn) must never strand a streamed reply.
  if (!S.stream) return false;
  const s = S.stream; S.stream = null;
  // finalText (turn.completed response_text) already carries the ⟦viz:…⟧
  // placeholders; the raw-stream fallback (s.answer) still has <<sherlock-viz>>
  // markers, so filter those out of the fallback so nothing raw survives.
  const txt = finalText != null && finalText !== "" ? finalText : filterVizForStream(s.answer);
  s.answerEl.className = "prose-md"; // drops the streaming caret
  s.answerEl.innerHTML = mdRender(txt);
  if (s.reasoning) s.thinkWrap.open = false; else s.thinkWrap.remove();
  if (txt) attachCopy(s.bubble, txt);
  // v1.12 B4: splice loading slots for any placeholders + register the bubble so
  // async viz.rendered events (which land AFTER finalize) can find it by turn.
  spliceVizSlots(s.answerEl);
  registerBubble("t" + turn, s.answerEl);
  autoScroll();
  return true;
}
// Auto-scroll only when the user is already near the bottom — never yank them
// back up while they're reading scroll-back (ChatGPT/Claude behavior).
function autoScroll() {
  const box = $("chat");
  if (box.scrollHeight - box.scrollTop - box.clientHeight < 120) box.scrollTop = box.scrollHeight;
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
  // Visual activity bar ONLY. The composer (Send↔Stop) is NOT tied to this —
  // it's freed when LLM-1's reply lands (turn.completed), so background
  // companions (LLM-2/LLM-3) never lock the input.
  $("thinking").classList.toggle("hidden", !on);
  $("thinking").classList.toggle("flex", on);
  $("thinkingLabel").textContent = text || t("thinking");
}

/* ---------------- dark mode ---------------- */
function applyDark(on) {
  document.documentElement.classList.toggle("dark", on);
  try { localStorage.setItem("sherlock_dark", on ? "1" : "0"); } catch (e) {}
  document.querySelectorAll("#darkToggle, #darkToggle2").forEach((b) => (b.textContent = on ? "☀️" : "🌙"));
  syncVizFrameTheme(); // hoisted from the viz section below
}
(function initDark() {
  let on = false;
  try { const s = localStorage.getItem("sherlock_dark"); on = s === "1" || (s == null && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches); } catch (e) {}
  applyDark(on);
  document.querySelectorAll("#darkToggle, #darkToggle2").forEach((b) => (b.onclick = () => applyDark(!document.documentElement.classList.contains("dark"))));
})();

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
    case "turn.completed": {
      S.lastTurnTokens = { i: d.prompt_tokens || 0, o: d.completion_tokens || 0 };
      // LLM-1's user-facing reply is done → free the composer NOW. Companions
      // (LLM-2/LLM-3) keep running in the background and update the panels via
      // their own events; they must not block the next message.
      setComposerBusy(false);
      // NEVER remove the message. A truthy `error` alongside real response_text
      // just means the backend's _looks_like_error_response heuristic misfired on
      // a valid reply — show it. Only fall back to a notice if there's nothing.
      const txt = d.response_text || "";
      if (!finalizeStream(ev.turn, txt)) {
        // v1.12 B4: register the fallback bubble too so async viz.rendered can find it
        if (txt) registerBubble("t" + ev.turn, addBubble("assistant", txt));
        else if (d.error) addBubble("system", "⚠ provider error — check the LLM I/O panel");
      }
      S.streamOpen = false; // reply done → any further deltas for this turn are stale, ignore them
      break;
    }
    case "llm.delta": streamAnswer(ev.turn, d.chunk || ""); break;
    case "llm.reasoning_delta": streamReasoning(ev.turn, d.chunk || ""); break;
    case "slot.assembled": renderSlot(d); break;
    case "llm.call": renderLLMIO(d); countTokens(d); break;
    case "infer.done": renderInference(d); break;
    case "compact.done": renderCompaction(d); break;
    case "memory.snapshot": renderMemory(d.rows || []); break;
    // v1.12 A5: long-term memory transparency — a chip in chat + refresh the tab
    case "memory.promoted": ltmRememberChip(d); refreshLTM(); break;
    case "memory.saved": ltmRememberChip(d); refreshLTM(); break;
    case "memory.updated": refreshLTM(); break;
    case "memory.deleted": addBubble("system", `🧠 ${t("ltm_forgot")}${d.count ? " (" + d.count + ")" : ""}`); refreshLTM(); break;
    case "memory.wiped": addBubble("system", `🧠 ${t("ltm_wiped")}${d.count != null ? " (" + d.count + ")" : ""}`); refreshLTM(); break;
    case "memory.imported": addBubble("system", `🧠 ${t("ltm_imported")}: ${d.imported || 0}${d.skipped ? " · skip " + d.skipped : ""}`); refreshLTM(); break;
    // v1.12 B4: LLM-4 inline visualizer — placeholder → sandboxed iframe lifecycle
    case "viz.pending": onVizPending(d); break;
    case "viz.rendered": onVizRendered(d); break;
    case "viz.repairing": onVizRepairing(d); break;
    case "viz.failed": onVizFailed(d); break;
    case "decay.done": S.lastDecay = d; break;
    case "carry.snapshot": renderCarry(d); break;
    case "carry.stored": renderCarry(d); break;
    case "tool.start": setThinking(true, `🔧 ${d.kind}: ${trim(d.payload, 40)}…`); break;
    case "tool.done": setThinking(true, d.ok ? `🔧 ${d.kind} done${d.result_count != null ? " · " + d.result_count + " results" : ""}` : `🔧 ${d.kind} failed: ${trim(d.error, 50)}`); break;
    case "background.start": setThinking(true, t("bg_running")); break;
    case "background.end": setThinking(false); break;
    // v1.12 H2: a finished turn may have created/renamed a conversation → refresh
    case "turn.done": setThinking(false); refreshHistory(); break;
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
      setThinking(true, t("dr_synth")); renderResearch(); break;
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
    // v1.11: the accuracy layer runs AFTER synthesis — keep the user informed
    // instead of leaving the indicator stuck on "synthesising…".
    case "deep_research.faithfulness":
    case "deep_research.consistency":
    case "deep_research.web_recheck":
      S.research.verify = d; setThinking(true, "🔎 verifying accuracy…"); renderResearch(); break;
    case "deep_research.verify_skipped":
    case "deep_research.coverage_steer":
      renderResearch(); break;
  }
}

/* ---------------- cumulative token bar ---------------- */
const TOK = { main: { i: 0, o: 0, n: 0, c: 0 }, summary: { i: 0, o: 0, n: 0, c: 0 }, inference: { i: 0, o: 0, n: 0, c: 0 }, viz: { i: 0, o: 0, n: 0, c: 0 } };
const BASE = { i: 0, o: 0 }; // cumulative bare-model (A/B baseline) tokens
const kfmt = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n));
function renderTokBar() {
  // v1.12 B4: L4 = the visualizer (LLM-4). Only shown once it's actually spent
  // tokens (a dedicated viz model) — viz-via-main tokens count under L1, honestly.
  const total = TOK.main.i + TOK.main.o + TOK.summary.i + TOK.summary.o + TOK.inference.i + TOK.inference.o + TOK.viz.i + TOK.viz.o;
  const cached = TOK.main.c + TOK.summary.c + TOK.inference.c + TOK.viz.c;
  const l4 = TOK.viz.i + TOK.viz.o ? ` · L4 ${kfmt(TOK.viz.i)}/${kfmt(TOK.viz.o)}` : "";
  $("tokBar").textContent =
    `🪙 L1 ${kfmt(TOK.main.i)}/${kfmt(TOK.main.o)} · L2 ${kfmt(TOK.summary.i)}/${kfmt(TOK.summary.o)} · L3 ${kfmt(TOK.inference.i)}/${kfmt(TOK.inference.o)}${l4} · Σ ${kfmt(total)}${cached ? ` · ⚡cached ${kfmt(cached)}` : ""}${BASE.i + BASE.o ? ` · single ${kfmt(BASE.i)}/${kfmt(BASE.o)}` : ""}`;
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
  S.streamOpen = true; // deep-research synthesis streams its own reply
  hideDRBanner();
  if (!S.research.topic) S.research = { topic: d.topic || "", rounds: [], docs: [], folded: [], answer: "" };
  S.research.status = "researching"; S.research.topic = d.topic || S.research.topic;
  if (d.plan) S.research.plan = d.plan;
  setThinking(true, t("dr_running")); renderResearch(); flashTab("research");
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
  const el = addBubble("assistant", d.answer || "");
  // v1.12 B4: register the DR answer bubble so its viz.rendered events (keyed by
  // research_id) can find the placeholders that addBubble just spliced into slots.
  if (d.research_id != null) registerBubble("dr:" + d.research_id, el);
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
  // v1.11 — the v1.10 accuracy layer, now visible in the flow
  "deep_research.faithfulness": (d) => `🔎 faithfulness (LLM-2) · ${d.groups_checked} groups · ${d.fixes_applied} fixed${d.flagged_for_web ? " · " + d.flagged_for_web + " → web" : ""}`,
  "deep_research.consistency": (d) => `🧩 consistency sweep (LLM-2) · ${d.reconciled} reconciled`,
  "deep_research.web_recheck": (d) => `🌐 web re-check (LLM-3) · ${d.checked} checked · ${d.corrected} corrected · ${d.unverifiable} unverified`,
  "deep_research.verify_skipped": (d) => `⚠ verify skipped · ${d.stage} · ${d.reason}`,
  "deep_research.coverage_steer": (d) => `🧭 coverage steer · ${d.covered}/${d.total} covered · gaps: ${trim((d.uncovered || []).join(", "), 45)}`,
  "deep_research.strategy_failed": (d) => `📋 ✗ strategy failed · ${trim(d.error, 55)}`,
  "memory.redaction_failed": (d) => `🔒 ✗ redaction failed — content withheld · ${trim(d.error, 45)}`,
  // v1.12 A5: long-term memory lifecycle
  "memory.promoted": (d) => `🧠 promoted ${d.count} → long-term` + ((d.items || [])[0] ? ` · ${trim(d.items[0].content, 45)}${d.items[0].category ? " (" + d.items[0].category + ")" : ""}` : ""),
  "memory.saved": (d) => `🧠 saved → long-term` + (d.category ? ` · ${d.category}` : ""),
  "memory.updated": (d) => `🧠 long-term updated`,
  "memory.delete_pending": (d) => `🧠 delete pending · ${d.count} row(s) — awaiting confirm`,
  "memory.deleted": (d) => `🧠 deleted ${d.count || 0} long-term row(s)`,
  "memory.wiped": (d) => `🧠 wiped long-term · ${d.count || 0} row(s)` + (d.backup_path ? " · backup saved" : ""),
  "memory.exported": (d) => `🧠 exported long-term · ${d.format} · ${d.count} rows`,
  "memory.imported": (d) => `🧠 imported long-term · ${d.imported || 0} (skipped ${d.skipped || 0})`,
  "memory.remember_cue": (d) => `🧠 remember-cue detected`,
  "memory.consistency_confirm_error": (d) => `LLM-2 ✗ consistency confirm error · ${trim(d.error, 45)}`,
  // v1.12 B4: LLM-4 inline visualizer
  "viz.pending": (d) => `📊 viz queued · ${trim(d.description || d.viz_id, 55)}`,
  "viz.rendered": (d) => `📊 viz rendered · ${d.viz_id} · ${d.validated || "static"} · ${d.bytes || 0}B`,
  "viz.repairing": (d) => `📊 viz repairing · ${d.viz_id} · round ${d.round}${d.runtime ? " (runtime)" : ""}`,
  "viz.failed": (d) => `📊 ✗ viz failed · ${d.viz_id} · ${trim(d.reason, 50)}`,
  "compact.error": (d) => `LLM-2 ✗ compaction error · ${trim(d.error, 55)}`,
  "infer.error": (d) => `LLM-3 ✗ inference error · ${trim(d.error, 55)}`,
};
const trim = (s, n) => { s = (s || "").replace(/\s+/g, " "); return s.length > n ? s.slice(0, n) + "…" : s; };
let lastTurn = null;
function appendFlow(ev) {
  // streaming deltas drive the live bubble + 💭 panel only — never the Flow log
  // (otherwise every generated token would spam a Flow card).
  if (ev.type === "llm.delta" || ev.type === "llm.reasoning_delta") return;
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
// v1.12 A5: long-term-memory lifecycle events flash the 🧠 Long-term tab; the
// per-conversation store events (snapshot/retrieval/redaction) stay on 🗃 Memory.
const LTM_FLOW_EVENTS = new Set([
  "memory.promoted", "memory.saved", "memory.updated", "memory.delete_pending",
  "memory.deleted", "memory.wiped", "memory.exported", "memory.imported", "memory.remember_cue",
]);
function tabForType(t) {
  if (t === "slot.assembled") return "slot";
  if (t === "llm.call") return "llmio";
  if (t === "infer.done") return "infer";
  if (t === "compact.done") return "compact";
  if (LTM_FLOW_EVENTS.has(t)) return "ltm";
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
  // v1.5: the FINAL user message — the per-turn SYSTEM-ANALYSIS block where the
  // perception OBSERVED/PRIOR, memory-consistency cue, and inference notebook are
  // injected. Highlight those headers so the upgrade's effect is visible.
  if (d.final_user_message) {
    const fpre = h("pre", "mono text-[11px] bg-white border rounded p-3 whitespace-pre-wrap leading-relaxed");
    fpre.innerHTML = esc(d.final_user_message)
      .replace(/(═══[^\n]*═══)/g, '<span class="font-bold text-slate-500">$1</span>')
      .replace(/(OBSERVED \(code-verified[^\n]*|PRIOR \(probabilistic[^\n]*|MEMORY-CONSISTENCY CHECK[^\n]*|INFERENCE NOTEBOOK[^\n]*|RAW STEPS[^\n]*|CONCLUSIONS[^\n]*|live\/time-sensitive[^\n]*)/g, '<span class="font-bold text-emerald-700">$1</span>');
    root.appendChild(h("div", "text-xs font-semibold text-slate-500 mt-3 mb-1", "Final user message (TIER 3 — this-turn SYSTEM ANALYSIS + the user's question)"));
    root.appendChild(fpre);
  }
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

/* ---------------- LONG-TERM MEMORY (v1.12 A5) ---------------- */
// Category → chip colour (fixed taxonomy from the library).
const LTM_CAT_CHIP = {
  user_directive: "bg-fuchsia-100 text-fuchsia-700",
  identity_health: "bg-rose-100 text-rose-700",
  stable_preference: "bg-blue-100 text-blue-700",
  relationship: "bg-emerald-100 text-emerald-700",
  long_term_project: "bg-amber-100 text-amber-700",
};
function ltmRememberChip(d) {
  // memory.promoted carries items[].content; memory.saved carries just category.
  const items = (d && d.items) || [];
  if (items.length) {
    items.slice(0, 3).forEach((it) =>
      addBubble("system", `🧠 ${t("ltm_remembered")}: ${trim(it.content, 70)}${it.category ? " (" + it.category + ")" : ""}`));
  } else {
    addBubble("system", `🧠 ${t("ltm_remembered")}${d && d.category ? " (" + d.category + ")" : ""}`);
  }
}
async function refreshLTM() {
  if (!S.sid) return;
  try {
    const r = await fetch(`/api/memory/long_term?session_id=${encodeURIComponent(S.sid)}`);
    const j = await r.json().catch(() => ({}));
    if (j && !j.error) renderLTM(j.rows || []);
  } catch (e) { /* best-effort — the tab keeps its last render */ }
}
async function ltmDelete(id) {
  if (!confirm(t("ltm_delete_confirm"))) return;
  try {
    const r = await fetch("/api/memory/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, id }) });
    const j = await r.json().catch(() => ({}));
    if (j.error) addBubble("system", "✗ " + j.error);
  } catch (e) { addBubble("system", "✗ " + e); }
  refreshLTM();
}
async function ltmWipe() {
  if (!confirm(t("ltm_wipe_confirm1"))) return;
  if (!confirm(t("ltm_wipe_confirm2"))) return;
  try {
    const r = await fetch("/api/memory/wipe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) });
    const j = await r.json().catch(() => ({}));
    if (j.error) addBubble("system", "✗ " + j.error);
    else addBubble("system", `🧠 ${t("ltm_wiped")} (${j.removed || 0})` + (j.backup_path ? ` · backup: ${trim(j.backup_path, 50)}` : ""));
  } catch (e) { addBubble("system", "✗ " + e); }
  refreshLTM();
}
function ltmExport(fmt) {
  if (!S.sid) return;
  window.open(`/api/memory/export?session_id=${encodeURIComponent(S.sid)}&fmt=${encodeURIComponent(fmt || "md")}`, "_blank");
}
function ltmImportFile() {
  const inp = document.createElement("input");
  inp.type = "file";
  inp.accept = ".md,.markdown,.json,.txt,text/markdown,application/json,text/plain";
  inp.onchange = () => {
    const f = inp.files && inp.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = async () => {
      const text = String(rd.result || "");
      if (!text.trim()) return;
      try {
        const r = await fetch("/api/memory/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, text }) });
        const j = await r.json().catch(() => ({}));
        if (j.error) addBubble("system", `🧠 ✗ ${t("ltm_import_fail")}: ${j.error}`);
        else addBubble("system", `🧠 ${t("ltm_imported")}: ${j.imported || 0}${j.skipped ? " · skip " + j.skipped : ""}`);
      } catch (e) { addBubble("system", "✗ " + e); }
      refreshLTM();
    };
    rd.readAsText(f);
  };
  inp.click();
}
function renderLTM(rows) {
  S.ltmRows = rows || S.ltmRows || [];
  rows = S.ltmRows;
  const root = $("tab-ltm"); if (!root) return;
  root.innerHTML = "";
  const head = h("div", "flex items-center gap-2 mb-2 flex-wrap");
  head.appendChild(h("div", "text-sm font-bold text-fuchsia-700", `🧠 ${t("tab_ltm_full")} (${rows.length})`));
  const btn = (label, cls) => h("button", "text-[11px] border rounded px-2 py-0.5 hover:bg-slate-100 " + (cls || ""), label);
  const bRefresh = btn("↻ " + t("ltm_refresh")); bRefresh.onclick = refreshLTM; head.appendChild(bRefresh);
  const fmtSel = h("select", "text-[11px] border rounded px-1 py-0.5");
  ["md", "json", "sql"].forEach((f) => { const o = h("option", "", f); o.value = f; fmtSel.appendChild(o); });
  head.appendChild(fmtSel);
  const bExport = btn("⬇ " + t("ltm_export")); bExport.onclick = () => ltmExport(fmtSel.value); head.appendChild(bExport);
  const bImport = btn("⬆ " + t("ltm_import")); bImport.onclick = ltmImportFile; head.appendChild(bImport);
  const bWipe = btn("🗑 " + t("ltm_wipe"), "border-rose-300 text-rose-600 hover:bg-rose-50"); bWipe.onclick = ltmWipe; head.appendChild(bWipe);
  root.appendChild(head);
  if (!rows.length) {
    root.appendChild(h("div", "text-xs text-slate-400 italic p-4 leading-relaxed", t("ltm_empty")));
    return;
  }
  const tbl = h("div", "border rounded-lg overflow-hidden bg-white");
  rows.forEach((r) => {
    const row = h("div", "flex items-center gap-2 px-2 py-1.5 border-b text-[11px]");
    row.appendChild(h("span", `px-1.5 py-0.5 rounded text-[9px] font-bold ${LTM_CAT_CHIP[r.category] || "bg-slate-100 text-slate-600"}`, esc(r.category || "—")));
    const c = h("span", "flex-1 truncate", esc(r.content)); c.title = r.content; row.appendChild(c);
    row.appendChild(h("span", "text-slate-400", `c${r.confidence}`));
    row.appendChild(h("span", "text-slate-300 text-[10px] mono", esc((r.created_at || "").slice(0, 10))));
    const del = h("button", "text-rose-500 hover:text-rose-700 text-[10px] shrink-0", t("ltm_delete"));
    del.onclick = () => ltmDelete(r.id);
    row.appendChild(del);
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
  hideDRBanner(); setThinking(true, t("dr_starting"));
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

/* ---------------- LLM-4 VISUALIZER (v1.12 B4) ---------------- */
// Lifecycle: the reply/DR text carries ⟦viz:id⟧ placeholders (they survive
// markdown). spliceVizSlots swaps each for a loading .viz-slot; a viz.rendered
// event mounts a SANDBOXED iframe (sandbox="allow-scripts" ONLY — opaque origin,
// no same-origin/network/storage). The artifact posts {sherlockViz:'ready'[,
// height]} when painted and {sherlockViz:'error',message} from window.onerror; a
// ~4s runtime harness validates the paint before the frame is un-hidden. A
// runtime failure/timeout round-trips the current HTML to /api/viz/repair (≤2
// client attempts); an exhausted/failed/orphaned viz degrades to a muted note —
// never a broken iframe, never a crash. The host page is NEVER innerHTML'd the
// artifact HTML — it only ever reaches the DOM as an iframe srcdoc.
const VIZ_TOKEN_RE = /⟦viz:([A-Za-z0-9._:-]{1,80})⟧/g;
const VIZ_DEF_H = 360, VIZ_MAX_H = 640, VIZ_HARNESS_MS = 4000, VIZ_ORPHAN_MS = 40000, VIZ_CLIENT_REPAIRS = 2;

// Bounded (≈50) KEY→element registry so async viz events find the right bubble
// AFTER it finalized (chat: "t{turn}", deep research: "dr:{research_id}").
function registerBubble(key, el) {
  if (!key || !el) return el;
  if (S.bubbles[key] && S.bubbles[key] !== el) delete S.bubbles[key]; // move to newest slot
  S.bubbles[key] = el;
  const keys = Object.keys(S.bubbles);
  if (keys.length > 50) delete S.bubbles[keys[0]];
  return el;
}
const cssEscId = (id) => (window.CSS && CSS.escape ? CSS.escape(id) : String(id).replace(/["\\\]]/g, "\\$&"));

// Replace EXACT ⟦viz:id⟧ tokens (only that precise pattern — model content can't
// forge one) in a container's text nodes with loading slots, then apply any viz
// events that arrived before the slot existed.
function spliceVizSlots(container) {
  if (!container) return;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
  const targets = [];
  let node;
  while ((node = walker.nextNode())) { VIZ_TOKEN_RE.lastIndex = 0; if (VIZ_TOKEN_RE.test(node.nodeValue)) targets.push(node); }
  targets.forEach(replaceTokenTextNode);
  container.querySelectorAll(".viz-slot[data-viz-id]").forEach((slot) => {
    const buf = S.vizBuffer[slot.dataset.vizId];
    if (buf) { delete S.vizBuffer[slot.dataset.vizId]; applyVizEvent(slot, buf); }
  });
}
function replaceTokenTextNode(textNode) {
  const text = textNode.nodeValue, frag = document.createDocumentFragment();
  let last = 0, m; VIZ_TOKEN_RE.lastIndex = 0;
  while ((m = VIZ_TOKEN_RE.exec(text))) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    frag.appendChild(makeVizSlot(m[1]));
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  textNode.parentNode.replaceChild(frag, textNode);
}
function makeVizSlot(id) {
  const slot = h("div", "viz-slot");
  slot.dataset.vizId = id;
  setVizLoading(slot, t("viz_loading"));
  if (!S.vizStatus[id]) S.vizStatus[id] = "pending";
  // orphan sweep: a placeholder that never gets a rendered/failed (dropped job /
  // over cap) degrades to the muted note after a global timeout.
  setTimeout(() => {
    const st = S.vizStatus[id];
    if (st == null || st === "pending") { S.vizStatus[id] = "failed"; degradeVizSlot(slot); }
    // duplicate slot for a shared viz_id: the real slot already settled the shared
    // status, but THIS slot never got an iframe → degrade it so it can't spin forever.
    else if (slot.dataset.state === "loading" && !slot.querySelector("iframe")) degradeVizSlot(slot);
  }, VIZ_ORPHAN_MS);
  return slot;
}
function setVizLoading(slot, text) {
  slot.dataset.state = "loading";
  slot.className = "viz-slot viz-slot-loading";
  slot.innerHTML = "";
  slot.appendChild(h("span", "viz-spin"));
  slot.appendChild(h("span", "viz-loading-text", esc(text)));
}
function degradeVizSlot(slot) {
  if (!slot) return;
  slot.dataset.state = "failed";
  slot.className = "viz-slot viz-slot-failed";
  slot.textContent = t("viz_unavailable");
}
// Prefer the registered bubble (by turn / research_id), then a global chat scan
// (runtime re-renders from /api/viz/repair carry neither turn nor research_id).
function findVizSlot(d) {
  const vid = d && d.viz_id;
  if (!vid) return null;
  const sel = '.viz-slot[data-viz-id="' + cssEscId(vid) + '"]';
  let host = null;
  if (d.research_id != null) host = S.bubbles["dr:" + d.research_id];
  else if (d.turn != null) host = S.bubbles["t" + d.turn];
  const slot = host && host.querySelector(sel);
  if (slot) return slot;
  const chat = $("chat");
  return chat ? chat.querySelector(sel) : null;
}

/* ---- viz.* event entry points ---- */
// Buffer a viz event whose slot hasn't appeared yet, capped (≈20) so a stream of
// buffered-but-never-mounted records can't retain up to 64KB html each forever.
function bufferViz(id, rec) { S.vizBuffer[id] = rec; const bk = Object.keys(S.vizBuffer); if (bk.length > 20) delete S.vizBuffer[bk[0]]; }
function onVizPending(d) { if (d && d.viz_id && !S.vizStatus[d.viz_id]) S.vizStatus[d.viz_id] = "pending"; }
function onVizRendered(d) {
  const id = d && d.viz_id;
  if (!id) return;
  // already mounted / a client repair is driving it → ignore the redundant WS echo
  if (S.vizStatus[id] === "ready" || S.vizStatus[id] === "harnessing") return;
  const slot = findVizSlot(d);
  if (slot) mountVizIframe(slot, d);
  else bufferViz(id, { kind: "rendered", data: d }); // arrived before its slot
}
function onVizRepairing(d) {
  const id = d && d.viz_id;
  if (!id || S.vizStatus[id] === "ready" || S.vizStatus[id] === "failed") return;
  const slot = findVizSlot(d);
  if (slot) setVizLoading(slot, t("viz_repairing", d.round || "?"));
  else bufferViz(id, { kind: "repairing", data: d });
}
function onVizFailed(d) {
  const id = d && d.viz_id;
  if (!id || S.vizStatus[id] === "ready") return;
  S.vizStatus[id] = "failed";
  const slot = findVizSlot(d);
  if (slot) degradeVizSlot(slot);
  else bufferViz(id, { kind: "failed", data: d });
}
function applyVizEvent(slot, buf) {
  if (!buf) return;
  if (buf.kind === "rendered") mountVizIframe(slot, buf.data);
  else if (buf.kind === "repairing") setVizLoading(slot, t("viz_repairing", buf.data.round || "?"));
  else if (buf.kind === "failed") { S.vizStatus[slot.dataset.vizId] = "failed"; degradeVizSlot(slot); }
}

/* ---- sandboxed iframe mount + runtime validation harness ---- */
// v1.12 Stage V2 (audit fix): the artifact card themes itself via
// prefers-color-scheme, but a sandbox="allow-scripts" srcdoc frame (opaque
// origin) cannot see the page's manual .dark class — the ONLY web mechanism
// that drives the frame's preferred scheme is the embedder's color-scheme on
// the <iframe> element itself. Stamp it at mount and reveal, and re-stamp every
// mounted frame when the toggle flips; the reveal background follows the same
// theme so a dark card never sits in a white box (and vice versa).
function vizFrameScheme() { return document.documentElement.classList.contains("dark") ? "dark" : "light"; }
function vizFrameBg() { return vizFrameScheme() === "dark" ? "#111827" : "#fff"; }
function syncVizFrameTheme() {
  const scheme = vizFrameScheme(), bg = vizFrameBg();
  document.querySelectorAll('iframe[title^="visualization"]').forEach((f) => {
    f.style.colorScheme = scheme;
    if (f.style.background) f.style.background = bg; // only revealed frames carry a bg
  });
}
let _vizMsgWired = false;
function ensureVizMessageListener() {
  if (_vizMsgWired) return;
  _vizMsgWired = true;
  // opaque-origin frames post with event.origin === "null", so correlate by
  // event.source === the iframe's contentWindow (not by origin).
  window.addEventListener("message", (e) => {
    if (!e.source) return; // a detached iframe has contentWindow===null; never match on null
    const data = e.data;
    if (!data || typeof data !== "object" || !data.sherlockViz) return;
    for (const hh of S.vizHarnesses) { if (hh.iframe.contentWindow === e.source) { hh.onMessage(data); return; } }
  });
}
function mountVizIframe(slot, d) {
  const id = slot.dataset.vizId;
  if (!d || !d.html) { S.vizStatus[id] = "failed"; degradeVizSlot(slot); return; }
  S.vizStatus[id] = "harnessing";
  startVizHarness(slot, d, d.html, 0);
}
function startVizHarness(slot, d, html, attempt) {
  ensureVizMessageListener();
  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "allow-scripts"); // NO allow-same-origin → opaque origin
  iframe.setAttribute("referrerpolicy", "no-referrer");
  iframe.setAttribute("title", "visualization " + (d.viz_id || ""));
  // overlay the loading placeholder INVISIBLY and in-flow-width (slot is
  // position:relative) so the artifact measures its real width; revealed only
  // once it posts {sherlockViz:'ready'}.
  iframe.style.cssText = "position:absolute;top:0;left:0;width:100%;height:" + VIZ_DEF_H + "px;border:0;visibility:hidden";
  iframe.style.colorScheme = vizFrameScheme(); // theme the frame from first paint
  iframe.srcdoc = html; // NEVER innerHTML — the artifact only runs inside the sandbox

  let settled = false;
  const harness = { iframe, height: 0 };
  const timer = setTimeout(() => settle("timeout"), VIZ_HARNESS_MS);
  harness.timer = timer; // so resetChatUI can cancel an in-flight settle across sessions
  function settle(kind, msg) {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    S.vizHarnesses.delete(harness);
    if (kind === "ready") { revealVizIframe(slot, iframe, harness.height); return; }
    try { if (iframe.parentNode) iframe.parentNode.removeChild(iframe); } catch (e) {}
    const err = kind === "timeout"
      ? "runtime timeout: no ready signal within " + VIZ_HARNESS_MS / 1000 + "s"
      : msg || "runtime error";
    repairViz(slot, d, html, attempt, err);
  }
  harness.onMessage = (data) => {
    if (data.sherlockViz === "ready") {
      const hgt = Number(data.height);
      if (hgt > 0) harness.height = Math.max(120, Math.min(hgt + 8, VIZ_MAX_H));
      settle("ready");
    } else if (data.sherlockViz === "error") {
      settle("error", String(data.message || "").slice(0, 300));
    }
  };
  S.vizHarnesses.add(harness);
  slot.appendChild(iframe); // keep the loading children visible beneath the hidden frame
}
function revealVizIframe(slot, iframe, height) {
  S.vizStatus[slot.dataset.vizId] = "ready";
  slot.dataset.state = "ready";
  slot.className = "viz-slot viz-slot-ready";
  [].slice.call(slot.childNodes).forEach((n) => { if (n !== iframe) slot.removeChild(n); });
  const hgt = height || VIZ_DEF_H;
  // background + color-scheme track the app's dark toggle (cssText wipes the
  // mount-time colorScheme, so re-stamp it after).
  iframe.style.cssText = "position:static;display:block;width:100%;height:" + hgt + "px;max-height:" + VIZ_MAX_H + "px;border:0;border-radius:10px;background:" + vizFrameBg() + ";overflow:auto";
  iframe.style.colorScheme = vizFrameScheme();
  autoScroll();
}
function repairViz(slot, d, html, attempt, errorMsg) {
  const id = slot.dataset.vizId;
  setVizLoading(slot, t("viz_repairing_runtime"));
  // client attempt cap (≤2), independent of the server's authoritative per-viz cap
  if (attempt >= VIZ_CLIENT_REPAIRS || !S.sid) { S.vizStatus[id] = "failed"; degradeVizSlot(slot); return; }
  fetch("/api/viz/repair", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, viz_id: id, html: html, error: errorMsg }), signal: AbortSignal.timeout(30000) })
    .then((r) => r.json()).catch(() => ({ ok: false }))
    .then((j) => {
      if (j && j.ok && j.html) startVizHarness(slot, Object.assign({}, d, { html: j.html }), j.html, attempt + 1);
      else { S.vizStatus[id] = "failed"; degradeVizSlot(slot); } // ok:false / exhausted / error
    });
}

/* ---------------- P1: ⚙ mid-session per-role model panel ---------------- */
// The same provider+model picks as setup, changeable mid-session; each change
// POSTs /api/select_models {role:{provider,model}} (applies from the NEXT turn)
// and mirrors into the top-bar live select so currentModels() stays consistent.
const PANEL_ROLES = [
  ["main", "role_main", "LLM-1", "liveMain"],
  ["summary", "role_summary", "LLM-2", "liveSummary"],
  ["inference", "role_infer", "LLM-3", "liveInference"],
  ["viz", "viz_role", "LLM-4", "liveViz"],
];
// The per-provider model list cached at setup; fetched on demand if missing.
async function providerModels(p) {
  const info = S.prov[p];
  if (!info) return [];
  if ((info.models || []).length) return info.models;
  try {
    const r = await fetch("/api/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: p, ...info.creds }) });
    const j = await r.json();
    if (!j.error && (j.models || []).length) { info.models = j.models; return j.models; }
  } catch (e) { /* fall through to empty */ }
  return [];
}
function buildModelsPanel() {
  const root = $("modelsPanelRows"); if (!root) return;
  root.innerHTML = "";
  PANEL_ROLES.forEach(([role, labelKey, llmN, liveId]) => {
    const cur = parseSpec($(liveId) && $(liveId).value); // null = viz "same as Main"
    const row = h("div", "flex items-center gap-1.5");
    row.appendChild(h("span", "w-24 shrink-0 text-[11px] font-semibold", `${esc(t(labelKey))} · ${llmN}`));
    const provSel = h("select", "border rounded px-1 py-0.5 text-[11px] mono w-24 shrink-0");
    Object.keys(S.prov).forEach((p) => { const o = h("option", "", esc(PROV_LABEL[p] || p)); o.value = p; provSel.appendChild(o); });
    const modSel = h("select", "border rounded px-1 py-0.5 text-[11px] mono flex-1 min-w-0");
    const ok = h("span", "w-4 text-green-500 shrink-0", "");
    row.appendChild(provSel); row.appendChild(modSel); row.appendChild(ok);
    root.appendChild(row);
    const fillModels = async (prov, keep) => {
      modSel.innerHTML = "";
      if (role === "viz") { const o = h("option", "", esc(t("viz_same_as_main"))); o.value = ""; modSel.appendChild(o); }
      (await providerModels(prov)).forEach((m) => { const o = h("option", "", esc(m.id)); o.value = m.id; modSel.appendChild(o); });
      if (keep != null && [...modSel.options].some((o) => o.value === keep)) modSel.value = keep;
    };
    const apply = async () => {
      const spec = modSel.value ? { provider: provSel.value, model: modSel.value } : null;
      try {
        await fetch("/api/select_models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, models: { [role]: spec } }) });
      } catch (e) { return; }
      // mirror into the live top-bar select (its options cover every connected model)
      const live = $(liveId);
      if (live) { const v = spec ? `${spec.provider}::${spec.model}` : ""; if ([...live.options].some((o) => o.value === v)) live.value = v; }
      ok.textContent = "✓"; setTimeout(() => { ok.textContent = ""; }, 1200);
    };
    provSel.onchange = async () => {
      await fillModels(provSel.value);
      // a provider hop implies a model hop: default to its first REAL model
      if (modSel.options.length) modSel.selectedIndex = role === "viz" && modSel.options.length > 1 ? 1 : 0;
      apply();
    };
    modSel.onchange = apply;
    const prov0 = cur && cur.provider && S.prov[cur.provider] ? cur.provider : Object.keys(S.prov)[0];
    if (prov0) { provSel.value = prov0; fillModels(prov0, cur ? cur.model : role === "viz" ? "" : null); }
  });
}
if ($("modelsBtn")) $("modelsBtn").onclick = () => {
  const p = $("modelsPanel"); if (!p) return;
  const show = p.classList.contains("hidden");
  if (show) buildModelsPanel(); // rebuilt on every open → fresh current picks
  p.classList.toggle("hidden", !show);
};
// click-away closes the panel (the ⚙ button's own handler toggled it above)
document.addEventListener("click", (e) => {
  const p = $("modelsPanel"), b = $("modelsBtn");
  if (!p || p.classList.contains("hidden")) return;
  if (p.contains(e.target) || (b && b.contains(e.target))) return;
  p.classList.add("hidden");
});

/* ---------------- H2: history sidebar (🗂) ---------------- */
// Conversations persisted in this session's (profile) store. Collapsed by
// default on narrow screens; the open/closed choice sticks in localStorage.
let histOpen = (() => {
  try { const s = localStorage.getItem("sherlock_hist_open"); if (s != null) return s === "1"; } catch (e) {}
  return window.innerWidth >= 1024;
})();
function applyHistOpen() {
  const sb = $("histSidebar"); if (!sb) return;
  sb.classList.toggle("hidden", !histOpen);
  sb.classList.toggle("flex", histOpen);
  if (histOpen) refreshHistory();
}
if ($("histToggle")) $("histToggle").onclick = () => {
  histOpen = !histOpen;
  try { localStorage.setItem("sherlock_hist_open", histOpen ? "1" : "0"); } catch (e) {}
  applyHistOpen();
};
async function refreshHistory() {
  if (!S.sid || !histOpen) return;
  // audit: a turn.done refresh mid-rename would destroy the editor (and the
  // typed text) — skip; the rename's own commit() refreshes when it closes.
  if ($("histList") && $("histList").querySelector("input")) return;
  try {
    const r = await fetch(`/api/history?session_id=${encodeURIComponent(S.sid)}`);
    const j = await r.json().catch(() => ({}));
    if (j && !j.error) renderHistory(j.conversations || []);
  } catch (e) { /* best-effort — the list keeps its last render */ }
}
function histDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  return d.toDateString() === new Date().toDateString()
    ? d.toTimeString().slice(0, 5)             // today → HH:MM
    : `${d.getMonth() + 1}/${d.getDate()}`;    // else → M/D
}
function renderHistory(convs) {
  const list = $("histList"); if (!list) return;
  list.innerHTML = "";
  if (!convs.length) { list.appendChild(h("div", "text-xs text-slate-400 italic p-3 text-center", esc(t("hist_empty")))); return; }
  convs.forEach((c) => {
    const item = h("div", "hist-item" + (c.active ? " active" : ""));
    const titleEl = h("div", "text-xs font-medium truncate", esc(c.title || String(c.id).slice(0, 8)));
    titleEl.title = `${c.title || ""} — ${t("hist_rename")}: 2×click`;
    const meta = h("div", "text-[10px] text-slate-400", esc(`${c.messages != null ? c.messages + " · " : ""}${histDate(c.created_at)}`));
    item.appendChild(titleEl); item.appendChild(meta);
    // Delay the single-click open so a double-click (rename) on a non-active
    // item doesn't first switch to it and rebuild the list under the editor.
    let clickTimer = null;
    item.onclick = () => {
      if (c.active) return;
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => openConversation(c.id), 250);
    };
    titleEl.ondblclick = (e) => { e.stopPropagation(); clearTimeout(clickTimer); startHistRename(c, titleEl); };
    list.appendChild(item);
  });
}
// Double-click a title → inline rename → POST /api/history/title.
function startHistRename(c, titleEl) {
  const inp = document.createElement("input");
  inp.type = "text"; inp.value = c.title || "";
  inp.className = "w-full border rounded px-1 py-0.5 text-xs";
  inp.onclick = (e) => e.stopPropagation(); // don't open the conversation
  titleEl.replaceWith(inp);
  inp.focus(); inp.select();
  let done = false;
  const commit = async (save) => {
    if (done) return; done = true;
    const v = inp.value.trim();
    if (save && v && v !== c.title) {
      try {
        const r = await fetch("/api/history/title", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, conversation_id: c.id, title: v }) });
        const j = await r.json().catch(() => ({}));
        if (j.error) addBubble("system", "✗ " + j.error);
      } catch (e) { addBubble("system", "✗ " + e); }
    }
    refreshHistory();
  };
  inp.onkeydown = (e) => {
    if (e.isComposing || e.keyCode === 229) return; // IME composition — never submit
    if (e.key === "Enter") { e.preventDefault(); commit(true); }
    else if (e.key === "Escape") commit(false);
  };
  inp.onblur = () => commit(true);
}
if ($("histNew")) $("histNew").onclick = async () => {
  if (!S.sid || S.busy) return;
  try {
    const r = await fetch("/api/history/new", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid }) });
    const j = await r.json().catch(() => ({}));
    if (j.error) { addBubble("system", "✗ " + j.error); return; }
    clearChatPane(); // same agent/session — only the pane restarts
    refreshHistory();
  } catch (e) { addBubble("system", "✗ " + e); }
};
// Open a persisted conversation: switch it active server-side, rebuild the chat
// pane from its messages (the SAME markdown path as live replies), then
// re-hydrate every ⟦viz:…⟧ placeholder from the persisted artifacts.
let _histOpening = false;
async function openConversation(cid) {
  if (!S.sid || S.busy) return; // never yank the pane out from under a live turn
  if (_histOpening) return; // audit: serialise opens — concurrent switch_session races
  _histOpening = true;
  let j;
  try {
    const r = await fetch("/api/history/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: S.sid, conversation_id: cid }) });
    j = await r.json().catch(() => ({}));
  } catch (e) { addBubble("system", "✗ " + e); _histOpening = false; return; }
  if (!j || j.error || !j.ok) { addBubble("system", `✗ ${t("hist_open_fail")}: ${(j && j.error) || "?"}`); _histOpening = false; return; }
  clearChatPane(); // chat pane only — the session, WS and system panels stay
  (j.messages || []).forEach((m) => {
    // addBubble renders assistant content through the same marked+DOMPurify
    // renderer as live replies AND splices ⟦viz:id⟧ placeholders into slots.
    addBubble(m.role === "user" ? "user" : "assistant", m.content || "");
  });
  rehydrateVizSlots(cid);
  refreshHistory(); // move the active highlight
  _histOpening = false;
}
// For each restored slot, fetch the persisted artifact and mount it through the
// SAME sandboxed harness as live renders — at the FINAL client attempt number,
// so a runtime failure degrades to the plain chip instead of burning LLM repair
// rounds. A non-HTML ({error:…} JSON / non-200) response degrades immediately.
function rehydrateVizSlots(cid) {
  $("chat").querySelectorAll(".viz-slot[data-viz-id]").forEach((slot) => {
    const id = slot.dataset.vizId;
    const st = S.vizStatus[id];
    if (st === "ready" || st === "harnessing" || st === "failed") return;
    // audit (P1): viz ids collide across conversations (t1-1 in each) — pass the
    // OPENED conversation so the server resolves inside its subdir only.
    fetch(`/api/viz/${encodeURIComponent(id)}?session_id=${encodeURIComponent(S.sid)}&conv=${encodeURIComponent(cid || "")}`)
      .then((r) => {
        const ct = r.headers.get("content-type") || "";
        if (!r.ok || ct.indexOf("text/html") === -1) throw new Error("unavailable");
        return r.text();
      })
      .then((html) => {
        if (!html) throw new Error("empty");
        S.vizStatus[id] = "harnessing";
        startVizHarness(slot, { viz_id: id }, html, VIZ_CLIENT_REPAIRS);
      })
      .catch(() => { S.vizStatus[id] = "failed"; degradeVizSlot(slot); });
  });
}
applyHistOpen();

/* ---------------- tabs ---------------- */
const TABS = ["flow", "slot", "llmio", "infer", "compact", "memory", "ltm", "carry", "research"];
document.querySelectorAll(".tabbtn").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  TABS.forEach((x) => $("tab-" + x).classList.toggle("hidden", x !== t));
  document.querySelectorAll(".tabbtn").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  if (t === "ltm") refreshLTM(); // fetch the freshest long-term snapshot on open
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
  hint("tab-ltm", "🧠 <b>Long-term memory</b> — durable facts promoted across sessions. Turn it on in the top bar; export/import/wipe here.");
  hint("tab-carry", "↪ <b>Pending hypotheses</b> that seed the next turn's slot appear here.");
  hint("tab-research", "🔬 <b>Deep research</b> — when LLM-1 proposes it and you approve, each round (search → read → meta-question Q&amp;A) streams here and is saved as a session document.");
  hint("tab-slot", "🧱 The assembled <b>LLM-1 context</b> (TIER 1–4) + token budget appears here each turn.");
  hint("tab-llmio", "💬 The exact <b>prompts + responses</b> for LLM-1 / LLM-2 / LLM-3 appear here.");
}
