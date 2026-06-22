# Sherlock

*다른 언어로 보기: [English](README.md) · **한국어***

**Sherlock는 어떤 LLM에도 붙일 수 있는 컨텍스트 레이어입니다.** 단 하나의 채팅 함수만 감싸면 됩니다 —
모델도, 스택도 그대로 유지하면서 영속적 메모리, 압축된
컨텍스트, 암묵적 의도 추론, 다국어 심층 리서치, 그리고
**모델이 정확히 무엇을 보았고 왜 그랬는지**를 보여주는 라이브 인스펙터를 얻을 수 있습니다.

```bash
pip install sherlock-context
```

```python
from sherlock import Sherlock
agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="...")
agent.chat("hi")   # that's the whole integration
```

Sherlock는 모델을 대신 고르지 않으며, 비용을 아끼겠다고 결과를 잘라내지도
않습니다. **모델 선택은 당신의 몫, 컨텍스트는 Sherlock의 몫입니다.** 토큰
절감은 오직 *낭비*(재전송된 자료, 중복된 컨텍스트, 유실된 호출)를 없애는 데서만
나오며 — 돌려받는 결과의 양에 상한을 두는 방식은 결코 아닙니다.

> **권장 설정.** Sherlock는 *어떤* 모델로도 동작하지만, 에이전트 기능 —
> 도구 기반 웹 검색, 다단계 추론, 심층 리서치 — 이 제대로 빛을 발하려면 충분히
> 유능한 모델이 필요합니다. 이런 용도에는 **파라미터 약 20B 이상, 컨텍스트
> 윈도우 64k 토큰 이상**의 모델을 쓰세요. 더 작은 모델도 메모리·암시적 의도
> 레이어의 이점은 받지만, 실시간 데이터·추론이 많이 필요한 질문에서는 헤매는
> 경향이 있습니다(검색은 하지만 그 결과를 제대로 된 답으로 못 만듭니다). 검색은
> 기본 포함된 **DuckDuckGo가 키 없이 바로 되지만 뉴스·실시간 데이터에는 정말
> 약합니다** — 답이 최신 사실에 좌우될 때는 **Brave, Tavily, Valyu**(각각 API 키만
> 있으면 됨) 중 하나를 권장합니다. DuckDuckGo는 키 없이 쓰는 데모용 기본값 정도로
> 생각하세요.

## Sherlock가 빛을 발할 때 (그리고 그렇지 않을 때)

Sherlock는 마법도 아니고 공짜도 아닙니다 — 백그라운드 LLM 작업을 추가합니다. 자체
A/B 테스트(같은 모델, 같은 프롬프트, Sherlock *적용* vs *미적용*, 독립적인 LLM 심판이 채점)에서
다음 네 가지 상황에서 제값을 합니다:

- **행간에 진짜 요구가 숨어 있는 짧은 입력.** 생략적이거나,
  함의가 담겼거나, 명세가 부족한 메시지에서 LLM-3는 *암묵적* 의도를 읽어
  앞으로 전달합니다 — 심판은 "사용자가 실제로 의도한 바를 파악했는가"에서 Sherlock에
  뚜렷하게 높은 점수를 주었고(여러 라운드 평균 ≈8.7 vs ≈7.3 / 10),
  맨몸 모델이 표면만 답하는 사이 Sherlock는 라운드를 완승했습니다.
- **컨텍스트 윈도우를 넘어서는 대화.** 오래된 턴이
  프롬프트에서 압축되어 빠지면 맨몸 모델은 그냥 잊어버립니다; Sherlock는 여전히
  고정된 사실 + 요약을 기억하고, 베이스라인이 답하지 못하는 곳에서 정확히
  답합니다.
- **작은 / 로컬 모델.** 핵심 베팅은 모델에게 *이미 갖고 있지 않던, 참이며
  상호 보완적인 무언가*를 먹이는 것이므로, 7~8B 로컬 모델이 추측에 의존하는 대신
  덩치 이상의 실력을 발휘합니다.
- **진짜 리서치 질문.** 승인을 거치는, 여러 라운드의, 다국어
  검색에 출처 삼각 검증 + 인용이 더해져 순진한 단일
  검색 한 번보다 훨씬 깊이 들어갑니다.

**무승부가 나는 곳 — 우리는 솔직히 말합니다:** 전체 대화가 이미 컨텍스트에 다 들어가는
강력한 모델에 던지는 짧고 단발성의 사실 질문입니다. 기억할 것도 없고,
숨은 의도도 없고, 리서치도 없는 경우 → Sherlock는 지연 시간과 약간의
백그라운드 토큰을 추가하면서도 품질 향상은 거의 없습니다. 그럴 때는 `companions_mode="off"`를 쓰세요(혹은
아예 Sherlock를 꺼내지 마세요). 플레이그라운드의 **A/B 모드**가 존재하는 이유가 바로
이것입니다 — 도입을 결정하기 전에 *당신의* 워크로드에서 직접 측정할 수 있도록 말이죠.

## 언제 Sherlock를 써야 하는가

다음이 필요한 어시스턴트를 만들고 있다면 Sherlock를 쓰세요:

- 긴 대화 전반에 걸쳐 **사용자를 기억** — 그러면서도 턴당
  프롬프트 크기를 한정된 상태로 유지(압축 프런티어: 오래된 턴은 프롬프트를
  떠나되 데이터베이스를 떠나지는 않습니다);
- 공급자를 바꾸지 않고 **작은 / 로컬 모델을 더 똑똑하게 행동하게** (Ollama, LM Studio, vLLM,
  llama.cpp) — 평문 태그 프로토콜, 정직한
  8K/16K/32K 컨텍스트 예산, JSON-수리 재시도;
- 어떤 턴에서든 **모델이 정확히 어떤 컨텍스트를 보았는지 디버깅** — 라이브
  인스펙터 + 원클릭 세션 내보내기, 그리고 같은 프롬프트를 같은 모델에 *Sherlock 없이*
  돌려보는 내장 A/B 모드;
- **진짜 리서치 수행**: 계획된 다국어 검색, 출처 삼각 검증,
  인용 검증, 승인 게이트가 있는 심층 리서치.

## Sherlock가 어디에 들어맞는가

| 필요한 것 | 가장 잘 맞는 도구 |
|---|---|
| 완전한 상태 보존형 에이전트 런타임 / 플랫폼 | Letta |
| 곧바로 쓰는 매니지드 메모리 API | Mem0 / Supermemory |
| 엔터프라이즈 시간축 지식 그래프 메모리 | Zep / Graphiti |
| LangGraph 네이티브 메모리 | LangMem |
| **BYO-LLM 컨텍스트 조립 + 라이브 컨텍스트 인스펙터, 프레임워크 종속 없음** | **Sherlock** |

Sherlock는 호스팅 메모리 API의 편리함이나 에이전트 런타임의 폭으로 경쟁하지
않도록 의도적으로 설계되었습니다 — 당신만의 콜러블을 유지하면서 모델이 받는 컨텍스트의
모든 토큰을 *보고*(그리고 디버깅하고) 싶을 때 그 진가를 발휘합니다.

## 작동 원리

세 가지 LLM 역할을 모두 당신이 연결합니다(하나의 함수일 수도, 셋일 수도 있습니다):

```
            ┌────────────────────────────────────────────────┐
 user ──────►  LLM-1 · main chat                              │
            │  answers using the assembled context slot:      │
            │  system msg: your prompt + protocol + pinned    │
            │     facts · persona · highlights  ── cached ──┐  │
            │  + prior conversation (verbatim turns) ──cached┘ │
            │  + final user msg: THIS-turn hypotheses ·       │
            │     fresh search · the user's question (volatile)│
            └───────┬────────────────────────────┬───────────┘
                    │ background                  │ background
            ┌───────▼──────────┐         ┌───────▼──────────┐
            │ LLM-2 · compactor │         │ LLM-3 · inferrer │
            │ summaries, facts, │         │ ≥3 hypotheses on │
            │ persona profile   │         │ the real ask;    │
            │ (pin/active/drop) │         │ search planning  │
            └───────┬──────────┘         └───────┬──────────┘
                    └────────► memory ◄──────────┘
                      SQLite + vector store · decay
                      (fresh → warm → cold → forgotten)
```

- **출처를 추적하는 메모리** — 사용자가 말한 사실은 시스템 추론과 절대
  혼동되지 않습니다; 각 프롬프트 블록은 출처와 그것이 학습된 턴을
  함께 담으므로(`(user t12)`) 충돌 시 더 최신 사실이 이깁니다.
- **슬롯 예산** — 컨텍스트는 계층별 실제 토큰 상한에
  맞춰 조립됩니다; 원본 턴 꼬리는 턴 단위로만 가져갑니다(생각이 중간에
  잘리는 일은 없습니다).
- **태그 프로토콜** — 당신의 LLM이 평문 태그로 모든 것을 구동합니다
  (`<<sherlock-companions: …>>`, `<<sherlock-tool: …>>`); 네이티브
  함수 호출이 필요 없는데, 이것이야말로 작은 모델이 가장 잘 다루는
  방식입니다. 네이티브 도구 호출 어댑터도 존재합니다.
- **심층 리서치** — 승인 게이트가 있는, 다국어의, 여러 라운드 웹
  리서치 루프이며 토큰을 아끼는 공유 상태 프로토콜을 씁니다(자세한 내용은
  아래에).

## 설치

```bash
pip install sherlock-context                       # base — incl. free DuckDuckGo search + page fetch
pip install "sherlock-context[embeddings]"         # + real local semantic memory (recommended)
pip install "sherlock-context[embeddings,search]"  # + Tavily provider (Brave/Valyu need only a key)
pip install "sherlock-context[playground]"         # + the Live Inspector web app
```

> 배포 이름은 **`sherlock-context`**입니다(PyPI의 `sherlock`은
> 무관한 잠금 라이브러리입니다); 임포트는 그대로 `import sherlock`입니다.

소스에서 최신 버전 받기(PyPI 불필요):

```bash
pip install "git+https://github.com/MinwooKim1990/sherlock.git"
pip install "sherlock-context[embeddings,playground] @ git+https://github.com/MinwooKim1990/sherlock.git"
```

또는 체크아웃에서 개발하기:

```bash
git clone https://github.com/MinwooKim1990/sherlock.git && cd sherlock
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,playground]"
```

Python 3.12를 대상으로 합니다(3.11 / 3.13도 동작합니다). `litellm`은 지연
임포트되므로 `import sherlock`는 빠르게 유지됩니다. 임베딩 기본값은
`"auto"`입니다: `[embeddings]` 엑스트라가 설치되어 있으면 진짜 로컬 임베딩
(fastembed, 다국어, API 키 불필요)을 쓰고, 그렇지 않으면 결정론적
해시 임베더로 우아하게 폴백합니다(경고와 함께). DuckDuckGo
검색 + 페이지 가져오기는 기본 설치에서 동작합니다.

## 빠른 시작 (30초)

```python
from sherlock import Sherlock

def my_llm(messages):
    """Receive list of {"role": ..., "content": ...}; return text."""
    import anthropic
    client = anthropic.Anthropic()
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(
        model="claude-haiku-4-5", max_tokens=2048, system=sys, messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="You are a candid, casual assistant.",
)

print(agent.chat("hi"))
print(agent.chat("what did i just say?"))   # Sherlock will have the history
```

이게 전부입니다. Sherlock가 턴별 메시지 저장소(SQLite),
백그라운드 압축(LLM-2), Sherlock 스타일 추론(LLM-3 — 표면적 의미 ≠
실제 요구일 때마다 사용자의 근본 요구에 대한 가설 ≥3개),
출처 추적, 메모리 감쇠를 처리합니다.

### 역할별로 다른 모델 사용하기

```python
def chat_via_main(messages): ...        # e.g. a strong model for user-facing replies
def chat_via_companion(messages): ...   # e.g. a small/cheap model for compaction + inference

agent = Sherlock.with_callable(
    main_chat=chat_via_main,
    summary_chat=chat_via_companion,
    inference_chat=chat_via_companion,
    system_prompt="You are a helpful assistant.",
)
```

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

def my_llm(messages):
    r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return r.choices[0].message.content

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="You are concise.")
```

### Ollama / 임의의 로컬 모델

```python
import requests

def my_llm(messages):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3", "messages": messages, "stream": False},
        timeout=120,
    ).json()
    return r["message"]["content"]

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="…")
```

### 비동기

```python
import anthropic
aio = anthropic.AsyncAnthropic()

async def my_llm(messages):
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = await aio.messages.create(
        model="claude-haiku-4-5", max_tokens=2048, system=sys, messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="…")
agent.chat("hi")          # sync entry point runs the async fn under the hood
# await agent.achat("…")  # native async entry point
```

LLM-1은 동기적으로 대기됩니다(응답을 게이트하므로); LLM-2/LLM-3 + 감쇠는
응답 이후 백그라운드에서 돌아갑니다.

**백그라운드 컴패니언은 기본값으로 켜져 있습니다(v1.8부터).** `chat()`은 LLM-1
응답을 즉시 반환하고, 컴패니언(LLM-2/LLM-3 + 감쇠)은 백그라운드 워커에서
돌립니다 — 사용자에게 보이는 답변이 정리 작업을 기다리지 않습니다. 워커는
non-daemon 스레드라서 정상 종료 시 남은 작업이 마무리됩니다(`chat()` 직후
종료해도 메모리 유실 없음). 명시적으로 기다리려면 `agent.drain()`을 호출하세요.
인라인(동기) 실행이 필요하면 — 예: `chat()` 직후 컴패니언 출력을 동기적으로
확인하려는 테스트/평가 — `background=False`를 넘기면 됩니다. 플레이그라운드에서는
상단 바의 ⚡ 비동기/동기 컨트롤로 세션 도중 실시간 전환할 수 있습니다.

## 🔍 플레이그라운드 — Sherlock Live Inspector

**Sherlock가 실시간으로 생각하는 것을 지켜보는** 단일 페이지 웹 앱입니다 —
실제 모델로 시스템을 처음부터 끝까지 검증하는 가장 빠른 방법입니다.

```bash
.venv/bin/python -m uvicorn playground.server:app --reload
# → open http://localhost:8000
```

**어떤 공급자든 가져오세요 — 그리고 역할별로 섞으세요.** 하나 이상을 연결합니다:

| 공급자 | 자격 증명 | 비고 |
|---|---|---|
| **Gemini** | AI Studio 키 (`AIza…`) | 키에서 라이브 모델 목록 |
| **OpenAI** | API 키 (`sk-…`) | 채팅 가능한 모델, 최신순 |
| **Anthropic** | API 키 (`sk-ant-…`) | 공식 모델 목록 |
| **DeepInfra / Together / OpenRouter** | API 키 | 오픈소스 모델 호스트(Llama, Qwen, DeepSeek, Mixtral…); 키를 붙여넣으면 라이브 모델 목록이 로드됩니다 |
| **Local** | base URL (예: `http://localhost:11434`) | OpenAI 호환 서버 무엇이든: Ollama, LM Studio, vLLM, llama.cpp |

그런 다음 역할마다 모델을 고릅니다 — 예를 들어 LLM-1에는 Together 호스팅
Llama-3.3-70B를, 컴패니언에는 작은 Qwen을, 혹은 모든 곳에 로컬 Qwen을, 혹은
Gemini Flash와 GPT-4o-mini를 조합할 수 있습니다. 선택은 세션 도중에도
상단 바에서 변경할 수 있습니다(다음 턴부터 적용). API 키는 서버 측
세션에 머무르며 브라우저로 절대 다시 노출되지 않습니다.

> 세 집계 호스트는 OpenAI 호환이며 litellm의
> 네이티브 접두사(`deepinfra/`, `together_ai/`, `openrouter/`)를 통해 라우팅되므로
> **패키지로도** 동작합니다 — `ModelConfig(provider="together", model="…")`
> 또는 YAML의 `models:` 블록으로요. 그 *외의* OpenAI 호환 호스트는
> **Local** 타일을 통해 동작합니다(base URL만 주면 됩니다).

**채팅 경험.** 응답은 토큰 단위로 스트리밍됩니다; 추론 모델은
사고 과정을 접을 수 있는 💭 패널에 드러냅니다; 생성 중에는 Send 버튼이
**Stop**으로 바뀝니다. 컴패니언 모드(`off` / `cold_start` / `turbo`)는
상단 바에서 실시간으로 전환할 수 있고, 다크 모드 토글이 있으며, UI는
7개 언어로 제공됩니다(English · 한국어 · 中文 · 日本語 · Français · Deutsch ·
Español) — 언어 설정은 외관(chrome)에만 영향을 주고 모델의 응답에는
결코 영향을 주지 않습니다.

인스펙터가 보여주는 것(관심사별 탭 하나씩):

- **⚡ Flow** — 모든 이벤트를 순서대로: 턴 시작, 검색, 슬롯
  조립, 지연/토큰이 포함된 LLM 호출, 도구 실행, 백그라운드 작업.
- **🧱 Slot** — 이번 턴에 LLM-1이 받은 정확한 조립 컨텍스트:
  TIER로 강조된 시스템 프롬프트, 블록별 토큰 예산, K-턴 꼬리.
- **💬 LLM I/O** — 세 역할 모두의 원문 프롬프트 + 응답,
  여러 호출로 이뤄진 턴의 모든 내부 호출 포함.
- **🧠 Inference / 🗜 Compaction** — 신뢰도 막대가 있는 LLM-3 가설;
  LLM-2 요약, 사실 표(고정 권고 포함), 페르소나.
- **🗃 Memory** — 감쇠 상태 칩, 출처, 신뢰도,
  사용 횟수가 있는 라이브 메모리 표.
- **🔬 Research** — 심층 리서치 진행 상황: 다국어 검색
  계획, 라운드별 카드(쿼리, 새 출처/조각, LLM-3가 생성한
  질문, 지금까지의 사실), 라이브 **🪙 단계별 토큰 사용량**, 그리고
  인용이 달린 최종 종합 + 세션 문서.
- 상단 바는 역할별 누적 세션 토큰을 보여줍니다(`L1 · L2 · L3 · Σ`).

**"Always run reasoning"** 토글은 매 턴 LLM-2 + LLM-3를 강제로
실행하여 컴패니언 태그를 덜 내보내는 모델에서도 패널이 항상
채워지도록 합니다. 웹 검색은 무료 DuckDuckGo가 기본값입니다; 더 나은 결과를 위해
Brave/Tavily/Valyu 키를 입력할 수 있습니다.

## 보안 & 프라이버시

Sherlock는 대화록과 장기 메모리를 다루므로 가드레일을 명시적으로
밝힙니다:

- **태그는 LLM 출력에서만 실행됩니다.** 사용자가 붙여넣은 도구/컴패니언
  태그는 결코 파싱되지 않습니다 — 사용자는 태그를 입력해 검색, 가져오기,
  메모리 읽기를 유발할 수 없습니다.
- **심층 리서치는 절대 자동 실행되지 않습니다.** 명시적인 사용자 "예",
  UI 승인 클릭, 또는 당신의 `deep_research_approver` 콜백이 필요합니다.
- **비밀/PII 마스킹**(`redact_secrets=True`)은 콘텐츠가
  장기 메모리와 RAG에 들어가기 전에 그것을 지웁니다; 원본 대화록은 결코 변경되지 않습니다.
- **기본적으로 로컬 우선**: SQLite + 온디바이스 임베딩(fastembed) —
  당신이 연결한 LLM/검색 호출 외에는 어떤 것도 당신의 기기를 떠나지 않습니다.
  플레이그라운드에서 공급자 API 키는 서버 측 세션에 머무르며
  브라우저나 이벤트 스트림으로 절대 다시 노출되지 않습니다.
- **진짜 삭제**: `delete_session()`은 메시지, 메모리
  엔트리, 벡터를 연쇄 삭제합니다. 메모리 수정은 비파괴적이며
  감사 가능합니다(대체된 사실은 그 이력을 유지하며, 무효화된 턴이 표시됩니다).
- 신뢰할 수 없는 웹 콘텐츠는 프롬프트 안에서 울타리로 묶이고("지시가 아니라 데이터")
  인용된 모든 URL은 실제로 수집된 출처와 대조해 검증됩니다.
- *아직 구축 안 됨*(로드맵에 있음): 저장 시 암호화, 감사 로그.

## 당신의 LLM이 태그로 할 수 있는 것

어떤 응답의 끝에서든, 당신의 LLM은 다음을 내보낼 수 있습니다(각각 자기 줄에):

```
<<sherlock-companions: compact, infer>>

<<sherlock-tool: search "Seoul weather today">>
<<sherlock-tool: search "nvidia earnings" k=8>>             # set result count (1–10)
<<sherlock-tool: fetch https://example.com/article>>
<<sherlock-tool: fetch raw https://example.com/article>>    # raw HTML

<<sherlock-tool: memory lookup "Yujin 알레르기">>            # semantic + entity recall
<<sherlock-tool: memory entity "Yujin">>                    # deterministic entity match
<<sherlock-tool: memory timeline last 10>>                  # raw recent turns
<<sherlock-tool: memory pinned>>                            # all pinned facts

<<sherlock-tool: deep_research "compare EU vs US AI regulation">>   # approval-gated
```

Sherlock는 태그를 **LLM 출력에서만 파싱하며, 사용자 입력에서는 절대 파싱하지 않습니다**(태그를
붙여넣은 사용자는 아무것도 유발할 수 없습니다), 도구를 실행하고,
결과를 합성 메시지로 다시 먹인 뒤, 당신의 LLM을 재호출합니다 — 턴당
`execution.max_tool_rounds`(기본 3)까지요. 태그는 사용자에게 보이는 응답에서 항상
제거됩니다.

### 컴패니언 게이팅 — LLM-2/LLM-3가 실제로 언제 돌아가는가 (v1.6)

기본적으로 Sherlock는 **`cold_start`** 모드로 동작합니다: 신호 기반 게이트가
대화가 진정으로 컴패니언을 필요로 할 때까지 각 턴을 단일 모델(LLM-1만)로
유지합니다 — 그런 다음 LLM-2/LLM-3로 에스컬레이션하고 상황이 잦아들면 스스로
디에스컬레이션하며, 고정된 턴 카운터는 없습니다. 강력한 모델에서는 차분한 턴에
훨씬 적은 토큰을 쓰면서도 진짜 신호(주제 전환, 모순, 암묵적 의도, 채움 압력)가
나타나는 즉시 발사합니다. 모드는 생성 시점에 고르세요:

```python
Sherlock.with_callable(..., companions_mode="cold_start")  # default
# "off"   — legacy v1.4 behavior, byte-identical (uses the safety net below)
# "turbo" — every companion, every turn (maximum signal, maximum cost)
```

> **≤ v1.4에서의 마이그레이션:** 기본값이 항상 켜진 컴패니언에서
> `cold_start`로 바뀌었습니다. 정확한 v1.4 동작을 복원하려면 `companions_mode="off"`를
> 전달하세요(또는 `SHERLOCK_COMPANIONS=off`를 설정).

**`off`** 모드에서는 당신의 LLM이 태그를 덜 내보낼 때 컴패니언을 살려두는
안전망이 작동합니다: `compact`는 N턴마다 자동 발사되고
(`memory.summarize_every_n_turns`), `infer`는 주제
전환 시 자동 발사됩니다(`memory.auto_infer`: 기본 `"smart"` | `"off"` | `"always"`).
`cold_start`/`turbo`에서는 게이트가 그 결정을 소유하므로 `auto_infer`는
무력화됩니다.

**플레이그라운드**는 같은 세 모드를 드롭다운으로 노출합니다(기본
`turbo`라서 Inference / Compaction 패널이 매 턴 눈에 띄게 채워집니다) —
게이트가 신호로 에스컬레이션되기 전까지 단일 모델을 유지하는 것을 보려면 `cold_start`로,
레거시 동작을 보려면 `off`로 전환하세요.

### 웹 검색 엔진

```python
agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="...",
    main_search_engine="duckduckgo",        # default; free, no key
    inference_search_engine="brave",        # LLM-3 freshness searches
    inference_search_api_key_env="BRAVE_API_KEY",
)
```

엔진: `duckduckgo`(키 없음; 비상업 약관, 뉴스에 약함),
`tavily`(`pip install "sherlock-context[search]"` 필요), `brave`, `valyu`,
`stub`(테스트용). 어떤 역할의 검색을 끄려면 `None`을 전달하세요. 네이티브
도구 호출 어댑터(`make_openai_tools()`, `make_anthropic_tools()`,
`make_openai_memory_tool()`, `dispatch_tool_call`, `dispatch_memory`)는
진짜 함수 호출을 선호하는 통합을 위해 존재합니다.

LLM-3의 프롬프트는 교차 검증 규율을 강제합니다: 주장당 ≥2개 출처,
의견 불일치 시 신뢰도 하향, 단일 출처 웹 사실은
결코 고정되지 않습니다.

## 🔬 심층 리서치

질문이 진짜 깊이를 요구할 때, LLM-1은
`<<sherlock-tool: deep_research "topic">>`를 *제안*합니다. 이것은 **절대 자동 실행되지 않습니다**:

1. Sherlock는 사용자에게 묻거나(플레이그라운드 버튼, 또는 라이브러리/CLI 세션에서 그냥
   "예"/"해줘"라고 답하기) — 혹은 프로그래밍 방식의
   `deep_research_approver(topic, plan)` 콜백이 결정합니다
   (`True`/`False`/`None`=물어보기). 명시적 거부("아니, 하지…",
   "하지마")는 트리거처럼 보이는 단어를 담고 있어도 항상 취소시킵니다.
   **v1.0:** 승인 요청은 초안 작성된 *리서치 전략*
   (목표 + 하위 주제)과 정말로 모호한 지점에 대한 최대 2개의 명료화 질문을
   함께 담습니다 — "예"와 함께 그 질문에 답하세요(혹은 답만 적어
   답하세요; Sherlock가 그것을 접어 넣고 한 번 더 묻습니다). 그 전략은 이후
   가이드라인으로서 실행을 안내하며, 결코 우리가 되지 않습니다.
2. 승인되면 루프가 돌아갑니다(이벤트 싱크나
   `background=True`가 활성화되어 있으면 백그라운드에서):
   - **다국어 키워드 계획** — LLM-3가 답이 있을 가능성이 가장 높은 웹의
     언어를 고르고(일본 여행 질문은
     일본어 + 한국어 + 영어를 훑습니다) 짧고 조사를 제거한
     키워드 쿼리를 내보냅니다. *쿼리 언어*가 i18n 레버입니다 — 전역
     검색, 로케일 파라미터 없음. 당신의 답은 여전히
     *당신의* 언어로 돌아옵니다.
   - **라운드 1은 넓게, 이후는 좁게** — 라운드 1은 폭넓은 스니펫 전용
     훑기입니다; 이후 라운드는 유망한 갈래를 깊게 팝니다. 페이지는
     라운드가 빈약할 때만 드물게 가져오고, 결코 두 번 가져오지 않습니다. 한 라운드에
     맞지 않는 조각은 백로그에서 대기합니다 — **어떤 결과도
     버려지지 않습니다**.
   - **압축된 공유 상태(토큰 절약자)** — 매 라운드 LLM-1은
     오직 NEW 조각 + 확인된 사실과 미해결 빈틈의 압축 다이제스트만 읽고,
     간결한 JSON으로 답합니다. LLM-3(라운드 3부터)는
     원본 페이지가 아니라 다이제스트만 읽어 — 다음 라운드의
     메타 질문을 생성합니다. *실행 중인 루프*는 오래된 조각에 결코 다시 비용을 내지 않습니다.
   - **원본 수집 → 재구성(v1.4, 복구 레이어)** — 매
     라운드의 원본 조각(스니펫 + 관련된 가져온 발췌)은
     버려지지 않고 하위 주제별로 보관됩니다. 최종 종합은 각
     섹션의 원본 버킷을 다시 읽습니다(URL로 중복 제거, 문자 수 상한) — 추출된 사실과 *나란히* —
     그래서 한 라운드의 간결한 추출이 놓친 구체적 디테일
     (행사명, 날짜, 장소)이 영영 사라지는 대신 마지막에
     복구됩니다. 사실은 검증된 척추로 남고; 원본은 복구 레이어이며;
     요청된 하위 주제는 결코 조용히 누락되지 않습니다(사라지는 대신
     솔직한 "확인 안 됨" 메모를 받습니다). *측정 결과*:
     실제 엔진(Brave) + 작은 워커(gemini-flash-lite)로 5개 도시
     일본 행사 쿼리에서, 이전에 "행사 없음"으로 돌아오던
     도시를 인용이 달린 진짜 행사 세 건으로 바꿔놓았습니다 — 그러면서도
     아직 공식 발표되지 않은 날짜는 여전히 표시했습니다.
     `search.deep_research_reconstruct_from_raw`(기본 켜짐) 뒤에 있습니다; 하위 주제별
     "알 가치가 있는 것" 체크리스트 + 커버리지 게이트가 걸린 정지가 작은
     모델이 요청된 모든 부분이 다뤄질 때까지 계속 파고들도록 밉니다.
   - **삼각 검증** — 같은 사실이 서로 다른
     도메인/출처 유형(커뮤니티 / 뉴스 / 공식 / 블로그)을 통해 발견되면
     보강이 누적됩니다; `[corroborated ×N]` 사실이 먼저 순위에 오르고
     종합에서 더 높은 신뢰도로 진술됩니다.
   - **정직한 정지** — `model_sufficient`,
     `converged_no_new_sources`(새 것이 없는 2개 라운드),
     `no_next_queries`, `search_engine_error`(연속 2개 라운드 전부 실패 —
     "수렴"으로 위장하지 않고 실패로 보고됨),
     또는 라운드 상한(≤20)에서 정지합니다.
3. 모든 라운드는 `DEEP_RESEARCH` 세션 문서로 저장되며(필요할 때만
   읽힘 — 리서치가 컨텍스트 윈도우나 고정 사실 상한을 절대 범람시키지 않습니다),
   인용이 달린 단일 종합 보고서가 실행을 마무리합니다.
   리서치 도중에 보낸 메시지는 큐에 쌓이고, 확인되고, 다음 체크포인트에서
   접혀 들어갑니다.
4. `deep_research.tokens` 이벤트가 단계별 입력/출력 토큰을 보고합니다
   (계획 / 라운드 답변 / 메타 질문 / 종합) — 플레이그라운드에서 라이브로
   보입니다 — 그래서 비용은 추측이 아니라 측정됩니다. 백그라운드
   실패는 침묵이 아니라 진짜 응답(`deep_research.failed`)으로
   드러납니다.

```python
agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="...",
    main_search_engine="brave",
    main_search_api_key_env="BRAVE_API_KEY",
    deep_research_approver=lambda topic, plan: None,   # None = ask the user
)
```

## 메모리

### 슬롯 예산 & 동적 K-턴

컨텍스트 슬롯은 명시적 토큰 예산을 가진 네 개의 TIER에 걸쳐
배치됩니다; 남는 것은 모두 원본 턴 꼬리로 가는데, 이 꼬리는
뒤로 걸어가며 *온전한 턴*을 쌓아갑니다(턴은 온전히 들어가거나 빠지거나 둘 중 하나):

```
SYSTEM MESSAGE  (fully stable → cached)
  [TIER 1 — GROUND TRUTH]    sherlock_system + tool_prompt + user_system
  [TIER 2 — SYSTEM-TRACKED]  pinned + persona_summary + compacted highlights
  [TIER 4 — trailer]         marks where the conversation begins
HISTORY MESSAGES (append-only, stable → cached)   ← last N raw turns
FINAL USER MESSAGE (volatile → uncached)
  ═ SYSTEM ANALYSIS FOR THIS TURN ═  this-turn inference + fresh search + fill%
  ═ THE USER'S ACTUAL MESSAGE ═      the user's question (always last)
```

> **v1.4 — 캐시 최적 순서.** 휘발성 이번 턴 블록(추론 +
> 검색)은 예전에 시스템 메시지 안에 있었는데, 그 때문에 뒤따르는
> 모든 대화의 프롬프트 캐싱이 깨졌습니다. 이제 그것은 *최종* 사용자 메시지에
> 실리므로, 시스템 메시지 + 전체 대화 이력이 하나의 **캐시 가능한
> 접두사**를 이루고 마지막 메시지(분석 + 새 질문)만 전액을
> 냅니다. 영역 헤더는 작은 모델이 프로토콜, 이전
> 대화, 이번 턴 시스템 분석, 사용자의 실제 말을 혼동하지 않게 합니다.

프로파일은 모델 컨텍스트 윈도우에 따라 자동 선택됩니다
(`MemoryConfig.slot_budget_profile`: `auto`/`default`/`small`/`off`,
그리고 `slot_budget_overrides`). 무엇이 쓰였는지 점검하세요:

```python
state = agent.inspect_last_turn()
print(state.slot_budget)
print(state.k_turn_turns_used, state.k_turn_tokens_used)
print("hypotheses:", state.hypotheses)
```

### 이력이 어떻게 저장되는가 (컨텍스트 윈도우 절약)

원본 턴은 말한 내용의 유일한 사본이 결코 아니며, 프롬프트를
영원히 키우지도 않습니다:

- **압축(LLM-2).** 백그라운드에서 — 조립된 프롬프트가
  모델 윈도우의 `memory.compact_at_fill_ratio`(기본 0.80)에 도달할 때, 주제
  변경 시, 또는 LLM-1이 요청할 때 — LLM-2가 최근 턴을 지속 가능한 메모리로 증류합니다: **출처가 있는 고정 사실**(`(user t12)`,
  충돌 시 더 최신이 이김), **롤링 페르소나 요약**,
  그리고 추가 전용 하이라이트. 사실은 대화록 인용에 근거해야 합니다; 근거 없는 것은
  신뢰도 상한이 걸리고 결코 고정될 수 없습니다. `corrections` 연산자는
  이후 턴이 이전 고정 사실을 비파괴적으로 대체하게 해줍니다.
- **프런티어 제거("무한 메모리" 메커니즘).** 일단 턴이
  요약되면, 그 *원본* 사본은 TIER-4 꼬리에서 제거되지만(마지막
  몇 턴은 항상 원본으로 남습니다), 그 행은 SQLite에 남아 메모리 도구를 통해 필요할 때
  접근 가능합니다(`memory timeline` / `lookup`). 그래서 턴당
  프롬프트 크기는 대화가 커져도 턴 수에 따라 선형적으로 오르는 대신
  **평탄해집니다** — 큐레이션된 TIER-2 메모리가 전체 대화록이 아니라
  중요한 것을 앞으로 실어 나릅니다.
- **절대 예산.** 각 계층은 단단한 토큰 상한을 가집니다; 원본 꼬리는
  남는 것을 가져가지만 그 자체가 상한이 걸려 있어(`k_turn_max_fraction`, 기본 윈도우의 0.5)
  큰 컨텍스트 윈도우라도 원본 이력이 압축을 밀어내지
  못합니다.

이것이 "컨텍스트 윈도우 절약" 뒤의 메커니즘입니다. 솔직한 주의 사항:
Sherlock의 큐레이션된 프롬프트가 전체 대화록을 재전송하는 것보다 **턴당
더 저렴해지는** *교차점*은 길고 다주제인 세션에서 나타납니다; 짧은
교환에서는 큐레이션 오버헤드 때문에 Sherlock가 *덜*이 아니라 *더* 씁니다
(아래 **비용 대 이득** 참조).

### 프롬프트 계층화

당신의 `system_prompt`는 주된 시스템 메시지로 남습니다; Sherlock의
프로토콜(태그 규약, 교차 검증 규칙)이 그 옆에 함께 실립니다.
`extension_position="before"`로 조정하거나, `sherlock_extension="…"`로
교체하거나, `sherlock_extension=""`로 빼버리세요.

### 세션

```python
for s in agent.list_sessions():
    print(s.id, s.created_at, s.turn_count, "—", s.persona_summary)

agent.new_session()                    # start fresh, keep history
agent.switch_session("abc-123")        # resume an earlier session
agent.delete_session("xyz-789")        # cascade-delete raw + memory + vectors
```

### 상태 & 저장소 점검

```python
for m in agent.messages():
    print(m.role, m.content[:80])
for m in agent.memory.list():
    print(m.type, m.source, m.state, "—", m.content[:80])
```

`with_callable()`은 기본적으로 임시 디렉터리를 씁니다; 실행 간 상태를
유지하려면 `storage_dir="~/.local/share/my_app/sherlock"`를 전달하세요.
비밀/PII는 무엇이든 장기 메모리에 들어가기 전에 마스킹할 수 있습니다
(`redact_secrets=True`); 원본 대화록은 결코 변경되지 않습니다.

## YAML + CLI

모든 것(공급자, 임베딩, 감쇠, 검색, 예산)을 한 파일에서
설정하세요:

```python
agent = Sherlock.from_yaml("sherlock.yaml")
```

전체 스키마는 `sherlock.example.yaml`을 보세요. 패키지는
`sherlock` 명령을 설치합니다:

```bash
sherlock chat --config sherlock.yaml
sherlock config validate | show
sherlock models
sherlock evaluate --config sherlock.yaml --conversation evaluation/dummy_conversation.md
```

## 검증 & 벤치마크

- 80턴 합성 벤치마크에 대한 종단 간 평가
  (`evaluation/dummy_conversation.md` + 골드 표준): Claude Opus 워커로
  **82/100**, 80% 게이트 통과.
- **행동 프로브** — 31개의 단일 능력 프로브(`evaluation/probes/`):

  ```bash
  python -m evaluation.probe_eval --probes evaluation/probes/ \
      --config sherlock.live.yaml --report probe.json
  ```

- 전체 pytest 스위트(500개 이상의 테스트)는 밀폐 환경에서 돌아갑니다 — 스크립트된
  콜러블 + 가짜 엔진, 네트워크나 키 불필요: `pytest -q`.
- **직접 측정하세요**: 플레이그라운드의 A/B 모드는 모든 프롬프트를 같은
  모델에 Sherlock 적용/미적용으로 돌립니다(베이스라인은
  같은 검색 엔진과 오늘 날짜를 받습니다 — 공정한 통제) — 지연 시간과
  토큰 수와 함께 나란히 보여줍니다. 우리 말을 믿기보다 비교하시길 바랍니다.
- 공개 메모리 벤치마크(LongMemEval/LoCoMo 스타일)는 로드맵에 있습니다
  (`docs/ROADMAP.md`, R28) — 우리가 돌려보지 않은 수치는 발표하지 않습니다.

### 비용 대 이득 — 우리가 실제로 측정한 것

Sherlock는 공짜가 아니며, 우리는 아닌 척하지 않습니다. 자체 A/B 실행에서
(양쪽 모두 같은 모델; 워커 = gemini-2.5-flash-lite, 의도적으로 작은
모델):

- **짧은 채팅(2~7턴), 전체 이력이 여전히 들어맞음** — Sherlock와 맨몸
  모델 모두 루브릭을 통과하고(품질에서 *무승부*) Sherlock는 토큰을 **더**
  씁니다(큐레이션 오버헤드). 여기서는 맨몸 모델이 그냥 더 저렴합니다; Sherlock는
  절감이 아니라 *행동*(출처, 정직성, 추론)을 위해 쓰세요.
- **Sherlock가 토큰값을 하는 곳:**
  - *길고 다주제인 세션* — 압축 + 프런티어 제거가
    턴당 프롬프트를 대체로 평탄하게 유지하는 반면 "전부 재전송" 프롬프트는
    한없이 자라며, 큐레이션된 회상은 원본 꼬리 윈도우를 넘어 살아남습니다. 정확한
    토큰 교차점은 트래픽에 따라 다릅니다 — A/B 모드로
    측정하세요; 우리는 공개 벤치마크에서 돌려보지 않은 수치를 인용하지 않습니다.
  - *다부분 심층 리서치* — 실제 엔진(Brave)으로,
    원본 수집→재구성 루프는 같은 검색에서 강력한 단발 RAG 베이스라인을
    이겼습니다: 베이스라인이 "정보 없음"으로 돌려준 도시들에 대해 진짜 인용된
    행사를 표면화했고, 둘 다 아직 발표되지 않은 날짜에 대해서는 정직했습니다.
    단발보다 대략 한 자릿수 더 많은 토큰/지연 비용이 듭니다 — *사용자가 호출하는*
    깊이 기능이며, 그에 맞게 값이 매겨집니다.
  - *쓰레기 검색에서의 정직성* — 깨진 무료 엔진으로(DuckDuckGo가
    무관한 페이지를 돌려줄 때), Sherlock는 지어내는 대신 정직하고 출처가 표시된
    답으로 품위 있게 떨어집니다(`verified` vs `general knowledge — not verified`);
    맨몸의 작은 모델은 진부하거나 지어낸 구체 정보를 단언하는 경향이 있습니다.
- **우리가 고친 실패:** 작은 워커가 풍부한 컨텍스트에서 *미루곤*
  했습니다(필요하지도 않은 디테일을 물어봄) — 추론 레이어가 평이한
  요청을 숨은 요구로 과독해했기 때문입니다. 영가설 브레이크 + 답변 우선
  소비 규칙으로 고쳤습니다 — 이제 모델은 이미 가진 컨텍스트로 답합니다
  (그것을 드러낸 스모크 테스트에서 1/3 vs 3/3).

결론: 짧고 잘 들어맞는 대화에서는 맨몸 모델이 더 저렴하고
똑같이 좋습니다; Sherlock는 **길이, 다부분 리서치, 정직성**에서 값을 합니다.
프레이밍은 "*옳고 완전하기* 위해 토큰을 쓴다"이지 "토큰을 덜
쓴다"가 아닙니다 — 큐레이션이 원본 비용에서도 이기는 긴 세션은 예외입니다.

## 한계

- 일관된 페르소나와 여러 주제 갈래를 가진 대화에서 가장 잘 작동합니다;
  무작위의 짧은 교환은 컴패니언에게 할 일을 거의 주지 않습니다.
- 출처 원장은 사용자가 말한 사실과 시스템이 추론한
  사실을 구분하지만 외부 주장을 검증하지는 않습니다.
- 메모리 감쇠는 시간/턴 기반입니다; 의미 클러스터 감쇠는 명세되었으나
  배선되지 않았습니다.
- Evolution Engine은 컴패니언 프롬프트를 버전 관리하지만 아직 사용자 피드백에서
  자동으로 학습하지는 않습니다.

우선순위가 매겨진 업그레이드 계획 — 작은 모델 지능, 공급자 프롬프트
캐싱, 심층 리서치 신뢰, 메모리 조정 — 은
**[docs/ROADMAP.md](docs/ROADMAP.md)**(R1–R35, 증거 링크 포함)에 있습니다.

## 변경 이력 하이라이트

### v1.4 — 캐시 최적 슬롯, 채움 기반 압축, 컴패니언 캐스케이드
- **캐시 최적 재정렬**: 휘발성 이번 턴 블록(추론 + 검색)이
  시스템 메시지에서 *최종* 사용자 메시지로 옮겨가, 시스템
  메시지 + 전체 대화 이력이 하나의 캐시 가능한 접두사가 됩니다 —
  캐싱 공급자에서 긴 대화는 가장 새로운 메시지에 대해서만 다시 비용을 냅니다.
  명시적 영역 헤더는 작은 모델이 프로토콜 / 이전
  대화 / 이번 턴 분석 / 사용자의 실제 말을 혼동하지 않게 합니다.
- **채움 기반 압축**: LLM-2는 고정된 턴 주기 대신 프롬프트가
  윈도우의 `memory.compact_at_fill_ratio`(기본 0.80)에 도달할 때 자동
  압축합니다 — 그 아래에서는 대화가 추가 전용으로 자라고 캐싱이
  비용을 낮게 유지합니다; 라이브 컨텍스트 채움 %가 LLM-1에 드러납니다.
- **컴패니언 캐스케이드 & 순서**: LLM-2가 LLM-3보다 먼저 돌아가고(추론이
  방금 압축된 메모리 위에서 추론하도록), LLM-2가 `worth_digging` 갈래를 표면화하면
  그것이 직접 LLM-3를 트리거합니다 — 잦은 가벼운 추론(LLM-1 구동)
  + 가끔의 깊은 추론(LLM-2 구동). 심층 리서치 LLM-3 페르소나가 이제 캐시됩니다.

### v1.4 — 잊지 않는 심층 리서치; 답변 우선의 작은 모델
- **원본 수집 → 재구성**: 각 라운드의 원본 조각이 하위
  주제별로 보관되고 **종합에서 다시 읽힙니다**(사실 = 검증된 척추, 원본 =
  복구 레이어), 그래서 한 라운드가 덜 추출한 구체 디테일이 유실되는 대신
  복구됩니다. 요청된 하위 주제는 결코 조용히 누락되지 않습니다;
  하위 주제별 "알 가치가 있는 것" 체크리스트와 커버리지 게이트가 걸린 정지가
  작은 모델이 모든 부분이 다뤄질 때까지 계속 파고들도록 밉니다. 모두
  설정 킬 스위치 뒤에 있습니다(off = 정확한 이전 동작). gemini-flash-lite + Brave에서
  *라이브 검증됨*: "행사 없음"으로 돌아오던 도시가 이제 진짜 인용된
  행사를 표면화하며, 단발 RAG 베이스라인을 이깁니다.
- **답변 우선 추론**: 추론 레이어가 더 이상 작은 모델로 하여금
  풍부한 컨텍스트에서 미루게 하지 않습니다 — 영가설 브레이크가 평이한
  요청을 숨은 요구로 읽는 것을 멈추고, 소비 규칙이 답변을 앞세웁니다
  (그런 다음 암묵적 연쇄를 다룹니다), 결코 답변을
  명료화 질문으로 대체하지 않습니다.
- **방법이지 강압이 아님**: "you MUST …"라고 했던 리서치/전략
  프롬프트가 안내로 다시 쓰였습니다 — 단단한 명령은 작은 모델을 멈춰 세웁니다.
- 367개 밀폐 테스트; 원본 복구, 커버리지
  게이팅, 미룸 수정에 대한 새 결정론적 증명.

### v1.2–1.3 — 라이브 피드백 강화
- 모든 리서치 프롬프트에 TODAY 날짜 주입("this December"가
  작년으로 해석되던 문제 수정); 암묵적 연쇄 추론(`really_asking` + 준비된
  다음 답변)을 LLM-1의 다음 턴 슬롯으로 운반; 인용 **페어링**
  검증; 플레이그라운드의 A/B 모드 + 턴별 마크다운 내보내기;
  **공정한** 베이스라인(같은 검색 엔진 + 오늘 날짜); 의미적 신규성
  수렴으로 재표현된 결론이 리서치 라운드를 태우지 않게 함.

### v1.1 — 로드맵 전체, 배송 완료
- **작은 모델 신뢰성**: 단발 예시가 LLM-2/LLM-3 JSON
  출력을 고정시킴; 공급자가 지원하는 곳에서는 제약 JSON 디코딩이 자동으로
  요청됨(그 외에는 메모이즈된 폴백); 거의 들어맞는 도구 태그
  (`sherlock_tool`, 단일 괄호)는 새어나가는 대신 수리됨.
- **심층 리서치 신뢰**: 인용된 모든 URL이 수집된
  출처와 대조됨 — 지어낸 인용은 인라인 `(unverified)` 플래그를 받음; 검색
  계획과 메타 질문이 능동적으로 반대 증거를 찾음; 큰 실행(>18
  사실 + 전략 개요)은 **섹션별로** 종합하며, 각 호출은 자기
  사실만 읽음; 증거는 문장 경계에서 잘림.
- **메모리 품질**: 검색이 최신성/중요도 부스트와 새 메모리 링크 표를 통한
  1홉 확장을 추가함(A-Mem 스타일); 유형별 결과
  상한; 대체된 사실은 시간 질문을 위해 `invalid_at_turn`("superseded at t7")을 운반함;
  LLM-2 사실은 대화록과 대조 검증되는 지지 **인용**을
  운반할 수 있음 — 근거 없는 사실은 신뢰도 상한이 걸리고
  결코 고정될 수 없음.
- **토큰 효율**: 페르소나 사실이 없으면 출처 원장을 전부
  건너뜀; RAG는 고정 사실을 다시 표면화하지 않음(TIER-2가 이미
  운반함); 운반된 검색 결과는 관련성 게이트가 걸림;
  시스템 프롬프트가 이제 두 개의 캐시 영역(프로토콜 / TIER-2)을 표시해 고정 사실
  교체가 더 이상 프로토콜 캐시를 무효화하지 않음; 선택적 LLMLingua-2
  압축이 같은 리서치 예산에 ~2.5배 더 많은 관련 페이지 텍스트를
  채워 넣음(`pip install "sherlock-context[compress]"`).

### v1.0 — 리서치 전략, 조각 재조립, 무한 메모리, 캐시 네이티브 프롬프트
- **리서치 전략 단계**: 심층 리서치 실행 전에 LLM-1이 짧은
  전략(목표, 하위 주제, 범위)을 초안하고 승인과 함께 최대 2개의
  명료화 질문을 던짐 — 답이 실행에 접혀 들어감; 전략은
  *가이드라인이지 우리가 아님*. 하위 주제가 미해결 빈틈 추적을 씨앗으로 심어,
  커버리지가 기존 수렴 기계로 측정됨.
- **조각 재조립**: 가져온 페이지가 쿼리 관련성으로 발췌됨
  (댓글에 묻힌 조각이 페이지 머리 대신 표면화됨); 재표현된 사실이
  출처를 병합함(보강이 표현과 언어를 가로질러 누적됨);
  거의 들어맞는 모순은 `[disputed]`로 태깅되어 양면으로 보고됨;
  NEW 사실을 추가하지 않는 라운드는 수렴함(`converged_no_new_facts`);
  보여지는 조각은 출처 유형 다양성을 앞세움(RRF + 라운드 로빈).
- **LLM-2 메모리 재공고화**: 압축기가 이제
  진부한 고정 사실을 비파괴적으로 대체하는 `corrections`를 내보낼 수 있음
  (옛 행은 질의 가능하게 남고, `(superseded)`로 표시되며, 프롬프트와
  중복 제거에서 제외됨); 그 `retrieval_keywords`가 이제 RAG 쿼리를 확장함; 한글 바이그램
  BM25가 한국어 교착 형태를 검색 가능하게 만듦.
- **무한 메모리(압축 프런티어)**: LLM-2 요약에 이미 포함된 원본
  턴이 K-턴 꼬리를 떠남(마지막 4턴은 항상 원본으로 유지;
  모든 것은 SQLite + 메모리 도구에 남음). 측정 결과: 8턴에 압축된 세션의
  14턴에서 턴당 −684토큰 — 절감은 세션 길이에 따라 커짐.
- **정직한 작은 윈도우**: `with_callable(context_window=8192,
  max_output_tokens=…, slot_budget_profile=…)` + 새 8K/16K/32K 예산
  프로파일; 가장 최근 턴은 예산을 우회하므로 **이력이 결코
  0이 되지 않음**; 윈도우가 선언되지 않으면 일회성 경고.
- **캐시 네이티브 프롬프트**: 시스템 메시지가 바이트 안정적인 TIER 1+2
  접두사를 표시함; LiteLLM이 그것을 `cache_control` 블록으로 변환하고(Anthropic)
  `cache_read/creation_tokens`를 보고함; LLM-2/LLM-3 프롬프트는 전체 메시지로
  힌트됨; BYO 콜러블은 `cache_hints` kwarg로 옵트인할 수 있음 — 평이한
  `f(messages)` 페이로드는 바이트 단위로 동일하게 유지됨. 플레이그라운드 토큰 바가
  `⚡cached`를 표시함.
- **낭비 제거**: 죽은 LLM-3 출력 필드 제거됨; 출처
  원장 / 메시지 래퍼 / 도구 결과 배너가 압축됨(109→52토큰,
  가드레일은 그대로); 프로토콜 문서가 이제 조건부임(검색 엔진 없음 →
  턴당 1,308→745토큰); 실패한 JSON 파싱은 컴패니언 호출 전체를
  낭비하는 대신 오류를 되먹여 한 번 재시도함.

### v0.9 — 강화 + 범용 플레이그라운드
- 플레이그라운드가 **멀티 공급자**임: Gemini, OpenAI, Anthropic,
  오픈소스 모델 호스트(DeepInfra · Together · OpenRouter), 그리고 임의의
  로컬 OpenAI 호환 서버, 역할별로 조합 가능; 누적 토큰
  바; WS 자동 재연결; 턴별 LLM 호출 이력; IME 안전 입력.
- 적대적 멀티 에이전트 감사에서 나온 심층 리서치 정확성(확인된 30개
  발견 수정됨): 손상된 작은 모델 JSON이 더 이상 실행을
  중단시킬 수 없음; 거부는 결코 승인하지 않음; 라운드 1 오버플로는
  버려지는 대신 백로그로 감; 엔진 장애가 루프를
  정직하게 멈춤; 백그라운드 실패가 응답으로 드러남; 리서치 도중
  메시지는 항상 확인되고, 영속되고, 접혀 들어가거나 회계
  처리됨; 리서치 문서가 더 이상 고정 사실을 제거하지 않음.
- 메모리 무결성: 재진술된 사실이 FORGOTTEN에서 부활함; 중복 제거 접두사를
  지난 수정이 이제 저장된 사실을 *갱신*함; LLM-2 출력이 더 이상
  "사용자 검증" 고정으로 세탁될 수 없음; 프롬프트 블록이 각 사실이 학습된
  턴을 운반함(`t12`) 그래서 충돌 시 더 최신이 이김.
- 한국어 검색 쿼리: 다문자 조사만 제거됨 —
  하와이/제주도/고양이는 살아남음; 따옴표 묶은 구절과 버전 번호
  (`"exact phrase"`, `3.12`, `C++`)는 정리를 거쳐도 그대로 통과함.

### v0.8 — 다국어 검색 + 토큰 위생
- LLM-3가 주제에 가장 관련 있는 언어들에 걸쳐 깨끗한 키워드 쿼리를
  계획함; 전역 검색, 레버로서의 쿼리 언어.
- 압축된 공유 리서치 상태: 라운드별 델타, LLM-3는 원본
  페이지를 결코 보지 않음, 간결한 JSON 라운드, 중복 제거된 사실에서의 종합;
  `deep_research.tokens` 단계별 측정.
- `[corroborated ×N]` 순위가 매겨진 조각 삼각 검증.

### v0.7 — 세 가지 검색 모드
- LLM-1이 검색 결과 수를 설정함(`k=N`); LLM-3가 자기 평가형
  백그라운드 검색 루프를 돌림; `deep_research`가 세션 문서와 메타 인지 Q&A를 갖춘
  승인 게이트 ≤20라운드 루프로 배송됨.

### v0.5 — 코어 루프
- 진짜 백그라운드 컴패니언, 기본 로컬 임베딩, 마스킹,
  세션 관리, 슬롯 예산, 행동 프로브.
