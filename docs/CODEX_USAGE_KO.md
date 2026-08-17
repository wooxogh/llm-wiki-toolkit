# Codex용 사용법

이 문서는 `llm-wiki`를 처음 보는 사람이 Codex와 함께 쓰기 위해 읽는 한국어
가이드입니다. 설치, vault 개념, 전체 명령어, 자주 쓰는 흐름을 한 곳에 정리합니다.

## 1. 이 프로젝트가 하는 일

`llm-wiki`는 Markdown 문서 폴더를 Codex가 다시 찾아볼 수 있는 로컬 지식 베이스로
만드는 도구입니다.

일반적인 사용 흐름은 이렇습니다.

```text
Markdown 문서 작성
  -> wiki-index로 index.yaml 생성
  -> wiki-embed로 검색용 임베딩 생성
  -> wiki-recall로 필요한 지식 검색
  -> Codex가 검색 결과의 원본 Markdown을 읽고 작업
```

핵심 원칙은 “코드 전체를 복사해서 저장하지 않고, 오래 남을 지식만 Markdown으로
정리한다”입니다. 예를 들어 설계 이유, 측정 결과, 실패한 접근, 재사용 가능한 패턴은
vault에 남기고, 현재 코드의 세부 구현은 실제 repo에서 읽습니다.

## 2. 간단한 아키텍처

처음 보면 이름이 낯설 수 있으니 주요 구성요소부터 잡고 가면 쉽습니다.

```text
llm-wiki-toolkit-main/
  src/llm_wiki/                 # 핵심 Python 패키지
  integrations/                 # Codex/Claude 메모리, ingest 자동화
  docs/                         # 문서
  examples/vault/               # 예제 vault
  pyproject.toml                # 설치/명령어 등록 정보

vault/
  wiki.toml                     # vault 설정 파일, 없어도 기본값으로 동작
  domain/                       # 도메인 지식
  patterns/                     # 반복되는 패턴, 교훈
  entities/                     # 특정 도구/개념/컴포넌트 설명
  raw/                          # 정리 전 원자료
  index.yaml                    # wiki-index가 생성하는 검색용 색인
  .embeddings/                  # wiki-embed가 생성하는 벡터 저장소
```

구성요소별 역할은 아래처럼 이해하면 됩니다.

| 구성요소 | 뜻 |
|---|---|
| toolkit | `wiki-index`, `wiki-recall` 같은 명령어를 제공하는 프로그램 |
| vault | Codex가 참고할 Markdown 지식 저장소 |
| `wiki.toml` | vault 설정 파일 |
| `index.yaml` | Markdown frontmatter를 모은 생성 파일 |
| `.embeddings/` | 의미 검색을 위한 로컬 임베딩 저장소 |
| Codex memory | Codex가 프로젝트별로 들고 있는 얇은 메모리 캐시 |

중요한 점은 toolkit과 vault가 꼭 같은 폴더일 필요는 없다는 것입니다. toolkit은 한 곳에
설치해두고, `WIKI_VAULT`만 바꿔서 다른 vault를 검사할 수 있습니다.

## 3. 설치

PowerShell 기준입니다. 먼저 toolkit 폴더로 이동합니다.

```powershell
cd "C:\Users\louis\Desktop\All\Code\opsource(2)\llm-wiki-toolkit-main\llm-wiki-toolkit-main"
```

가상환경을 만들고 활성화합니다. 가상환경은 이 프로젝트에 필요한 Python 패키지를
전역 Python과 분리해서 설치하는 공간입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

기본 기능을 설치합니다.

```powershell
pip install -e .
```

`wiki-embed`, `wiki-recall`, rerank처럼 임베딩 모델이 필요한 기능까지 쓰려면 ML extra를
설치합니다.

```powershell
pip install -e ".[ml]"
```

개발/테스트까지 하려면:

```powershell
pip install -e ".[dev]"
```

## 4. vault 지정

`WIKI_VAULT`는 “이번 명령이 어떤 vault를 볼지” 정하는 환경변수입니다.

예제 vault로 테스트:

```powershell
$env:WIKI_VAULT = "$PWD\examples\vault"
```

본인이 만든 `test` 폴더를 vault로 쓰기:

```powershell
$env:WIKI_VAULT = "C:\abs\path\to\test"
```

`WIKI_VAULT`는 한 번에 하나만 지정합니다. 여러 vault를 동시에 검색하는 기능은 현재
없습니다. 여러 프로젝트를 나누고 싶다면 보통 하나의 vault 안에서 `projects:` 필드로
구분합니다.

## 5. 최소 vault 예시

처음 테스트할 때는 아래처럼 만들면 됩니다.

```text
test/
  wiki.toml
  domain/
    example.md
```

`wiki.toml`:

```toml
[vault]
content_dirs = ["domain", "patterns", "entities", "raw"]

[eval.minimums]
total = 0
recent_cases = 0
layer = {}
domain = {}
category = {}
```

`domain/example.md`:

```markdown
---
id: example
layer: domain
domain: test
projects: [test]
tags: [demo]
confidence: confirmed
status: active
updated: 2026-08-14
summary: '테스트 vault가 정상적으로 인덱싱되는지 확인하는 예시 페이지.'
---

# Example

이 페이지는 `llm-wiki` 테스트용 예시 문서입니다.
```

확인:

```powershell
wiki-index
wiki-health --mode ci
```

## 6. 가장 자주 쓰는 흐름

새 문서를 만들었거나 수정했다면:

```powershell
wiki-index
wiki-health --mode ci
```

의미 검색까지 쓰려면:

```powershell
wiki-index
wiki-embed
wiki-recall "검색할 질문" --k 8 --rerank 10
wiki-health --mode full
```

Codex에게 작업을 시킬 때는 이렇게 말하면 됩니다.

```text
먼저 `wiki-recall "<질문>" --k 8 --rerank 10`으로 vault를 검색하고,
검색 결과의 snippet만 믿지 말고 원본 Markdown 파일을 직접 읽은 뒤 작업해줘.
```

## 7. 명령어 전체 정리

### 설치/환경 명령

| 명령어 | 기능 | 사용 예시 |
|---|---|---|
| `python -m venv .venv` | 현재 폴더에 Python 가상환경 생성 | `python -m venv .venv` |
| `.\.venv\Scripts\Activate.ps1` | PowerShell에서 가상환경 활성화 | `.\.venv\Scripts\Activate.ps1` |
| `pip install -e .` | toolkit 기본 설치 | `pip install -e .` |
| `pip install -e ".[ml]"` | 임베딩/검색 모델 의존성 설치 | `pip install -e ".[ml]"` |
| `pip install -e ".[dev]"` | 테스트 도구 설치 | `pip install -e ".[dev]"` |
| `$env:WIKI_VAULT = "..."` | 사용할 vault 지정 | `$env:WIKI_VAULT = "C:\vaults\test"` |

### `wiki-index`

Markdown 문서의 frontmatter를 읽어서 `index.yaml`을 만듭니다. vault 문서를 추가하거나
수정한 뒤 가장 먼저 실행하는 명령입니다.

```powershell
wiki-index
```

CI나 검증용으로 “파일을 쓰지 말고 현재 `index.yaml`이 최신인지 확인”하려면:

```powershell
wiki-index --check
```

주로 실패하는 경우:

| 원인 | 뜻 |
|---|---|
| filename과 `id` 불일치 | `example.md`면 `id: example`이어야 함 |
| 필수 frontmatter 누락 | `layer`, `projects`, `tags`, `summary` 등이 필요 |
| 잘못된 wikilink | `[[id]]`가 실제 페이지 id와 맞지 않음 |
| stale index | Markdown은 바뀌었는데 `index.yaml`이 갱신되지 않음 |

### `wiki-health`

vault 전체 상태를 검사합니다. “생성 파일이 최신인지, 링크/요약/커뮤니티가 이상하지
않은지”를 보는 gate입니다.

임베딩 없이 빠르게 검사:

```powershell
wiki-health --mode ci
```

임베딩까지 포함해서 전체 검사:

```powershell
wiki-health --mode full
```

보통은 Markdown만 수정했으면 `ci`, `.embeddings/`까지 관리하면 `full`을 씁니다.

### `wiki-embed`

Markdown 문서를 의미 검색용 벡터로 변환해서 `.embeddings/`에 저장합니다. 처음 실행할
때 모델 다운로드가 필요할 수 있고, `pip install -e ".[ml]"`가 필요합니다.

```powershell
wiki-embed
```

장치를 명시하고 싶으면:

```powershell
$env:WIKI_EMBED_DEVICE = "cpu"
wiki-embed
```

가능한 값은 보통 `cpu`, `cuda`, `mps`입니다. Windows에서는 우선 `cpu`가 가장 무난합니다.

### `wiki-recall`

vault에서 질문과 관련된 페이지를 검색합니다. Codex가 작업 전에 가장 자주 쓰게 되는
명령입니다. `pip install -e ".[ml]"`와 `wiki-embed`가 필요합니다.

기본 검색:

```powershell
wiki-recall "BM25와 dense score를 왜 같이 쓰지?" --k 8
```

개념/패턴 질문처럼 표현이 다양할 때 rerank 사용:

```powershell
wiki-recall "검색 품질이 떨어질 때 무엇을 확인해야 하지?" --k 8 --rerank 10
```

특정 project만 검색:

```powershell
wiki-recall "임베딩 갱신 방식" --project my-repo
```

JSON으로 출력:

```powershell
wiki-recall "질문" --json
```

자동 판단 모드:

```powershell
wiki-recall "질문" --auto --json
```

`--auto` 결과는 이렇게 해석합니다.

| 결과 | 의미 |
|---|---|
| `answer` | vault가 충분히 확신하는 답 |
| `review` | 후보는 있지만 사람이 읽고 판단해야 함 |
| `none` | vault가 모름 |

### `wiki-eval`

검색 품질을 gold set으로 측정합니다. `eval_gold.json`이 있는 vault에서 의미가 있습니다.

gold set 형식과 커버리지만 확인:

```powershell
wiki-eval --validate-only
```

검색 품질 측정:

```powershell
wiki-eval --k 8 --mode hybrid
```

자동 판단 품질 측정:

```powershell
wiki-eval --split test --auto
```

`--auto` 기준값 보정:

```powershell
wiki-eval --calibrate auto_thresholds.json
```

처음 작은 테스트 vault에서는 `eval_gold.json`이 없을 수 있으니 이 명령은 건너뛰어도
됩니다.

### `wiki-gate`

현재 검색 품질이 기존 baseline보다 나빠졌는지 확인합니다. 검색 알고리즘을 바꿨거나
gold set을 관리하는 vault에서 씁니다.

```powershell
wiki-gate
```

baseline을 새로 쓰기:

```powershell
wiki-gate --update
```

`--update`는 “품질 기준을 바꾸는 작업”이라 측정 이유를 기록하고 신중하게 써야 합니다.

### `wiki-lint`

측정 없이 애매하게 주장하는 문장을 찾아 경고합니다. 실패로 처리하지 않고 warning만
냅니다.

전체 vault 검사:

```powershell
wiki-lint
```

특정 페이지만 검사:

```powershell
wiki-lint --page example
```

한국어 문장까지 보고 싶으면 `wiki.toml`에 lint pack을 지정합니다.

```toml
[lint]
packs = ["en", "ko"]
```

### `python -m llm_wiki.reports.graph_report`

vault 페이지 사이의 링크 그래프 보고서를 만듭니다.

보고서 생성:

```powershell
python -m llm_wiki.reports.graph_report --write
```

생성 파일:

```text
GRAPH_REPORT.md
```

링크가 너무 끊겨 있거나 orphan 페이지가 많을 때 구조를 파악하는 데 씁니다.

### `python -m llm_wiki.reports.community_report`

링크 그래프에서 관련 페이지 묶음, 즉 community를 찾아 보고서를 만듭니다.

보고서 생성:

```powershell
python -m llm_wiki.reports.community_report --write
```

요약이 필요한 community만 확인:

```powershell
python -m llm_wiki.reports.community_report --stale
```

생성/관련 파일:

```text
COMMUNITIES.md
community_summaries.json
```

`--stale`이 무언가 출력하면 해당 community를 읽고 `community_summaries.json`에 짧은
근거 기반 요약을 써야 `wiki-health`가 통과합니다.

### `python -m llm_wiki.hygiene.contradict`

서로 모순될 가능성이 있는 페이지 후보를 찾습니다. 자동으로 고치지 않고 후보만
출력합니다. 임베딩이 필요합니다.

전체 검사:

```powershell
python -m llm_wiki.hygiene.contradict --tau 0.78
```

변경된 페이지 주변만 검사:

```powershell
python -m llm_wiki.hygiene.contradict --changed id1,id2 --tau 0.75
```

결과 파일:

```text
.contradictions.md
```

### `python -m llm_wiki.hygiene.compact`

너무 커졌거나 합치기/나누기/수정이 필요해 보이는 페이지 후보를 찾습니다. 이것도
자동 수정 도구가 아니라 검토 후보 생성기입니다. 임베딩이 필요합니다.

```powershell
python -m llm_wiki.hygiene.compact --merge-tau 0.85 --size-kb 12 --corrections 6
```

결과 파일:

```text
.compaction.md
```

### `python -m integrations.agent_memory.sync_cache`

vault의 지식을 Codex 또는 Claude 프로젝트 메모리로 동기화합니다. 원본을 복사하는 게
아니라 “canonical 문서는 vault에 있으니 거기를 읽어라”라는 포인터를 만듭니다.

Codex용:

```powershell
python -m integrations.agent_memory.sync_cache --project "C:\abs\path\to\repo" --target codex
```

Claude용:

```powershell
python -m integrations.agent_memory.sync_cache --project "C:\abs\path\to\repo" --target claude
```

둘 다:

```powershell
python -m integrations.agent_memory.sync_cache --project "C:\abs\path\to\repo" --target both
```

쓰기 전에 미리 보기:

```powershell
python -m integrations.agent_memory.sync_cache --project "C:\abs\path\to\repo" --target codex --dry-run
```

### `python -m integrations.ingest.ingest_pipeline`

daily ingest 자동화 파이프라인입니다. repo의 변경 내용을 agent가 읽고 vault 페이지를
작성한 뒤, 인덱스/임베딩/보고서/health/commit 순서로 처리합니다.

먼저 `wiki.toml`에 대상 repo와 Codex 사용을 지정합니다.

```toml
[ingest]
repos = ["C:/abs/path/to/repo"]
agent = "codex"
```

실행될 단계만 보기:

```powershell
python -m integrations.ingest.ingest_pipeline --dry-run
```

agent authoring step을 빼고 deterministic 단계만 실행:

```powershell
python -m integrations.ingest.ingest_pipeline --skip-llm
```

Codex로 실행:

```powershell
python -m integrations.ingest.ingest_pipeline --agent codex
```

특정 vault와 stamp 파일 지정:

```powershell
python -m integrations.ingest.ingest_pipeline --vault "C:\vaults\test" --stamp "C:\tmp\llm-wiki-last-run"
```

이 파이프라인은 작업 시작 전에 git working tree가 더러우면 중단합니다. 자동 commit에
사용자 작업물이 섞이는 일을 막기 위해서입니다.

### `integrations/ingest/run_ingest.sh`

macOS/Linux 스케줄러에서 ingest를 부르기 위한 얇은 shell wrapper입니다. Windows
PowerShell에서는 보통 직접 `python -m integrations.ingest.ingest_pipeline`을 실행하면
됩니다.

```bash
./integrations/ingest/run_ingest.sh --agent codex
```

### `integrations/macos/install.sh`

macOS LaunchAgent 등록용 설치 스크립트입니다. Windows에서는 사용하지 않습니다.

```bash
./integrations/macos/install.sh
```

## 8. Codex와 함께 쓰는 추천 방식

Codex에게 아래 규칙을 프로젝트 `AGENTS.md`에 넣어두면 좋습니다.

```text
작업 전에 관련 지식을 `wiki-recall "<질문>" --k 8 --rerank 10`으로 먼저 조회한다.
검색 결과의 snippet만 믿지 말고, 반환된 canonical Markdown 파일을 직접 읽는다.
새로 확인한 durable knowledge만 vault에 기록한다.
측정되지 않은 추측은 기록하지 않는다.
vault를 편집했다면 `wiki-index`와 `wiki-health --mode ci`를 실행해 drift를 확인한다.
```

Codex 자동 ingest를 쓰면 내부적으로 `codex exec`가 실행됩니다. 이때 vault가 작업 루트가
되고, `[ingest] repos`에 적힌 repo들은 `--add-dir`로 추가됩니다. 그래서 Codex는 repo의
변경 이력을 읽고, vault에는 정리된 지식만 남기는 방식으로 동작합니다.

## 9. 목적별 빠른 레시피

### 새 vault를 처음 테스트

```powershell
$env:WIKI_VAULT = "C:\vaults\test"
wiki-index
wiki-health --mode ci
```

### 검색까지 테스트

```powershell
pip install -e ".[ml]"
$env:WIKI_VAULT = "C:\vaults\test"
wiki-index
wiki-embed
wiki-recall "테스트 문서가 뭐였지?" --k 8 --rerank 10
```

### Codex 메모리 동기화

```powershell
$env:WIKI_VAULT = "C:\vaults\test"
python -m integrations.agent_memory.sync_cache --project "C:\repos\my-repo" --target codex --dry-run
python -m integrations.agent_memory.sync_cache --project "C:\repos\my-repo" --target codex
```

### vault 편집 후 검증

```powershell
wiki-index
wiki-lint
wiki-health --mode ci
```

### 전체 관리 루틴

```powershell
wiki-index
wiki-embed
python -m llm_wiki.reports.graph_report --write
python -m llm_wiki.reports.community_report --write
python -m llm_wiki.reports.community_report --stale
wiki-health --mode full
```

## 10. 자주 헷갈리는 점

| 질문 | 답 |
|---|---|
| `wiki.toml`이 꼭 필요한가? | 없어도 기본값으로 동작하지만, vault 규칙을 명확히 하려면 두는 것이 좋습니다. |
| vault를 여러 개 동시에 지정할 수 있나? | 한 명령 실행에서는 하나만 지정합니다. 필요하면 `WIKI_VAULT`를 바꿔가며 실행합니다. |
| 가상환경은 꼭 필요한가? | 필수는 아니지만 Python 패키지 충돌을 막기 위해 권장합니다. |
| `wiki-recall` 전에 `wiki-embed`가 필요한가? | 의미 검색을 하려면 필요합니다. |
| `index.yaml`을 직접 고쳐도 되나? | 안 됩니다. Markdown을 고치고 `wiki-index`로 재생성합니다. |
| `.embeddings/`를 git에 커밋하나? | 보통 커밋하지 않습니다. 로컬 생성물입니다. |
| Codex memory가 원본인가? | 아닙니다. 원본은 vault Markdown이고, Codex memory는 포인터 캐시입니다. |

