# llm-wiki v2 전체 사용 설명서

이 문서는 llm-wiki를 처음 접하는 사용자가 Windows PowerShell에서 설치하고, Markdown 문서를 Concept와 NET으로 변환하고, 검색과 검토 및 품질 검사까지 수행할 수 있도록 설명한다.

llm-wiki v2의 핵심은 다음 한 문장으로 요약할 수 있다.

> 사람이 작성한 Markdown을 원본으로 유지하면서, LLM이 Atomic Concept를 추출하고 deterministic code가 검증된 계층과 관계만 NET에 저장한다.

## 1. 먼저 알아둘 용어

| 용어 | 의미 |
|---|---|
| vault | llm-wiki가 관리할 Markdown 폴더 전체. `wiki.toml`이 있는 폴더를 말한다. |
| Markdown | 사람이 직접 작성하고 수정하는 원본 지식. 유일한 source of truth다. |
| Document | Markdown 파일 하나를 나타내는 v2 record다. |
| Chunk | Concept 추출을 위해 Markdown을 의미 단위로 나눈 내부 조각이다. NET node는 아니다. |
| Atomic Concept | 독립적으로 참/거짓 또는 현재 유효성을 판단할 수 있는 최소 주장이다. |
| Topic | Concept를 의미적으로 분류하는 계층 node다. |
| Collection | 월간 보고서처럼 여러 Document가 한 시리즈임을 나타내는 선택적 계층 node다. |
| NET | Topic, Collection, Document, Concept의 tree backbone과 Concept 관계를 합친 구조다. |
| Proposal | LLM이 제안했지만 아직 deterministic validation 또는 사용자 승인이 끝나지 않은 결과다. |
| Artifact | Markdown에서 다시 생성할 수 있는 `index.yaml`, vector, Concept, NET 등의 파생 파일이다. |

## 2. 간단한 아키텍처

```text
Markdown + YAML frontmatter                 사람이 관리하는 원본
        |
        +---------------------> wiki-index -> index.yaml
        |                                      legacy page 검색 metadata
        |
        +-> deterministic chunking
              # -> ## -> paragraph
              |
              v
          User LLM agent
          Codex 또는 Claude
              |
              v
          Atomic Concepts
              |
              +-> Concept embedding + BM25
              |        |
              |        v
              |    hybrid candidate search
              |        |
              |        v
              |    relation proposals
              |        |
              |    safe       risky
              |      |          |
              |      v          v
              |   auto gate  wiki-review
              |        \       /
              |         \     /
              v          v   v
                         NET
          Topic / Collection / Document / Concept
              |
              v
          wiki-recall --v2
          tree + dense + BM25 + graph + lifecycle
              |
              v
          health / eval / gate
```

### LLM과 embedding 모델은 서로 다르다

| 구분 | 담당 | 설정 | 다운로드 가능성 |
|---|---|---|---|
| Codex/Claude agent | Concept 추출, Topic 배치, 관계 의미 판단 | `[v2] agent = "codex"` 또는 `"claude"` | 각 CLI가 사용하는 원격 모델. llm-wiki가 모델 파일을 받지 않는다. |
| Concept embedding | Concept 의미 유사도 계산 | `WIKI_V2_EMBED_BACKEND=hash` 또는 `qwen` | `qwen`을 처음 사용하면 Qwen embedding 모델을 받을 수 있다. |
| legacy page embedding | v1 page/chunk 검색 | `wiki-embed` | 처음 실행하면 Qwen3 Embedding 모델을 받을 수 있다. |
| cross-encoder reranker | 검색 후보의 최종 관련성 재정렬 | `--rerank 10` 또는 `--auto` | 처음 실행하면 BGE reranker 모델을 받을 수 있다. |

`agent = "codex"`는 embedding 모델을 선택하는 설정이 아니다. Codex는 문장의 의미를 판단하고, Qwen embedding은 문장을 숫자 vector로 바꾼다.

## 3. 폴더와 artifact 구조

예시 vault:

```text
my-vault/
  wiki.toml
  domain/
    architecture.md
    retry-policy.md
  patterns/
  entities/
  raw/
  index.yaml
  .embeddings/
  .llm_wiki_v2/
    documents.jsonl
    chunks.jsonl
    concepts.jsonl
    concept_build_state.json
    schemas.json
    cache/
      concepts/
    concept_embeddings/
      vectors.npy
      meta.json
      model.txt
    net/
      nodes.jsonl
      edges.jsonl
      proposals.jsonl
      review_queue.jsonl
      operations.jsonl
      NET.md
```

직접 수정해야 하는 것은 Markdown과 `wiki.toml`이다. `.embeddings/`와 `.llm_wiki_v2/`는 명령으로 다시 만드는 파생 artifact이므로 일반적으로 직접 편집하지 않는다.

## 4. 설치와 실행 환경

### 4.1 프로젝트 폴더로 이동

```powershell
cd "C:\path\to\llm-wiki-v2"
```

### 4.2 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

가상환경은 llm-wiki 전용 Python 패키지와 PyTorch를 다른 프로젝트의 패키지와 분리한다. 이미 `.venv`가 정상적으로 준비되어 있다면 새로 만들 필요가 없다.

### 4.3 기본 설치

```powershell
pip install -e .
```

Embedding과 reranker까지 사용하려면 ML dependency가 필요하다.

```powershell
pip install -e ".[ml]"
```

`pip install` 도중에는 아직 사용할 vault 경로를 알 수 없고 pip 자체도 비대화형 설치를 지원해야 하므로 선택 메뉴를 띄우지 않는다. 대신 아래에서 `WIKI_VAULT`를 지정한 뒤 처음 실제 `wiki-*` 명령을 실행할 때 한 번만 초기화 메뉴가 나타난다.

NVIDIA GPU를 사용할 때는 현재 CUDA 환경에 맞는 PyTorch가 설치되어 있어야 한다. 다음 명령으로 확인한다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4.4 vault 지정

한 번의 명령은 vault 하나를 대상으로 한다. 여러 vault를 사용할 때는 명령을 실행하기 전에 `WIKI_VAULT`를 바꾸거나, `--vault`를 지원하는 명령에 경로를 직접 전달한다.

```powershell
$env:WIKI_VAULT = "C:\path\to\my-vault"
```

PowerShell 창을 닫으면 이 설정은 사라진다. 다른 vault를 사용할 때는 값을 바꾸면 된다.

```powershell
$env:WIKI_VAULT = "D:\wiki\work-vault"
```

## 5. wiki.toml 설정

### 첫 실행 자동 초기화

vault에 `wiki.toml`이 없는 상태에서 `wiki-concepts build`, `wiki-net build` 같은 실제 명령을 처음 실행하면 터미널에 다음 메뉴가 나타난다.

```text
llm-wiki first-run setup
Vault: C:\path\to\my-vault
AI agent를 선택하세요:
  1. Codex (권장)
  2. Claude
선택 [1]:

Concept embedding 장치를 선택하세요:
  1. CUDA (감지됨, 권장)
  2. CUDA (NVIDIA GPU)
  3. CPU
  4. MPS (Apple Silicon)
  5. AUTO
선택 [1=cuda]:
```

`1`, `codex` 또는 Enter를 입력하면 Codex를 선택한다. `2` 또는 `claude`를 입력하면 Claude를 선택한다. 선택 직후 vault 최상위에 다음 최소 설정이 자동 생성되고, 사용자가 원래 입력한 명령이 계속 실행된다.

```toml
[vault]
content_dirs = ["."]

[v2]
enabled = true
agent = "codex"
embed_backend = "qwen"
embed_device = "cuda"
```

`content_dirs = ["."]`는 vault 아래의 Markdown을 재귀적으로 찾는 간단한 기본값이다. 기존 `wiki.toml`이 있으면 자동 초기화는 아무것도 수정하지 않는다. `--help`만 실행할 때도 파일을 생성하지 않는다.

CI, script, Codex 자동 실행처럼 stdin이 터미널이 아닌 환경에서는 선택을 기다리지 않고 초기화 방법을 출력한 뒤 종료한다. 이때는 먼저 다음 중 하나를 실행한다.

```powershell
wiki-init --agent codex
wiki-init --agent claude
wiki-init --vault "D:\wiki\work-vault" --agent codex --device cuda
```

### 선택 설정 확장

기본값을 변경해야 할 때만 자동 생성된 `wiki.toml`에 다음 항목을 추가한다.

```toml
[v2]
enabled = true
agent = "codex"
embed_backend = "qwen"
embed_device = "auto"
chunk_target_chars = 700
relation_candidate_topk = 10
safe_relation_min_confidence = 0.90
allow_ai_topic_creation = true
require_user_approval = ["CONTRADICTS", "SUPERSEDES", "OVERRIDES"]
```

### v2 설정 항목

| 설정 | 기본값 | 설명 |
|---|---:|---|
| `enabled` | `false` | v2 health와 ingest 단계를 활성화한다. |
| `agent` | 없음 | `codex` 또는 `claude`. 생략하면 외부 LLM을 호출하지 않는 규칙 기반 offline adapter를 사용한다. |
| `embed_backend` | `hash` | `qwen`은 실제 semantic embedding, `hash`는 모델 없는 smoke test다. 자동 초기화는 `qwen`을 기록한다. |
| `embed_device` | `auto` | `auto`, `cuda`, `mps`, `cpu`. `auto`는 CUDA → MPS → CPU 순으로 선택한다. |
| `chunk_target_chars` | `700` | Chunk 목표 글자 수. 단일 oversized paragraph는 강제로 자르지 않는다. |
| `relation_candidate_topk` | `10` | Concept 하나당 LLM이 비교할 관련 후보 수다. 모든 Concept pair를 비교하지 않는다. |
| `safe_relation_min_confidence` | `0.90` | safe relation을 자동 commit할 최소 confidence다. |
| `allow_ai_topic_creation` | `true` | LLM이 적절한 Topic/Collection이 없을 때 새 node를 제안할 수 있게 한다. |
| `require_user_approval` | 위험 관계 3종 | 지정한 relation이 반드시 review queue를 거치게 한다. |

### Codex와 Claude 연결

```toml
[v2]
enabled = true
agent = "codex"
```

Codex CLI가 설치되어 있고 로그인되어 있으면 llm-wiki가 `codex exec`를 read-only/ephemeral structured-output 모드로 자동 호출한다.

```toml
[v2]
enabled = true
agent = "claude"
```

Claude CLI가 설치되어 있고 로그인되어 있으면 `claude -p` structured output을 자동 호출한다.

일반 사용자는 `WIKI_V2_LLM_COMMAND`나 모델명을 설정할 필요가 없다. 예전에 설정한 override가 남아 있다면 제거한다.

```powershell
Remove-Item Env:WIKI_V2_LLM_COMMAND -ErrorAction SilentlyContinue
Remove-Item Env:WIKI_V2_LLM_MODEL -ErrorAction SilentlyContinue
```

### Concept embedding backend

모델 다운로드 없이 기능만 확인할 때:

```toml
[v2]
embed_backend = "hash"
```

실제 의미 검색을 사용할 때:

```toml
[v2]
embed_backend = "qwen"
embed_device = "cuda"
```

`hash`는 deterministic offline 테스트 backend다. 의미 품질과 자동 answer를 사용하려면 `qwen`을 사용해야 한다. `hash` 상태에서 `--auto`는 안전을 위해 `answer` 대신 `review`로 닫힌다. `WIKI_V2_EMBED_BACKEND`와 `WIKI_EMBED_DEVICE` 환경변수는 TOML 값을 일시적으로 덮어쓰는 고급 override로만 사용한다.

## 6. Markdown 작성 형식

일반적인 Document 예시:

```markdown
---
id: retry-policy
layer: domain
domain: tooling
projects: [my-project]
tags: [retry, policy]
confidence: confirmed
status: active
summary: Production retry policy
updated: 2026-08-17
---
# Retry Policy

The production retry count is 2.

## History

The 2025 retry count was 3.
```

`id`는 vault 안에서 고유해야 한다. `updated`는 historical query와 `SUPERSEDES` 판단에 사용될 수 있으므로 실제 날짜가 있을 때만 기록한다.

## 7. 처음 실행하는 권장 순서

### v2 핵심 기능만 사용할 때

```powershell
wiki-concepts build
wiki-net build
wiki-review
wiki-recall --v2 "현재 retry 횟수는?" --k 5
wiki-health --v2
```

이것이 frontmatter 없는 일반 Markdown vault를 포함한 **기본 v2 경로**다. `wiki-concepts build`가 Markdown을 직접 읽으므로 `wiki-index`, `index.yaml`, `wiki-embed`, `.embeddings/`가 필요하지 않다. `wiki-health --v2`는 legacy 검사를 제외하고 v2 source hash, Chunk, Concept index, NET만 검사한다.

### legacy와 v2를 모두 포함한 전체 pipeline

```powershell
wiki-index
wiki-embed
wiki-concepts build
wiki-net build
wiki-review
wiki-recall --v2 "현재 retry 횟수는?" --k 5
wiki-health --mode full
```

legacy page 검색도 함께 사용하려는 경우에만 각 파일을 legacy frontmatter 규격으로 작성하고 `wiki-index`와 `wiki-embed`를 실행한다.

## 8. 모든 기본 명령 요약

| 명령 | 역할 |
|---|---|
| `wiki-init` | vault에 최소 v2 `wiki.toml`을 만들고 Codex/Claude를 선택한다. |
| `wiki-index` | Markdown frontmatter를 검증하고 legacy `index.yaml`을 만든다. |
| `wiki-embed` | legacy page/chunk embedding store인 `.embeddings/`를 만든다. |
| `wiki-concepts build` | v2 Chunk, Concept, Concept embedding index를 만든다. |
| `wiki-net build` | Topic/Collection/Document/Concept NET과 relation proposal을 만든다. `tree`, `visualize`, `export`로 구조를 확인한다. |
| `wiki-review` | 위험 relation과 낮은 confidence proposal을 검토한다. |
| `wiki-recall` | legacy page 검색 또는 `--v2` Concept/NET 검색을 수행한다. |
| `wiki-health` | source와 파생 artifact의 drift 및 invariant를 검사한다. |
| `wiki-eval` | gold set으로 검색 및 v2 품질 metric을 측정한다. |
| `wiki-gate` | 현재 metric이 baseline보다 나빠졌는지 검사한다. |
| `wiki-lint` | 측정 근거가 약해 보이는 문장을 warning으로 찾는다. |

### `wiki-init`

대부분의 사용자는 첫 명령에서 나타나는 자동 메뉴만 사용하면 된다. 명시적으로 먼저 초기화하거나 자동화 환경에서 agent를 지정할 때 사용한다.

```powershell
wiki-init
wiki-init --agent codex
wiki-init --vault "C:\path\to\vault" --agent claude --device cuda
```

| 옵션 | 기능 |
|---|---|
| `--vault PATH` | 초기화할 vault. 생략하면 `WIKI_VAULT`, 현재 위치 순으로 결정한다. |
| `--agent codex|claude` | 메뉴를 생략하고 agent를 직접 지정한다. 비대화형 환경에서는 필수다. |
| `--device auto|cuda|mps|cpu` | Qwen Concept embedding 장치를 지정한다. `--agent`만 지정하면 기본은 `auto`다. |

이미 `wiki.toml`이 있으면 내용을 덮어쓰지 않고 해당 경로만 알려준다.

## 9. `wiki-index`

### 기능

Markdown의 YAML frontmatter를 검증하고 `index.yaml`을 다시 생성한다. 이 명령은 Concept나 embedding을 만들지 않는다.

> `wiki-index`는 **v2의 필수 단계가 아니다.** 파일이 `# 제목`으로 바로 시작하는 일반 Markdown이라면 실행하지 않는다. legacy 검색은 모든 파일에 YAML frontmatter, filename stem과 동일한 ASCII kebab-case `id`, 필수 metadata를 요구한다.

### 기본 실행

```powershell
wiki-index
```

### 특정 vault 지정

```powershell
wiki-index --vault "C:\path\to\vault"
```

### 변경하지 않고 검사만 하기

```powershell
wiki-index --check --vault "C:\path\to\vault"
```

`--check`는 현재 Markdown으로 만들어야 할 index와 저장된 `index.yaml`을 비교한다. 다르면 exit code 1과 함께 `wiki-index` 실행을 요구한다.

### 주요 실패 원인

- 필수 frontmatter 누락
- 중복 `id`
- 허용되지 않은 layer/domain 값
- `--check`에서 stale index 발견

## 10. `wiki-embed`

### 기능

legacy v1 검색이 사용하는 `.embeddings/`를 만든다. Markdown page body를 chunk로 나누고 Qwen3 Embedding vector를 저장한다.

이 저장소는 v2의 `.llm_wiki_v2/concept_embeddings/`와 별개다.

### 증분 실행

```powershell
wiki-embed
```

변경된 page만 다시 embedding하고 변경되지 않은 vector는 재사용한다.
embedding을 계산하는 동안 처리한 batch 수, 전체 batch 수, 속도와 예상 남은 시간이 tqdm 진행바로 표시된다.

### 전체 재생성

```powershell
wiki-embed --full
```

모델, chunk schema, metadata schema가 바뀌었거나 store가 깨진 경우 사용한다. 첫 실행에는 embedding 모델 다운로드가 발생할 수 있다.

### GPU 선택

```powershell
$env:WIKI_EMBED_DEVICE = "cuda"
wiki-embed --full
```

CPU를 강제하려면 `cpu`를 사용한다.

```powershell
$env:WIKI_EMBED_DEVICE = "cpu"
```

`wiki-embed`에는 `--vault` 옵션이 없으므로 `WIKI_VAULT`를 먼저 설정해야 한다.

## 11. `wiki-concepts build`

### 기능

한 번의 명령으로 다음 작업을 수행한다.

1. Markdown을 Document로 등록한다.
2. `# -> ## -> paragraph` 규칙으로 Chunk를 만든다.
3. Codex/Claude agent가 각 Chunk에서 Atomic Concept를 추출한다.
4. exact source quote와 schema를 검증한다.
5. 이전 Concept와 stable id로 reconcile한다.
6. Concept dense vector와 BM25용 metadata를 만든다.

### 기본 실행

```powershell
wiki-concepts build
```

실행 중 `phase 1/2`는 Codex/Claude가 Chunk별 Atomic Concept를 추출하는 단계다. Codex/Claude CLI 추론은 로컬 NVIDIA GPU에서 실행되지 않으므로 이 구간의 GPU 사용률이 낮은 것이 정상이다. `phase 2/2`에서 Qwen embedding을 계산할 때 `embed_device = "cuda"`가 로컬 GPU를 사용한다.

`phase 1/2`에는 완료 Chunk 수, 전체 Chunk 수, 처리 속도, 예상 남은 시간과 현재 파일이 한 줄짜리 진행바로 표시된다. `phase 2/2`의 Qwen embedding도 batch 단위 진행바를 표시한다. 출력이 파일로 redirect되거나 CI처럼 대화형 터미널이 아니면 동적 진행바는 자동으로 숨겨진다.

### 특정 vault

```powershell
wiki-concepts build --vault "C:\path\to\vault"
```

### 변경된 문서만 처리

```powershell
wiki-concepts build --changed
```

`--changed`라도 다음 항목이 바뀌면 자동으로 full rebuild로 승격된다.

- Chunk target
- artifact schema
- Concept prompt version
- LLM agent/model identity

### Chunk 목표 크기 일시 변경

```powershell
wiki-concepts build --chunk-target 900
```

일회성 override다. 일반적으로는 `wiki.toml`의 `chunk_target_chars`를 수정하는 편이 낫다.

### 생성 파일

- `.llm_wiki_v2/documents.jsonl`
- `.llm_wiki_v2/chunks.jsonl`
- `.llm_wiki_v2/concepts.jsonl`
- `.llm_wiki_v2/concept_embeddings/`
- `.llm_wiki_v2/cache/concepts/`

## 12. `wiki-net build`

### 기능

Concept를 Topic에 배치하고, Collection/Document 계층을 구성하고, 관련 Concept 후보를 찾아 relation proposal을 생성한다.

실행 중 다음 진행바가 순서대로 표시된다.

- `NET placement`: 아직 Topic이 없는 Concept 배치
- `Relation candidates`: 각 Concept의 관계 후보 검색
- `Relation analysis`: 후보 Concept pair의 관계 및 시간성 판정

각 진행바에는 완료 수, 전체 수, 처리 속도, 예상 남은 시간과 현재 처리 항목이 표시된다.

```powershell
wiki-net build
```

특정 vault:

```powershell
wiki-net build --vault "C:\path\to\vault"
```

이번 실행에서 AI Topic 생성을 막으려면:

```powershell
wiki-net build --no-ai-topic-creation
```

기존 사용자의 Topic 이름, 문서 위치, Collection, 승인된 relation은 rebuild 후에도 보존된다. Markdown, Chunk, Concept가 삭제된 경우에는 더 이상 존재하지 않는 source-derived node와 dangling edge가 정리된다.

### Relation 정책

| Relation | 의미 | 자동 commit |
|---|---|---|
| `SUPPORTS` | source가 target을 뒷받침 | confidence가 기준 이상이면 가능 |
| `COMPLEMENTS` | 양립 가능하며 서로 보완 | confidence가 기준 이상이면 가능 |
| `DUPLICATE_OF` | 실질적으로 같은 지식 | confidence가 기준 이상이면 가능. 물리 삭제는 안 함 |
| `CONTRADICTS` | 같은 subject/scope에서 동시에 참일 수 없음 | 불가. 항상 review |
| `SUPERSEDES` | source가 더 최신이며 target을 대체 | 불가. 항상 review |
| `OVERRIDES` | 시간보다 scope/priority 때문에 source가 우선 | 불가. 항상 review |

## 13. `wiki-net` 구조 편집 명령

모든 사용자 구조 변경은 `.llm_wiki_v2/net/operations.jsonl`에 기록된다. Markdown 원문은 수정하거나 삭제하지 않는다.

### NET 내보내기

```powershell
wiki-net export
```

기본 출력은 `.llm_wiki_v2/net/NET.md`다.

```powershell
wiki-net export --out "C:\path\to\NET.md"
```

### 터미널에서 Tree 보기

```powershell
wiki-net tree
```

기본 출력은 NET의 엄격한 계층 backbone인 `Topic → Collection/하위 Topic → Document`다. 각 Document 옆에는 소속 Concept 개수가 표시된다. Concept까지 펼쳐 보려면:

```powershell
wiki-net tree --show-concepts
```

사용 가능한 옵션:

| 옵션 | 기능 |
|---|---|
| `--show-concepts` | Document 아래 Atomic Concept를 펼친다. Concept가 많으면 출력이 길어진다. |
| `--show-ids` | label 옆에 `topic:...`, `document:...`, `concept:...` ID를 표시한다. 구조 편집 명령에 넣을 ID를 찾을 때 유용하다. |
| `--max-depth N` | root를 깊이 0으로 보고 N단계까지만 표시한다. 숨겨진 자식 수를 함께 표시한다. |
| `--active-only` | archived/superseded/disputed 등 ACTIVE가 아닌 node를 제외한다. |
| `--ascii` | `├──` 대신 `|--` 문자를 사용한다. 터미널에서 선 문자가 깨질 때 사용한다. |
| `--vault PATH` | 이번 명령에서 사용할 vault를 직접 지정한다. |

```powershell
wiki-net tree --show-concepts --show-ids
wiki-net tree --max-depth 2 --active-only
wiki-net tree --show-concepts --ascii --vault "C:\path\to\vault"
```

Tree는 한눈에 읽을 수 있도록 `PARENT_OF`, `CONTAINS_DOCUMENT`, `DOCUMENT_HAS_CONCEPT`만 표시한다. Concept 간 semantic relation과 primary/secondary Topic membership은 마지막에 생략된 edge 개수로 알리고, 실제 선은 다음 HTML 그래프에서 확인한다.

### 인터랙티브 Graph 보기

```powershell
wiki-net visualize --open
```

기본적으로 `.llm_wiki_v2/net/NET.html`을 만들며 `--open`을 사용하면 운영체제의 기본 브라우저에서 바로 연다. HTML 하나에 데이터, 스타일, 동작 코드가 모두 들어 있으므로 CDN, 인터넷 연결, 별도 웹 서버가 필요 없다.

화면에서 할 수 있는 작업:

- label 또는 ID 검색
- Topic, Collection, Document, Concept 유형별 표시 전환
- hierarchy/membership edge와 semantic relation edge 각각 표시 전환
- 마우스 휠 확대/축소와 빈 공간 드래그 이동
- node 드래그 재배치
- node 클릭 후 state, ID, 속성 확인
- 전체 graph가 화면에 들어오도록 `Fit graph`

```powershell
wiki-net visualize --active-only
wiki-net visualize --out "C:\path\to\NET.html" --open
```

| 옵션 | 기능 |
|---|---|
| `--out PATH` | HTML 출력 경로를 바꾼다. |
| `--open` | 생성된 HTML을 기본 브라우저에서 연다. |
| `--active-only` | ACTIVE node와 그 node 사이의 edge만 HTML에 포함한다. |
| `--vault PATH` | 이번 명령에서 사용할 vault를 직접 지정한다. |

`wiki-net graph`는 `wiki-net visualize`의 짧은 alias이므로 같은 방식으로 사용할 수 있다. 시각화와 Tree 명령은 NET을 읽기만 하며 Markdown, node, edge, operation log를 변경하지 않는다.

### Collection 생성

```powershell
wiki-net create-collection `
  --id monthly-reports `
  --label "Monthly Reports" `
  --parent topic:business `
  --type monthly `
  --actor louis
```

`--parent`를 생략하면 `topic:knowledge` 아래에 생성된다.

### Topic 이름 변경

```powershell
wiki-net rename-topic --id topic:frontend --label "Frontend Platform" --actor louis
```

### Topic 이동

```powershell
wiki-net move-topic --id topic:frontend --target topic:engineering --actor louis
```

`--target`은 새 parent Topic이다. cycle이나 다중 parent가 생기면 거부된다.

### Topic 병합

```powershell
wiki-net merge-topic --id topic:old-frontend --target topic:frontend --actor louis
```

source Topic의 child, membership, document placement를 target으로 이동하고 source는 `ARCHIVED` 처리한다.

### Topic 삭제와 복구

```powershell
wiki-net delete-topic --id topic:frontend --actor louis
wiki-net restore-topic --id topic:frontend --actor louis
```

삭제는 Topic을 `ARCHIVED` 처리할 뿐 Markdown, Document, Concept를 물리 삭제하지 않는다.

### Document 이동

```powershell
wiki-net move-document --id architecture --target topic:engineering --actor louis
```

`--id`에는 `architecture` 또는 `document:architecture`를 사용할 수 있다. Document는 항상 Topic 또는 Collection 위치를 정확히 하나만 가진다.

### Concept primary Topic 변경

```powershell
wiki-net primary-topic --id concept:abc123 --target topic:frontend --actor louis
```

Concept object를 복제하지 않고 primary membership만 바꾼다.

### secondary Topic 추가와 제거

```powershell
wiki-net add-secondary-topic --id concept:abc123 --target topic:architecture --actor louis
wiki-net remove-secondary-topic --id concept:abc123 --target topic:architecture --actor louis
```

primary Topic과 같은 Topic을 secondary로 중복 지정할 수 없다.

### 마지막 사용자 작업 되돌리기

```powershell
wiki-net undo --actor louis
```

이미 undo된 작업과 system build operation을 제외하고 가장 최근 사용자 작업을 정확한 이전 NET snapshot으로 되돌린다. 원 operation과 UNDO record는 감사 목적으로 계속 남는다.

## 14. `wiki-review`

### review queue 보기

```powershell
wiki-review
```

다음 정보가 함께 표시된다.

- relation type과 confidence
- same subject / same scope
- temporal change 가능성
- 판단 이유와 evidence
- source/target Concept text와 source quote

### proposal 그대로 승인

```powershell
wiki-review --approve "review:proposal-id" --actor louis
```

`SUPERSEDES` 승인 시 newer Concept는 `ACTIVE`, target Concept는 `SUPERSEDED`가 된다. `CONTRADICTS` 승인 시 관련 Concept가 `DISPUTED`가 된다.

### 거절

```powershell
wiki-review --reject "review:proposal-id" --actor louis
```

거절된 item은 terminal 상태가 되며 나중에 다시 승인할 수 없다.

### 충돌 상황을 명시적으로 해결

```powershell
wiki-review --resolve "review:proposal-id" --decision source-current --actor louis
```

| decision | 의미 | 처리 |
|---|---|---|
| `source-current` | source가 현재 우선 | source가 target을 `OVERRIDES`하도록 승인 |
| `target-current` | target이 현재 우선 | 방향을 뒤집어 target이 source를 `OVERRIDES`하도록 승인 |
| `different-scope` | 둘 다 맞지만 scope가 다름 | 충돌 proposal 거절 |
| `disputed` | 실제 충돌이며 아직 해결 불가 | `CONTRADICTS` commit, 관련 Concept를 `DISPUTED` 처리 |
| `unrelated` | 관계 없음 | proposal 거절 |

## 15. `wiki-recall`

### legacy page 검색

```powershell
wiki-recall "embedding store rebuild" --k 5
```

legacy `.embeddings/`가 필요하다.

주요 legacy 필터:

```powershell
wiki-recall "query" --layer domain --domain tooling --project my-project
wiki-recall "query" --confidence confirmed --status active
wiki-recall "query" --mode dense
wiki-recall "query" --full
```

`--layer`, `--domain`, `--project`, `--confidence`, `--mode`, `--full`은 현재 legacy page 검색용 옵션이다.

### v2 Concept/NET 검색

```powershell
wiki-recall --v2 "현재 retry 횟수는?" --k 5
```

특정 vault를 직접 지정할 수 있다.

```powershell
wiki-recall --v2 "현재 retry 횟수는?" --vault "C:\path\to\vault"
```

검색 신호:

1. Topic label 기반 tree soft routing
2. Concept dense similarity
3. BM25 sparse score
4. Dense 2.0 + Sparse 1.0 weighted RRF
5. committed relation graph 확장
6. ACTIVE/SUPERSEDED/DISPUTED lifecycle
7. optional cross-encoder rerank

### 결과 개수

```powershell
wiki-recall --v2 "질문" --k 10
```

### JSON 출력

```powershell
wiki-recall --v2 "질문" --json
```

### reranker 사용

```powershell
wiki-recall --v2 "질문" --k 5 --rerank 10
```

상위 10개 후보를 local cross-encoder로 다시 정렬한 뒤 5개를 출력한다. 첫 실행에는 reranker 모델 다운로드가 발생할 수 있다.

### 과거 지식 검색

```powershell
wiki-recall --v2 "2025년 retry 정책은?" --historical
```

기본 current query에서는 `SUPERSEDED`, `DUPLICATE`, `ARCHIVED`를 제외한다. `--historical`은 과거 Concept를 허용하고 질문에 포함된 연도를 `updated_at`과 비교해 soft ranking signal로 사용한다.

`--status any`도 v2에서 historical 검색을 활성화하지만, 명확성을 위해 `--historical` 사용을 권장한다.

### answer / review / none 자동 결정

```powershell
wiki-recall --v2 "현재 retry 횟수는?" --auto --json
```

| decision | 의미 |
|---|---|
| `answer` | absolute cosine, top-two margin, reranker가 모두 기준을 통과한 하나의 Concept |
| `review` | 관련 후보는 있지만 자동 확정하기에는 불확실함 |
| `none` | 최소 confidence에도 도달하지 못했거나 후보가 없음 |

다음 상황에서는 fail closed한다.

- hash Concept embedding 사용
- reranker unavailable
- threshold file 누락 또는 오류
- top-two margin 부족
- 관련 `DISPUTED` evidence 존재
- stale artifact 또는 깨진 NET 발견

threshold 파일을 직접 지정할 수 있다.

```powershell
wiki-recall --v2 "질문" --auto --thresholds "C:\path\to\auto_thresholds.json" --json
```

## 16. `wiki-health`

### v2 전용 검사

```powershell
wiki-health --v2
```

일반 Markdown을 사용하는 v2 vault의 기본 명령이다. legacy `index.yaml`, YAML frontmatter, `.embeddings/`, legacy graph/community report는 검사하지 않는다. live Markdown source hash, provenance, Chunk/Concept, Concept vector index, NET, relation approval와 lifecycle 무결성을 검사한다.

### CI 모드

```powershell
wiki-health --mode ci
```

로컬 `.embeddings/` 검사를 생략한다. index, Markdown hygiene, reports, v2 artifact는 계속 검사한다.

### full 모드

```powershell
wiki-health --mode full
```

legacy `.embeddings/`까지 검사한다. `wiki-embed`를 하지 않았다면 `embedding-store-missing`으로 실패하는 것이 정상이다.

### 특정 vault와 JSON 출력

```powershell
wiki-health --mode full --vault "C:\path\to\vault" --json
```

### 주요 v2 검사

- live Markdown와 Document hash drift
- Chunk/Concept provenance와 source quote
- prompt/schema/model/config identity
- Concept vector row/hash/model mismatch
- Topic cycle, missing parent, invalid node nesting
- Document 위치 정확히 1개
- Concept primary Topic 정확히 1개
- dangling/self relation
- 승인 trace 없는 위험 relation
- `SUPERSEDES` cycle과 successor 없는 `SUPERSEDED`
- 근거 없는 `DISPUTED`
- 손상된 UNDO operation

health error가 있으면 v2 recall도 조용히 검색을 계속하지 않고 rebuild를 요구한다.

## 17. `wiki-eval`

### legacy gold schema만 검사

```powershell
wiki-eval --validate-only
```

Embedding을 실행하지 않고 gold schema와 coverage를 검사한다.

### legacy 검색 품질 측정

```powershell
wiki-eval --k 10 --mode hybrid
wiki-eval --k 10 --rerank 10 --json
wiki-eval --split test
```

### legacy auto threshold 보정

```powershell
wiki-eval --calibrate "C:\path\to\auto_thresholds.json"
```

calibration split으로 false automatic answer가 없도록 threshold를 맞춘다.

### v2 gold 검사와 평가

```powershell
wiki-eval --v2 --validate-only
wiki-eval --v2 --k 8 --json
```

`wiki-eval --v2`는 `WIKI_VAULT`로 vault를 찾는다. 기본 gold 파일은 `.llm_wiki_v2/v2_gold.json`이다.

간단한 v2 gold 예시:

```json
{
  "concepts": [
    {"concept_id": "concept:abc", "source_quote": "The retry count is 2."}
  ],
  "placements": [
    {"concept_id": "concept:abc", "primary_topic_id": "topic:operations"}
  ],
  "relations": [],
  "supersessions": [],
  "queries": [
    {"query": "현재 retry 횟수는?", "expect": ["concept:abc"], "historical": false, "auto": true}
  ]
}
```

주요 v2 metric:

- Concept precision / recall / faithfulness
- primary placement accuracy / top-k route recall
- relation 전체 및 type별 precision / recall / F1
- supersession precision / recall / false rate
- current Hit@k / MRR / current fact accuracy
- historical accuracy
- false auto answer / outdated answer
- human review rate / approval rate
- 승인되지 않은 risky relation 수

## 18. `wiki-gate`

### v2 baseline 최초 생성

```powershell
wiki-gate --v2 --update
```

기본 위치는 `.llm_wiki_v2/v2_eval_baseline.json`이다. 측정 결과가 의도한 품질임을 확인한 후에만 baseline을 갱신한다.

### v2 regression 검사

```powershell
wiki-gate --v2
```

다음 상황에서 실패한다.

- quality metric이 허용 범위 이상 하락
- outdated answer rate 증가
- false supersession rate 증가
- false auto answer 증가
- 승인되지 않은 risky relation 존재

### 옵션

```powershell
wiki-gate --v2 --k 10 --tolerance-pp 2
wiki-gate --v2 --baseline "C:\path\to\baseline.json"
wiki-gate --v2 --json
```

legacy gate도 같은 명령에서 `--v2` 없이 사용한다.

```powershell
wiki-gate
wiki-gate --update
```

## 19. `wiki-lint`

측정 근거 없이 애매하게 표현된 claim 후보를 찾는다. 자동으로 문서를 수정하지 않으며 exit code도 실패로 만들지 않는 warning 도구다.

```powershell
wiki-lint
wiki-lint --page retry-policy
wiki-lint --limit 50
wiki-lint --vault "C:\path\to\vault"
```

언어 pack은 `wiki.toml`에서 설정한다.

```toml
[lint]
packs = ["en", "ko"]
```

## 20. 보조 명령

다음 명령은 console script가 아니라 `python -m`으로 실행한다.

### legacy contradiction 후보

```powershell
python -m llm_wiki.hygiene.contradict --tau 0.78 --topk 3
python -m llm_wiki.hygiene.contradict --changed "page-a,page-b"
python -m llm_wiki.hygiene.contradict --only-corrected
```

실제 contradiction을 확정하는 도구가 아니라 관련 page 후보를 만드는 legacy 도구다. v2의 Concept relation pipeline과는 별개다.

### legacy compaction 후보

```powershell
python -m llm_wiki.hygiene.compact --merge-tau 0.85 --size-kb 12 --corrections 6
```

UPDATE, MERGE, SPLIT 검토 후보를 생성한다. 자동 병합이나 삭제는 하지 않는다.

### graph report

```powershell
python -m llm_wiki.reports.graph_report --write
python -m llm_wiki.reports.graph_report --json
python -m llm_wiki.reports.graph_report --suggest --tau 0.55
```

Markdown wikilink graph를 분석한다. v2 semantic NET과는 다른 report다.

### community report

```powershell
python -m llm_wiki.reports.community_report --write
python -m llm_wiki.reports.community_report --stale
```

`--stale`은 grounded synthesis가 필요한 community를 출력한다.

### Codex/Claude project memory 동기화

```powershell
python -m integrations.agent_memory.sync_cache `
  --project "C:\path\to\project" `
  --target codex
```

가능한 target은 `claude`, `codex`, `both`다.

```powershell
python -m integrations.agent_memory.sync_cache --project "C:\path\to\project" --target both --dry-run
```

### 일일 ingest pipeline

```powershell
python -m integrations.ingest.ingest_pipeline --vault "C:\path\to\vault"
```

`[v2] enabled = true`이면 기존 index/embed/report 단계 사이에 `wiki-concepts build --changed`와 `wiki-net build`가 포함된다.

```powershell
python -m integrations.ingest.ingest_pipeline --dry-run --skip-llm
python -m integrations.ingest.ingest_pipeline --agent codex
python -m integrations.ingest.ingest_pipeline --stamp "C:\path\to\last-run-date"
```

ingest는 dirty Git working tree에서 시작하지 않으며, 모든 health 단계가 성공한 뒤에만 성공 stamp를 기록한다.

## 21. Concept와 NET 상태 이해하기

### Concept lifecycle

| State | 의미 | current 검색 |
|---|---|---|
| `ACTIVE` | 현재 유효한 지식 | 우선 검색 |
| `SUPERSEDED` | 더 최신 Concept에 대체됨 | 기본 제외 |
| `DISPUTED` | 해결되지 않은 충돌이 있음 | 검색 가능하지만 warning/review |
| `DUPLICATE` | canonical Concept와 실질적으로 동일 | 기본 제외 또는 대표 Concept 사용 |
| `ARCHIVED` | 보존하지만 현재 운용 지식은 아님 | 기본 제외 |

### 중요한 불변식

- Chunk는 NET node가 아니다.
- Concept는 Topic마다 복제하지 않는다.
- Concept는 primary Topic 1개와 secondary Topic 0개 이상을 가진다.
- Document는 Topic 또는 Collection 위치를 정확히 1개 가진다.
- Topic/Collection 구조 변경은 Markdown을 변경하지 않는다.
- `CONTRADICTS`, `SUPERSEDES`, `OVERRIDES`는 confidence와 무관하게 사용자 승인 전에는 commit되지 않는다.
- `SUPERSEDES` 방향은 newer Concept에서 older Concept 방향이다.

## 22. 자주 사용하는 작업별 명령

### Markdown을 수정한 뒤

```powershell
wiki-concepts build --changed
wiki-net build
wiki-review
wiki-health --v2
```

legacy page 검색도 사용한다면 legacy frontmatter를 유지하고 `wiki-index`, `wiki-embed`를 추가한다.

```powershell
wiki-index
wiki-embed
wiki-health --mode full
```

### agent를 Claude에서 Codex로 변경한 뒤

```toml
[v2]
enabled = true
agent = "codex"
```

```powershell
wiki-concepts build --changed
wiki-net build
```

agent/model identity가 바뀌었으므로 `--changed`를 사용해도 안전을 위해 Concept full rebuild로 자동 전환된다.

### Topic 구조를 정리한 뒤 실수했을 때

```powershell
wiki-net move-topic --id topic:a --target topic:b --actor louis
wiki-net undo --actor louis
```

### 현재 정책과 과거 정책을 각각 찾을 때

```powershell
wiki-recall --v2 "현재 retry 정책" --k 5
wiki-recall --v2 "2025년 retry 정책" --historical --k 5
```

## 23. 오류 메시지별 해결 방법

### `concept index missing` 또는 `concept index ... stale`

```powershell
wiki-concepts build
```

### `v2 NET ... stale` 또는 primary/document 위치 오류

```powershell
wiki-net build
wiki-health --v2
```

사용자가 만든 잘못된 구조라면 `wiki-net undo` 또는 명시적인 move/primary 명령으로 고친다.

### `v2 agent 'codex' ... not installed or not on PATH`

Codex CLI 설치와 로그인을 확인한다.

```powershell
codex --version
```

Claude를 선택했다면 다음을 확인한다.

```powershell
claude --version
```

### `embedding-store-missing`

`wiki-health --mode full`이 legacy `.embeddings/`를 찾지 못한 상태다.

```powershell
wiki-embed --full
```

일반 Markdown v2 vault에서는 다음을 사용한다.

```powershell
wiki-health --v2
```

legacy index는 검사하되 로컬 embedding만 생략하는 CI라면 `wiki-health --mode ci`를 사용한다.

### GPU가 있는데 CPU fallback

```powershell
$env:WIKI_EMBED_DEVICE = "cuda"
python -c "import torch; print(torch.cuda.is_available())"
```

`False`이면 CUDA 지원 PyTorch가 현재 `.venv`에 설치되지 않은 것이다. NVIDIA driver가 정상이어도 CPU용 PyTorch이면 CUDA를 사용할 수 없다.

### 모델 다운로드가 시작됨

어떤 명령인지 먼저 확인한다.

- `wiki-embed`: legacy Qwen embedding 모델
- `WIKI_V2_EMBED_BACKEND=qwen` 상태의 `wiki-concepts build`: Concept Qwen embedding 모델
- `--rerank` 또는 `--auto`: BGE cross-encoder reranker
- `agent = "codex"` 또는 `"claude"`: llm-wiki가 local model file을 다운로드하는 동작이 아님

### 위험 relation이 자동 적용되지 않음

정상 동작이다. queue를 확인한다.

```powershell
wiki-review
```

### live Markdown을 수정한 뒤 recall이 거부됨

stale artifact를 조용히 사용하지 않는 fail-closed 동작이다.

```powershell
wiki-concepts build --changed
wiki-net build
```

## 24. 도움말 확인

각 명령의 현재 옵션은 `--help`로 확인할 수 있다.

```powershell
wiki-index --help
wiki-embed --help
wiki-concepts --help
wiki-concepts build --help
wiki-net --help
wiki-net build --help
wiki-review --help
wiki-recall --help
wiki-health --help
wiki-eval --help
wiki-gate --help
wiki-lint --help
```

## 25. 가장 짧은 실사용 예시

```powershell
cd "C:\path\to\llm-wiki-v2"
.\.venv\Scripts\Activate.ps1
$env:WIKI_VAULT = "C:\path\to\my-vault"

wiki-concepts build
wiki-net build
wiki-review
wiki-recall --v2 "현재 프로젝트의 핵심 결정은?" --k 5 --rerank 10
wiki-health --v2
```

`wiki.toml`에는 다음만 있으면 Codex 연결이 자동으로 이루어진다.

```toml
[v2]
enabled = true
agent = "codex"
embed_backend = "qwen"
embed_device = "cuda"
```
