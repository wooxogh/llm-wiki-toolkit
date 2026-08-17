# llm-wiki v2 사용법

v2는 Markdown을 원본으로 유지한 채 `Document -> Chunk -> Atomic Concept -> NET` 파생 데이터를 만듭니다. 원본 Markdown은 바꾸지 않으며, 모든 v2 데이터는 vault의 `.llm_wiki_v2/` 아래에 저장됩니다.

## 시작

```powershell
cd "C:\path\to\llm-wiki-v2"
.\.venv\Scripts\Activate.ps1
pip install -e ".[ml]"
$env:WIKI_VAULT = "C:\path\to\your\vault"

wiki-concepts build
wiki-net build
wiki-recall --v2 "현재 React 결정" --k 5
wiki-health --v2
```

처음 `wiki-concepts build` 같은 실제 명령을 실행했는데 vault에 `wiki.toml`이 없으면 다음 선택 메뉴가 자동으로 나타납니다.

```text
llm-wiki first-run setup
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
```

선택하면 vault 최상위에 최소 `wiki.toml`이 자동 생성되고 원래 명령이 계속 실행됩니다. Enter만 누르면 Codex가 선택됩니다. 기존 `wiki.toml`은 절대 덮어쓰지 않습니다.

CI나 script처럼 입력할 수 없는 환경에서는 먼저 `wiki-init --agent codex` 또는 `wiki-init --agent claude`를 실행합니다. 특정 vault를 직접 지정할 수도 있습니다.

```powershell
wiki-init --vault "C:\path\to\your\vault" --agent codex --device cuda
```

## LLM 연결

자동 생성된 `wiki.toml`의 `[v2] agent`에 따라 현재 로그인된 Codex 또는 Claude CLI를 사용합니다. 별도 API 키, bridge 명령, model 이름을 입력할 필요가 없습니다.

```toml
[v2]
enabled = true
agent = "codex" # 또는 "claude"
embed_backend = "qwen"
embed_device = "cuda" # NVIDIA, Apple Silicon은 mps, 강제 CPU는 cpu
```

`codex`를 선택하면 `codex exec`를 read-only/ephemeral 모드로 호출하고, `claude`를 선택하면 `claude -p`의 structured JSON 출력을 사용합니다. 두 경우 모두 LLM은 Concept/placement/relation proposal만 반환하며 파일이나 NET 저장소를 직접 변경하지 않습니다. 해당 CLI가 설치되어 있고 로그인된 상태여야 합니다.

`agent`를 생략하면 모델 호출 없는 규칙 기반 adapter를 사용하므로 테스트와 오프라인 smoke run이 가능합니다. `WIKI_V2_LLM_COMMAND`와 `WIKI_V2_LLM_MODEL`은 사내 bridge나 특정 model identity를 강제로 써야 하는 고급 override로만 남아 있습니다.

자동 초기화는 실제 의미 검색을 위해 `embed_backend = "qwen"`과 선택한 `embed_device`를 기록합니다. `hash`는 모델 다운로드 없이 artifact와 pipeline만 검증하는 smoke test용입니다. `WIKI_V2_EMBED_BACKEND`와 `WIKI_EMBED_DEVICE` 환경변수는 TOML 설정을 한 번만 덮어쓰는 고급 override입니다.

`wiki-concepts build`의 `phase 1/2`는 Codex/Claude의 Chunk별 Concept 추출이라 로컬 GPU를 사용하지 않습니다. `phase 2/2`의 Qwen embedding에서만 CUDA/MPS가 사용됩니다. CLI는 Chunk 완료 수, 속도, 예상 남은 시간과 현재 파일을 tqdm 진행바로 표시하며, embedding도 batch 단위 진행바를 표시합니다.

요청의 `task`는 `extract_concepts`, `place_concept`, `classify_relation`, `resolve_temporal` 중 하나입니다. adapter는 proposal만 만들며, 저장소 변경과 위험 관계 승인에는 관여하지 못합니다.

## 주요 명령

`wiki-concepts build`는 Markdown을 heading-aware chunk로 나누고 Concept를 추출하여 인덱스를 만듭니다. `# -> ## -> paragraph` 순으로 나누며, 700자를 넘는 단일 문단은 의미를 보존하기 위해 자르지 않습니다. `--changed`는 문서 hash가 바뀐 부분만 처리하지만, chunk 크기·artifact schema·prompt version·LLM model identity가 바뀌면 오래된 Concept가 섞이지 않도록 자동으로 전체 rebuild로 전환됩니다.

문서를 추가하거나 수정한 뒤에는 incremental 빌드를 사용할 수 있습니다.

```bash
wiki-concepts build --changed
wiki-net build --changed
```

이 경로는 변경되지 않은 문서의 chunk와 Concept를 재사용하고, 새롭거나 변경된 Concept만 embedding합니다. NET 관계도 변경된 Concept가 포함된 후보만 다시 분석하며, 기존에 사용자가 승인한 관계와 Topic/Collection 구조는 유지합니다. 삭제된 문서의 Concept와 vector는 제거됩니다.

다음 변경이 있으면 안전을 위해 전체 rebuild가 필요합니다.

- embedding model 또는 vector dimension 변경
- chunk target, prompt, artifact schema, indexed text schema 변경
- identity 또는 provenance artifact 누락·불일치

일반적인 변경이 아닌 경우에는 `wiki-concepts build`와 `wiki-net build`를 인자 없이 실행해 전체 결과를 재생성할 수 있습니다.

`wiki-net build`는 Topic/Document/Concept 구조를 배치하고, 하이브리드 후보 탐색 후 관계 proposal을 생성합니다. `NET placement`, `Relation candidates`, `Relation analysis` 세 진행바로 현재 단계와 남은 시간을 확인할 수 있습니다. `SUPPORTS`, `COMPLEMENTS`, `DUPLICATE_OF`만 설정한 신뢰도 이상에서 자동 반영됩니다. `CONTRADICTS`, `SUPERSEDES`, `OVERRIDES`는 절대로 자동 반영되지 않습니다.

```powershell
wiki-review
wiki-review --approve "review:proposal-id" --actor louis
wiki-review --reject "review:proposal-id" --actor louis
wiki-review --resolve "review:proposal-id" --decision source-current --actor louis
wiki-review --resolve "review:proposal-id" --decision target-current --actor louis
wiki-review --resolve "review:proposal-id" --decision different-scope --actor louis
wiki-review --resolve "review:proposal-id" --decision disputed --actor louis
wiki-review --resolve "review:proposal-id" --decision unrelated --actor louis
```

목록 화면에는 source/target Concept의 정규화 문장과 원문 인용, relation, confidence, same subject/scope, 시간 변경 가능성, 판단 이유가 함께 표시됩니다. `SUPERSEDES`는 같은 subject/scope와 사용자 승인뿐 아니라 source가 더 최신이라는 metadata 또는 원문의 명시적 개정 근거가 있어야 저장됩니다.

`wiki-recall --v2 "질문" --k 5`는 tree, concept dense/text 인덱스, 확정 relation, lifecycle 신호를 함께 사용합니다. 기본 검색에서는 `SUPERSEDED`, `DUPLICATE`, `ARCHIVED`를 제외하고 `DISPUTED`에는 경고를 붙입니다. 과거 정책을 찾을 때만 `--historical`을 사용합니다.

목록 검색도 `--rerank 10`을 주면 상위 10개 evidence를 로컬 cross-encoder로 다시 정렬합니다. historical query에 연도가 포함되면 Concept의 `updated_at` 연도도 soft ranking 신호로 사용하며, tree 분류는 recall을 끊는 hard filter로 사용하지 않습니다.

`wiki-recall --v2 "질문" --auto --json`은 목록 대신 `answer`, `review`, `none` 중 하나를 반환합니다. 자동 `answer`에는 Qwen Concept embedding, 보정된 absolute cosine/margin threshold, cross-encoder 동의가 모두 필요합니다. 기본 hash embedding, reranker 실패, stale artifact, 관련 `DISPUTED` 근거는 자동 답변하지 않고 `review` 또는 `none`으로 닫힙니다.

```powershell
wiki-net export
wiki-net tree
wiki-net tree --show-concepts --show-ids
wiki-net visualize --open
wiki-net create-collection --id monthly-reports --label "Monthly Reports" --parent topic:business --type monthly
wiki-net rename-topic --id topic:frontend --label "Frontend" --actor louis
wiki-net move-topic --id topic:frontend --target topic:engineering
wiki-net merge-topic --id topic:old-frontend --target topic:frontend
wiki-net delete-topic --id topic:frontend
wiki-net restore-topic --id topic:frontend
wiki-net move-document --id architecture --target topic:engineering
wiki-net primary-topic --id concept:abc --target topic:frontend
wiki-net add-secondary-topic --id concept:abc --target topic:architecture
wiki-net remove-secondary-topic --id concept:abc --target topic:architecture
wiki-net undo
```

`wiki-net export`는 `.llm_wiki_v2/net/NET.md`에 Mermaid NET 다이어그램을 만듭니다. `wiki-net tree`는 터미널에 Topic → Collection → Document 계층을 출력하며, `--show-concepts`를 추가하면 Document 아래 Concept까지 표시합니다. `wiki-net visualize`는 검색, 노드 필터, 관계선 전환, 확대/축소, 이동, 상세 보기가 가능한 독립 HTML 파일 `.llm_wiki_v2/net/NET.html`을 만듭니다. `--open`을 붙이면 생성 후 기본 브라우저에서 바로 엽니다.

```powershell
wiki-net tree --max-depth 2 --active-only
wiki-net tree --show-concepts --ascii
wiki-net visualize --active-only --out "C:\path\to\NET.html" --open
```

Tree는 계층을 읽기 쉽게 보여주기 위해 semantic relation과 primary/secondary topic membership 선을 생략합니다. 모든 NET edge를 확인할 때는 HTML 시각화를 사용합니다. HTML은 CDN이나 인터넷 연결이 필요 없으며 원본 Markdown과 NET 데이터를 수정하지 않습니다. Topic 삭제/복구 및 재배치도 Markdown이나 Concept 원문을 지우지 않습니다. 모든 사용자 구조 작업은 변경 전후 NET 전체 상태와 함께 append-only operation log에 기록되므로 `wiki-net undo`가 생성·이동·병합·삭제·복구·membership 변경을 정확히 되돌립니다.

## 품질 검사

```powershell
wiki-health --v2
wiki-eval --v2 --validate-only
wiki-eval --v2
wiki-gate --v2 --update  # 측정 근거가 있을 때만 첫 baseline 생성
wiki-gate --v2
```

v2 gold 파일은 `.llm_wiki_v2/v2_gold.json`입니다. 필수 `concepts`, `relations`, `queries`와 선택 `placements`, `supersessions` 배열로 extraction precision/recall/faithfulness, primary placement, relation per-type 결과, supersession, current/historical retrieval, auto-answer safety, review/approval 비율을 측정합니다. gate는 품질 하락, outdated/false auto answer 증가, false supersession 증가, 승인 trace가 없는 위험 관계를 실패로 처리합니다.

일반 Markdown v2 vault에서는 `wiki-health --v2`를 사용합니다. Markdown source hash, provenance, exact source quote, schema/prompt/model/config identity, Concept index, tree cardinality/cycle, relation approval trace, lifecycle, undo log를 검사하되 legacy YAML frontmatter와 `index.yaml`, `.embeddings/`는 요구하지 않습니다. 문제가 있으면 `wiki-recall --v2`도 조용히 계속하지 않고 rebuild를 요구합니다.

`wiki-index`와 `wiki-embed`는 모든 문서에 legacy YAML frontmatter를 작성하고 v1 page 검색도 함께 사용할 때만 실행합니다. `# 제목`으로 바로 시작하는 일반 Markdown을 사용하는 대부분의 v2 vault에는 필요하지 않습니다.
