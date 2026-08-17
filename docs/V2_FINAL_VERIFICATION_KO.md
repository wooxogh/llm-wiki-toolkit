# llm-wiki v2 최종 구현 검증

검증 기준은 `llm-wiki_v2_Codex_Architecture_Migration_Spec_detailed.pdf` 31페이지 전체와 최신 `v2_migration_verification_report.md`다. 검증일은 2026-08-17이다.

## 결론

명세의 architecture migration 범위가 구현되었다. Markdown은 계속 source of truth이고, v2는 `.llm_wiki_v2/` 아래에 Document, Chunk, Atomic Concept, Concept index, NET, proposal/review/audit artifact를 분리한다. LLM은 versioned structured proposal만 만들며 deterministic persistence가 schema, tree, approval, temporal invariant를 검사한다.

## 명세 대조

| 명세 영역 | 판정 | 구현 근거 |
|---|---|---|
| Markdown source of truth / 기존 artifact 병행 | 적용 | 원문 수정 없이 `.llm_wiki_v2/`에 파생 artifact 저장 |
| `# -> ## -> paragraph`, target 700, oversized paragraph 유지 | 적용 | `v2/chunking.py`, chunk schema/target identity, acceptance tests |
| Document -> Chunk -> Concept provenance | 적용 | source document/chunk/hash/span/exact quote 검증 |
| Atomic Concept 0..N / prompt version / grounding | 적용 | extraction v2 instruction+schema, exact quote와 lexical grounding guard |
| User LLM provider-neutral adapter | 적용 | `[v2] agent = "codex"|"claude"` 자동 CLI 연결과 task별 versioned instruction+JSON schema |
| Topic/Collection/Document/Concept NET, Chunk 제외 | 적용 | typed node/edge validation, Chunk는 provenance artifact에만 존재 |
| Tree invariant | 적용 | Topic만 parent, one parent, cycle 방지, active orphan 탐지 |
| Document tree location 정확히 1개 | 적용 | persistence cardinality 검증과 health 검사 |
| Concept primary 1개 / secondary 0..N | 적용 | persistence cardinality, 중복/primary-secondary 충돌 방지, artifact 동기화 |
| AI Topic/Collection 생성 및 rich placement context | 적용 | label/parent/state/collection/current document placement를 adapter에 전달 |
| 사용자 구조 작업과 undo | 적용 | move/rename/merge/delete/restore/document/primary/secondary/collection 전체 snapshot undo |
| append-only operation audit | 적용 | 원 operation을 삭제하지 않고 UNDO target을 별도 기록 |
| safe/risky relation 정책 | 적용 | safe threshold path, risky 3종은 confidence와 무관하게 review 필수 |
| SUPERSEDES 방향/조건/lifecycle | 적용 | same subject/scope, incompatible revision signal, newer metadata 또는 explicit evidence, 승인 검사 |
| committed/proposal/review 분리 | 적용 | edges, proposals, review queue가 별도이며 검색은 committed edge만 사용 |
| Concept candidate discovery | 적용 | Concept Qwen/hash vector + BM25, Dense 2/Sparse 1 weighted RRF K=60, top-N만 LLM 전달 |
| Query tree+dense+text+graph+lifecycle | 적용 | tree soft routing, hybrid seeds, one-hop typed graph expansion, current/historical 정책 |
| optional cross-encoder / auto fail-closed | 적용 | `--rerank`, auto의 reranker 강제, absolute cosine/margin/rerank threshold, dispute/hash fallback review |
| answer/review/none | 적용 | v2 `--auto --json`, uncalibrated/stale/reranker failure에서 answer 금지 |
| schema/prompt/model/config invalidation | 적용 | manifest를 자동 덮어쓰지 않음, health 비교, changed build의 자동 full rebuild |
| strict artifact parsing | 적용 | unknown model field를 조용히 버리지 않고 오류 처리 |
| health 확장 | 적용 | source drift, provenance, quote, index, model/schema/config, tree, lifecycle, approval trace, undo 검사 |
| eval/gate 확장 | 적용 | extraction, placement, per-relation, supersession, current/history, safety, review, decay metrics |
| ingest orchestration | 적용 | `[v2] enabled` 시 concepts -> net을 기존 health/commit 전에 실행, Windows shell 호환 |
| CLI/report/docs | 적용 | concepts/net/review/recall/eval/gate/export 및 한국어 사용 문서 갱신 |
| backward compatibility | 적용 | legacy retrieval/eval/health 경로 유지, 전체 기존 테스트 통과 |

## 최신 피드백 해소

- Concept prompt를 `concept-extraction.v2`로 올리고 old cache/artifact를 무효화했다.
- `CREATE_COLLECTION`, `MERGE_TOPIC`, secondary 제거를 포함한 모든 user operation을 정확히 undo한다.
- `merge-topic`, `remove-secondary-topic` CLI를 추가했다.
- relation/temporal LLM instruction과 strict schema, persistence temporal guard, 상세 review UX를 추가했다.
- PlacementProposal과 NET build에 Collection 생성/선택/문서 배치를 연결했다.
- schema manifest silent overwrite를 제거하고 build identity와 health 비교를 추가했다.
- 문서별 `CONTAINS_DOCUMENT` 정확히 1개를 persistence와 health에서 강제한다.
- v2 answer/review/none, true BM25+weighted RRF, 실제 저장 vector 재사용, optional cross-encoder를 연결했다.
- v2 eval metric과 gate의 `None` arithmetic 문제를 제거했다.
- UTF-8 BOM config, Windows ingest `.sh`, Claude/Codex memory 경로 회귀를 해결했다.

## 실행 검증

- 전체 pytest 최종 결과: `377 passed, 1 skipped` (`58.43s`). Codex/Claude agent 자동 선택과 structured-output 호출 계약 테스트를 포함한다.
- 별도 smoke vault: 2 Documents -> 2 Chunks -> 4 Concepts -> Concept index -> 9 NET nodes / 12 edges.
- v2 recall: `The retry count is 2.`를 1위로 반환했다.
- v2 auto: 기본 hash backend에서 `review / uncalibrated-concept-embedding`으로 fail closed했다.
- v2 health: 이슈 0개.

## 외부 제한

별도 smoke vault의 legacy `.embeddings`까지 새로 만들기 위한 Hugging Face 모델 다운로드는 비인증 Hub 요청 단계에서 실패했다. 따라서 그 임시 vault의 `wiki-health --mode full`은 legacy `embedding-store-missing` 한 건을 보고했다. 이는 v2 오류가 아니며, v2 health는 별도로 0건을 확인했다. 프로젝트 전체 테스트와 기존 embedding 코드 회귀는 모두 통과해야 최종 완료로 판정한다.
