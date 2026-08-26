# PayBridge Synthetic Vault

35 AI-generated Korean markdown documents from a fictional fintech startup
("PayBridge"), used as a supplementary benchmark corpus alongside the
English ADR corpora (Cosmos SDK, EdgeX Foundry).

## Why this exists

The English ADR corpora used elsewhere in this project's benchmarks
(`benchmarks/cosmos-sdk-adr/`, `benchmarks/edgex-foundry-adr/`) are entirely
"software architecture decision" documents in one language. Evaluating
only on that genre risks overfitting the retrieval/hygiene pipeline to a
single document style — the same "eval_gold_v2.json only covers one
company's domain" problem noted as a limitation of the v1 gold set.

This vault specifically targets two scenarios the real ADR corpora do not
cover:

- **Collection (repeated document series)**: `monthly_reports/` — 10
  monthly business reports (2025-10 ~ 2026-07) with evolving revenue,
  active-merchant, and churn figures, cross-referencing the retry-policy
  decisions below.
- **DISPUTED (genuine unresolved conflict)**: `meetings/2026-05-04-인사팀-공지.md`
  vs `meetings/2026-05-06-플랫폼팀-스크럼.md` — two documents issued two
  days apart, stating contradictory remote-work policies (주 2회 vs 주
  3회), where scope/time alone cannot resolve which is authoritative.

## Contents

| Folder | Count | Scenario |
|---|---|---|
| `meetings/` | 17 | 15 routine weekly meeting notes + 1 intentional DISPUTED pair |
| `project_docs/` | 8 | 3 SUPERSEDES chains (payment retry policy: 0002→0004→0009; DB choice: 0003→0007) |
| `monthly_reports/` | 10 | Collection-style recurring report series |

## ⚠️ Disclosure

**This is 100% AI-generated synthetic data.** The company, people, and
figures are entirely fictional and do not reflect real facts. This
distinction must be preserved wherever this dataset is referenced (demo
scripts, evaluation reports, competition submission materials) — it should
never be presented with the same evidentiary weight as the real ADR
corpora.

## Known data-quality note

`meetings/2026-02-10-결제팀-스크럼.md` states the retry-policy change is
"반영 완료" one day before `project_docs/0004`'s dated approval
(2026-02-11). This does not affect the SUPERSEDES chain ordering (dates
0002 < 0004 < 0009 remain correct) and was left as-is because minor
same-week date noise is realistic for organizational documents; flag if
it causes eval issues.
