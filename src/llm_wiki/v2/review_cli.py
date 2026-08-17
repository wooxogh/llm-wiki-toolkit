"""CLI for v2 review queue."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_wiki.v2.concept_store import read_concepts
from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.review import approve_review_item, reject_review_item, resolve_review_item


def main() -> int:
    ap = argparse.ArgumentParser(prog="wiki-review")
    ap.add_argument("--vault", type=Path)
    ap.add_argument("--approve", help="review item id to approve")
    ap.add_argument("--reject", help="review item id to reject")
    ap.add_argument("--resolve", help="review item id to resolve with an explicit decision")
    ap.add_argument("--decision", choices=("source-current", "target-current", "different-scope",
                                            "disputed", "unrelated"))
    ap.add_argument("--actor", default="user")
    args = ap.parse_args()
    store = NetStore(args.vault)
    if args.approve:
        approve_review_item(store, args.approve, actor=args.actor, vault=args.vault)
        print(f"approved {args.approve}")
        return 0
    if args.reject:
        reject_review_item(store, args.reject, actor=args.actor)
        print(f"rejected {args.reject}")
        return 0
    if args.resolve:
        if not args.decision:
            ap.error("--resolve requires --decision")
        resolve_review_item(store, args.resolve, args.decision, actor=args.actor, vault=args.vault)
        print(f"resolved {args.resolve}: {args.decision}")
        return 0
    items = store.review_items()
    proposals = {proposal.id: proposal for proposal in store.proposals()}
    concepts = {concept.id: concept for concept in read_concepts(args.vault)}
    for item in items:
        print(f"{item.id} [{item.state}] {item.reason} -> {item.proposal_id}")
        proposal = proposals.get(item.proposal_id)
        if not proposal:
            continue
        source = concepts.get(proposal.source_concept_id)
        target = concepts.get(proposal.target_concept_id)
        print(f"  relation={proposal.relation} confidence={proposal.confidence:.3f} "
              f"same_subject={proposal.same_subject} same_scope={proposal.same_scope} "
              f"temporal_change_possible={proposal.temporal_change_possible}")
        print(f"  reason: {proposal.reason or '(none)'}")
        print(f"  evidence: {proposal.evidence}")
        if source:
            print(f"  source: {source.text}\n    quote: {source.source_quote}")
        if target:
            print(f"  target: {target.text}\n    quote: {target.source_quote}")
    if not items:
        print("review queue is empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
