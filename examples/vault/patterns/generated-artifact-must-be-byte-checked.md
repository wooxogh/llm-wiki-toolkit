---
id: generated-artifact-must-be-byte-checked
layer: pattern
projects: ["llm-wiki"]
tags: ["ci", "generated-artifact", "drift"]
confidence: confirmed
status: active
updated: 2026-01-14
summary: 'A committed generated artifact that is syntactically valid but stale still passes a naive check; the only check that catches drift compares rendered bytes against what is committed.'
links: ["embedding-store-layout"]
---

# A generated file that only "looks" valid still poisons the system

A repository that commits a generated artifact — an index, a lockfile, a
rendered report — usually wants two different checks, and it is easy to
write only the weaker one by mistake.

The weak check asks: is this file well-formed and internally consistent? Do
its entries reference things that exist, do its types match its schema, are
its cross-references resolvable? A file can pass every one of those
questions and still be *wrong*, in one specific way: it can be a perfectly
valid rendering of the source data as it existed at some earlier commit,
sitting alongside source data that has since moved on. Nothing about the
file itself signals that it is stale — staleness is not a property of the
file in isolation, it is a property of the *relationship* between the file
and its source.

The strong check asks a different question: if I regenerate this artifact
right now from the current source, do I get byte-for-byte what is already
committed? That check does not care whether the committed file is
individually well-formed; it cares whether it is still an accurate
rendering. It is strictly more work — it requires the ability to
deterministically re-render the artifact, which usually means sorting
outputs, pinning float precision, and avoiding anything nondeterministic in
the render step — but it is the only one of the two that actually catches
drift, because drift by definition still produces something syntactically
fine.

A syntactically-fine-but-stale generated artifact is worse than an obviously
broken one, precisely because nothing downstream complains about it. A
retrieval index built from three-versions-ago frontmatter will confidently
serve wrong metadata for every page that changed since, at full speed, with
no error anywhere in the chain — the exact failure shape the weak check is
structurally unable to see. Any pipeline that treats "generate once, commit,
trust forever" as safe is really relying on nobody ever changing the source
without remembering to regenerate, which is not an assumption a CI check
should ever make on a team's behalf.

The same reasoning applies to more than one kind of generated file in a
knowledge vault — see `embedding-store-layout` for the same drift class
showing up as four separate, independently stale artifacts rather than one.
