---
id: calibrate-and-report-on-different-splits
layer: pattern
projects: ["llm-wiki"]
tags: ["evaluation", "calibration", "overfitting"]
confidence: confirmed
status: active
updated: 2026-01-15
summary: 'Fitting a decision threshold and reporting its accuracy on the same set of cases makes the threshold look safe purely by having memorised those exact cases.'
links: ["unmeasured-claims-are-worse-than-gaps"]
---

# Why calibration and reporting need different cases

Any system that has to decide "answer automatically" vs. "say I don't know"
needs a threshold — some cutoff on a confidence score above which it trusts
its own top result. That threshold has to be *fit* to data: pick it too high
and the system refuses to answer things it actually knows; pick it too low
and it confidently states wrong answers. Fitting it well requires a labelled
set of queries with known-correct answers.

The trap is using the same labelled set to both fit the threshold and then
report how well that threshold performs. A threshold search will happily
find the cutoff that maximizes accuracy on whatever cases you show it — that
is exactly what it is supposed to do. If you then measure accuracy on those
same cases, the number you get back is not a measurement of how the
threshold generalizes; it is closer to a measurement of how well the search
memorised the cases it was given. A threshold that is quietly overfit to its
calibration set can report a very good number while failing on the next real
query that is not exactly like anything it was tuned against — and nothing
in the calibration-and-report-on-the-same-set number would ever reveal that,
because the number was never capable of catching it.

The fix is a split: cases marked for calibration are the only ones a
threshold-fitting routine is allowed to look at, and cases marked for
reporting (a disjoint set) are the only ones a reported accuracy number is
allowed to come from. This is the same basic idea as a train/test split
anywhere else in machine learning, applied to something that looks more like
configuration than modelling — a hand-picked confidence cutoff feels less
like "a model" than a trained classifier does, but it is fit to data in
exactly the same way, and is exactly as capable of overfitting to it.

The failure mode this guards against is specifically insidious because it
produces a number that *looks* rigorous — an evaluation harness, a reported
accuracy, a threshold chosen by search rather than by hand — while still
being wrong in the one way that matters. See
`unmeasured-claims-are-worse-than-gaps` for a related case where the
appearance of rigor (a hedge, phrased carefully) is not the same thing as
the rigor itself.
