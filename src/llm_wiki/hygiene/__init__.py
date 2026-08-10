"""Vault hygiene passes: claim linting, contradiction and compaction candidates.

These tools never rewrite pages. Each one surfaces *candidates* for a human (or
an LLM session following the vault's own review process) to judge — the
judgment itself stays out of scope here.
"""
from __future__ import annotations
