"""Retrieval evaluation: gold-file schema, ranking metrics, and the regression gate.

Kept dependency-light on purpose (stdlib + math, plus pyyaml for the index):
the whole scoring layer is unit-testable without embedding anything, so a
metric regression can be localised to *ranking* rather than to the model or
the store.
"""
from __future__ import annotations
