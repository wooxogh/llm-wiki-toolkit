"""Read-only hygiene reports over the vault's [[wikilink]] graph.

`graph_report` computes the graph itself (nodes, edges, degrees, communities,
dangling links). `community_report` builds on top of it to add a GraphRAG-style
extractive/abstractive summary per community. Neither module rewrites pages.
"""
from __future__ import annotations
