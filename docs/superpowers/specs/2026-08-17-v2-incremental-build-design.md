# v2 Incremental Build Design

## Goal

Make repeated v2 builds proportional to changed documents and affected
Concepts, while preserving the current provenance, artifact identity, and
fail-closed guarantees.

## Scope

This change covers Concept extraction, Concept vector indexing, and NET
relation reconciliation. Temporal-aware candidate partitioning is reserved for
a later change; this implementation must keep relation results conservative
without assuming that recency alone proves supersession.

## Design

1. **Document and chunk reuse**
   - Collect live document hashes first.
   - With `changed_only=True`, read stored documents/chunks and rebuild chunks
     only for added or changed documents.
   - Reuse unchanged chunks and Concepts byte-for-byte where possible.
   - Remove artifacts belonging to deleted documents.

2. **Vault-scoped extraction cache**
   - Pass the target vault through Concept extraction to the cache layer.
   - Cache keys continue to include source hash, prompt version, and model
     identity.
   - A cache from one vault must never be read or written for another vault.

3. **Incremental Concept index**
   - Preserve vectors and metadata for unchanged Concept IDs.
   - Embed only new or changed Concept records.
   - Remove deleted Concept records and write vectors/meta in Concept order.
   - Force a full rebuild when model identity, vector dimension, index schema,
     or indexed text schema changes.

4. **Affected relation reconciliation**
   - Preserve approved and rejected terminal proposals and user-owned NET
     structure.
   - Mark Concepts whose source document changed, plus added and deleted
     Concepts, as dirty.
   - Recompute candidates and classifications for dirty Concepts against the
     existing Concept candidate index; remove only open proposals and edges
     whose affected endpoints are dirty or deleted.
   - Keep approved relations unless an explicit source deletion invalidates an
     endpoint; never silently rewrite a human decision.

5. **Health and safety**
   - Incremental artifacts must pass the same provenance, hash, identity,
     lifecycle, and NET integrity checks as full builds.
   - Any incompatible identity or missing artifact must fail closed and require
     a full rebuild rather than mixing generations.

## Non-goals

- Changing the semantic embedding model.
- Inferring temporal validity from file timestamps.
- Automatically approving risky relations.
- Removing the existing full rebuild path.

## Acceptance criteria

- Adding one document does not call the Concept adapter for unchanged
  documents.
- Adding one document does not re-embed unchanged Concepts.
- Deleting a document removes its Concepts and vectors.
- A model/schema identity change forces a full index rebuild.
- Existing approved relations survive an incremental build.
- Changed source provenance or stale artifacts prevent normal query use.
- Full rebuild and incremental rebuild produce equivalent canonical artifacts
  for the same final vault, apart from explicitly documented ordering or
  generated timestamps.
