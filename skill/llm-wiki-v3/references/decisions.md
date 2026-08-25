# Hygiene Decision Contracts

All decisions require `user_approved: true`, `reason`, and the reviewed old
chunk's `expected_content_hash`. Preserve the user's wording in the reason.

## Partial supersede using an existing chunk

```json
{
  "type": "partial_supersede",
  "user_approved": true,
  "old_chunk_id": "chunk:old",
  "superseding_chunk_id": "chunk:new",
  "expected_content_hash": "sha256",
  "claim_id": "claim:stable-name",
  "quote": "Exact obsolete text from the old chunk",
  "replacement_quote": "Exact current text from the successor",
  "reason": "User-confirmed explanation"
}
```

## Partial supersede using a new resolution chunk

Use a self-contained `replacement_text`. The old chunk remains searchable and
the generated resolution chunk is attached whenever it is returned.

```json
{
  "type": "partial_supersede",
  "user_approved": true,
  "old_chunk_id": "chunk:old",
  "expected_content_hash": "sha256",
  "claim_id": "claim:stable-name",
  "quote": "Exact obsolete text from the old chunk",
  "replacement_quote": "The current claim",
  "replacement_text": "Enough context and the current claim to stand alone.",
  "reason": "User-confirmed explanation"
}
```

## Factual error replacement

`corrected_text` must retain useful context from the old chunk while replacing
the error. The old chunk becomes `retracted` and is excluded from normal search.

```json
{
  "type": "error_correction",
  "user_approved": true,
  "old_chunk_id": "chunk:old",
  "expected_content_hash": "sha256",
  "quote": "Exact incorrect text",
  "corrected_text": "A self-contained corrected replacement chunk.",
  "reason": "User-provided correction"
}
```

## Dispute

```json
{
  "type": "dispute",
  "user_approved": true,
  "chunk_ids": ["chunk:a", "chunk:b"],
  "claim_quotes": {
    "chunk:a": "Exact claim A",
    "chunk:b": "Exact claim B"
  },
  "reason": "The user could not establish which claim is current."
}
```

