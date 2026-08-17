from llm_wiki.v2.chunking import chunk_document
from llm_wiki.v2.concept_store import concepts_from_chunks


def test_two_claims_become_two_concepts_with_provenance():
    chunks = chunk_document(
        "doc",
        "domain/doc.md",
        "# Stack\n\nFrontend uses React. Backend uses Spring.",
    )
    concepts = concepts_from_chunks(chunks)
    assert len(concepts) == 2
    assert all(c.document_id == "doc" for c in concepts)
    assert all(c.chunk_id == chunks[0].id for c in concepts)
    assert all(c.source_quote in chunks[0].text for c in concepts)


def test_no_claim_returns_no_concepts():
    chunks = chunk_document("doc", "domain/doc.md", "# Notes\n\nMaybe later\n\n- idea")
    assert concepts_from_chunks(chunks) == []


def test_unsupported_claim_is_dropped():
    class BadAdapter:
        def extract_concepts(self, chunk):
            from llm_wiki.v2.models import ConceptProposal
            return [ConceptProposal("Hallucinated claim", "bad", "not in source", 0.99)]

    chunks = chunk_document("doc", "domain/doc.md", "# Source\n\nBackend uses Spring.")
    assert concepts_from_chunks(chunks, BadAdapter()) == []


def test_concept_progress_reports_completed_chunks():
    chunks = chunk_document(
        "doc", "domain/doc.md", "# Stack\n\nFrontend uses React.\n\n## API\n\nBackend uses Spring."
    )
    events = []

    concepts_from_chunks(chunks, progress=lambda done, total, chunk: events.append(
        (done, total, chunk.id)))

    assert [event[0] for event in events] == list(range(1, len(chunks) + 1))
    assert all(event[1] == len(chunks) for event in events)
    assert events[-1][0] == events[-1][1]
