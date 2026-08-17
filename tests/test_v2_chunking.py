from llm_wiki.v2.chunking import chunk_document


def test_h1_block_under_target_is_one_chunk():
    raw = "# Decision\n\n" + "a" * 480
    chunks = chunk_document("doc", "domain/doc.md", raw, target_chars=700)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Decision"]


def test_large_h1_splits_into_h2_chunks_without_parent_heading_chunk():
    raw = "# Parent\n\n## First\n\n" + "a" * 430 + "\n\n## Second\n\n" + "b" * 430
    chunks = chunk_document("doc", "domain/doc.md", raw, target_chars=700)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Parent", "First"]
    assert chunks[1].heading_path == ["Parent", "Second"]


def test_no_heading_large_block_splits_by_paragraph():
    raw = "a" * 450 + "\n\n" + "b" * 450
    chunks = chunk_document("doc", "domain/doc.md", raw, target_chars=700)
    assert len(chunks) == 2


def test_single_large_paragraph_stays_oversized():
    raw = "a" * 1200
    chunks = chunk_document("doc", "domain/doc.md", raw, target_chars=700)
    assert len(chunks) == 1
    assert chunks[0].oversized is True
