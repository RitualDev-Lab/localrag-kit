"""Tests for Reciprocal Rank Fusion, Context Synthesis, and Grounded RAG Orchestration."""

import tempfile
from pathlib import Path
from localrag.core.models import Chunk, ChunkMetadata, FileMetadata, FileType
from localrag.providers import FastFeatureEmbeddingProvider, MockLLMProvider
from localrag.retrieval.orchestrator import RAGOrchestrator
from localrag.retrieval.rrf import reciprocal_rank_fusion
from localrag.retrieval.synthesizer import ContextSynthesizer
from localrag.storage.sqlite_store import SQLiteStore, SearchResult


def _make_dummy_chunk(chunk_id: str, text: str, rel_path: str = "main.py", start_line: int = 1, end_line: int = 10) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            file_path=f"/dummy/{rel_path}",
            relative_path=rel_path,
            file_type=FileType.CODE,
            start_line=start_line,
            end_line=end_line,
            char_count=len(text),
            estimated_tokens=len(text.split()),
            section_title="test_section",
        ),
    )


def test_reciprocal_rank_fusion():
    chunk_a = _make_dummy_chunk("chk_a", "Common chunk in both lists")
    chunk_b = _make_dummy_chunk("chk_b", "Only in BM25")
    chunk_c = _make_dummy_chunk("chk_c", "Only in Vector")

    bm25_res = [
        SearchResult(chunk=chunk_a, score=0.9, match_type="bm25"),
        SearchResult(chunk=chunk_b, score=0.8, match_type="bm25"),
    ]
    vec_res = [
        SearchResult(chunk=chunk_c, score=0.95, match_type="vector"),
        SearchResult(chunk=chunk_a, score=0.85, match_type="vector"),
    ]

    fused = reciprocal_rank_fusion(bm25_res, vec_res, k_rrf=60, limit=3)
    assert len(fused) == 3

    # chunk_a is in both rankings (rank 1 in bm25, rank 2 in vector), so its RRF score is highest!
    assert fused[0].chunk.id == "chk_a"
    assert "hybrid(bm25+vector)" in fused[0].match_type
    assert fused[0].score > fused[1].score


def test_context_synthesizer_dedup_and_budget():
    synthesizer = ContextSynthesizer(max_token_budget=50)

    chunk1 = _make_dummy_chunk("c1", "First chunk with some words here.", start_line=1, end_line=10)
    chunk1_duplicate = _make_dummy_chunk("c1_dup", "Duplicate of first chunk.", start_line=1, end_line=10)
    chunk2 = _make_dummy_chunk("c2", "Second distinct chunk with other words.", start_line=11, end_line=20)
    huge_chunk = _make_dummy_chunk("c_huge", " ".join(["overflow"] * 100), start_line=21, end_line=120)

    results = [
        SearchResult(chunk=chunk1, score=1.0, match_type="hybrid"),
        SearchResult(chunk=chunk1_duplicate, score=0.9, match_type="hybrid"),  # Same line range -> should deduplicate
        SearchResult(chunk=chunk2, score=0.8, match_type="hybrid"),
        SearchResult(chunk=huge_chunk, score=0.7, match_type="hybrid"),        # Exceeds token budget -> should skip
    ]

    ctx = synthesizer.synthesize(results)

    # Must contain chunk1 and chunk2, but not duplicate or huge overflow
    assert len(ctx.citations) == 2
    assert ctx.citations[0].source_index == 1
    assert ctx.citations[1].source_index == 2
    assert "[Source #1]" in ctx.formatted_context
    assert "[Source #2]" in ctx.formatted_context
    assert ctx.total_tokens < 50


def test_rag_orchestrator_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        with SQLiteStore(db_path) as store:
            embed_provider = FastFeatureEmbeddingProvider()
            llm_provider = MockLLMProvider()

            # Seed parent file
            file_meta = FileMetadata(
                file_path="/dummy/auth.py",
                relative_path="auth.py",
                file_type=FileType.CODE,
                sha256="hash_auth",
            )
            store.upsert_file(file_meta)

            # Seed store with chunks
            chunk_jwt = Chunk(
                text="def verify_jwt(token):\n    '''Verify cryptographic JSON web tokens.'''\n    return decode(token)",
                metadata=ChunkMetadata(
                    chunk_id="chk_jwt",
                    file_path="/dummy/auth.py",
                    relative_path="auth.py",
                    file_type=FileType.CODE,
                    start_line=1,
                    end_line=5,
                    section_title="verify_jwt",
                ),
                embedding=embed_provider.embed_text("verify cryptographic JSON web tokens auth"),
            )
            store.upsert_chunks([chunk_jwt])

            orchestrator = RAGOrchestrator(
                store=store,
                embedding_provider=embed_provider,
                llm_provider=llm_provider,
            )

            # Test synchronous ask()
            response = orchestrator.ask("How does JWT verification work?", top_k=3, mode="hybrid")
            assert response.retrieved_count >= 1
            assert len(response.citations) >= 1
            assert response.citations[0].relative_path == "auth.py"
            assert "Based on the provided context" in response.answer

            # Test streaming stream_ask()
            received_citations = []
            tokens = list(orchestrator.stream_ask(
                query="How does JWT verification work?",
                top_k=3,
                mode="hybrid",
                citations_callback=lambda c: received_citations.extend(c),
            ))
            assert len(received_citations) >= 1
            assert len(tokens) > 2
            assert "".join(tokens) == response.answer
