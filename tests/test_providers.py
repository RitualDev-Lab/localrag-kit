"""Tests for embedding providers, LLM providers, and vector-backed indexing."""

import tempfile
from pathlib import Path
from localrag.core.ingestion import IngestionPipeline
from localrag.providers import (
    FastFeatureEmbeddingProvider,
    MockLLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from localrag.storage.sqlite_store import SQLiteStore
from localrag.storage.vector_ops import cosine_similarity


def test_fast_feature_embedding_provider():
    provider = FastFeatureEmbeddingProvider(dimension=384)
    assert provider.dimension == 384
    assert provider.name == "fast-local"
    assert provider.is_available() is True

    # Empty text
    empty_vec = provider.embed_text("")
    assert len(empty_vec) == 384
    assert all(x == 0.0 for x in empty_vec)

    # Unit norm
    text_a = "High performance database index optimization using B-trees and hash maps."
    vec_a = provider.embed_text(text_a)
    assert len(vec_a) == 384
    norm_a = sum(x * x for x in vec_a)
    assert abs(norm_a - 1.0) < 1e-4

    # Similarity test
    text_b = "Database query indexing and performance tuning."
    text_unrelated = "French cooking recipes with butter and garlic."

    vec_b = provider.embed_text(text_b)
    vec_unrelated = provider.embed_text(text_unrelated)

    sim_related = cosine_similarity(vec_a, vec_b)
    sim_unrelated = cosine_similarity(vec_a, vec_unrelated)

    assert sim_related > sim_unrelated
    assert sim_related > 0.12
    assert sim_unrelated < 0.05


def test_mock_llm_provider():
    llm = MockLLMProvider()
    assert llm.is_available() is True

    # Test sync generation
    full_resp = llm.generate("How do I index files?")
    assert "Based on the provided context" in full_resp

    # Test streaming generation
    tokens = list(llm.stream_generate("Tell me about local RAG"))
    assert len(tokens) > 3
    assert "".join(tokens) == full_resp


def test_provider_factories():
    embed_fast = get_embedding_provider("fast")
    assert isinstance(embed_fast, FastFeatureEmbeddingProvider)

    embed_hash = get_embedding_provider("hash")
    assert isinstance(embed_hash, FastFeatureEmbeddingProvider)

    llm_mock = get_llm_provider("mock")
    assert isinstance(llm_mock, MockLLMProvider)


def test_pipeline_indexing_with_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / ".localrag" / "index.db"

        (root / "auth.py").write_text("""def verify_jwt_token(token):
    # Verify cryptographic signature of JWT
    return True
""")
        (root / "payments.py").write_text("""def process_credit_card(card_number):
    # Charge Stripe or PayPal gateway
    return {"status": "paid"}
""")

        pipeline = IngestionPipeline(root)
        embed_provider = FastFeatureEmbeddingProvider()

        with SQLiteStore(db_path) as store:
            stats = pipeline.index_to_store(
                store,
                incremental=True,
                embedding_provider=embed_provider,
            )
            assert stats.chunks_indexed == 2

            store_stats = store.get_stats()
            assert store_stats.total_vectorized_chunks == 2

            # Perform Vector Search for authentication
            query_vec = embed_provider.embed_text("JWT cryptographic tokens authentication")
            results = store.search_vector(query_vec, limit=2)

            assert len(results) == 2
            # Auth chunk must be top rank
            assert results[0].chunk.metadata.relative_path == "auth.py"
            assert results[0].score > results[1].score
            assert results[0].match_type == "vector"
