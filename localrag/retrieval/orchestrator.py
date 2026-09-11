"""High-level RAG orchestrator coordinating hybrid search, RRF, and LLM generation."""

from typing import Callable, Iterator, List, Optional
from pydantic import BaseModel, Field

from localrag.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from localrag.retrieval.rrf import reciprocal_rank_fusion
from localrag.retrieval.synthesizer import ContextSynthesizer, SourceCitation, SynthesizedContext
from localrag.storage.sqlite_store import SQLiteStore, SearchResult


SYSTEM_PROMPT = """You are LocalRAG-Kit, an expert offline AI assistant for local codebases and documents.
Answer the user's inquiry thoroughly and accurately using ONLY the provided context sources.

Guidelines:
1. Always cite relevant source numbers like [Source #1], [Source #2] directly in your answer whenever stating facts or code details.
2. If code snippets or commands are relevant, format them with appropriate syntax highlighting markdown.
3. If the answer cannot be determined from the provided context, state clearly: "I cannot find sufficient information in the indexed documents to answer this question." Do not fabricate or hallucinate.
"""


class RAGResponse(BaseModel):
    """Complete answer with provenance citations and retrieval metrics."""
    query: str
    answer: str
    citations: List[SourceCitation]
    retrieved_count: int
    context_tokens: int


class RAGOrchestrator:
    """Coordinates hybrid retrieval, reciprocal rank fusion, and grounded answer synthesis."""

    def __init__(
        self,
        store: SQLiteStore,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_synthesizer: Optional[ContextSynthesizer] = None,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.synthesizer = context_synthesizer or ContextSynthesizer()

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """
        Execute search according to mode ('hybrid', 'bm25', or 'vector').
        In 'hybrid' mode, performs both searches and fuses rankings via RRF.
        """
        if mode == "bm25":
            return self.store.search_bm25(query, limit=top_k)

        if mode == "vector":
            if not self.embedding_provider:
                return []
            query_vec = self.embedding_provider.embed_text(query)
            return self.store.search_vector(query_vec, limit=top_k)

        # Hybrid search: fetch from both and merge with RRF
        bm25_res = self.store.search_bm25(query, limit=top_k * 2)

        vector_res: List[SearchResult] = []
        if self.embedding_provider:
            query_vec = self.embedding_provider.embed_text(query)
            vector_res = self.store.search_vector(query_vec, limit=top_k * 2)

        if not vector_res:
            return bm25_res[:top_k]
        if not bm25_res:
            return vector_res[:top_k]

        return reciprocal_rank_fusion(bm25_res, vector_res, limit=top_k)

    def ask(
        self,
        query: str,
        top_k: int = 6,
        mode: str = "hybrid",
        temperature: float = 0.5,
    ) -> RAGResponse:
        """Retrieve context and generate grounded answer synchronously."""
        if not self.llm_provider:
            raise ValueError("No LLM provider configured on RAGOrchestrator")

        results = self.retrieve(query, top_k=top_k, mode=mode)
        ctx: SynthesizedContext = self.synthesizer.synthesize(results)

        if not ctx.formatted_context:
            return RAGResponse(
                query=query,
                answer="No relevant documents or code chunks were found in the index.",
                citations=[],
                retrieved_count=0,
                context_tokens=0,
            )

        prompt = f"Context from local files:\n\n{ctx.formatted_context}\n\nUser Question:\n{query}\n\nAnswer:"
        answer = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
        )

        return RAGResponse(
            query=query,
            answer=answer,
            citations=ctx.citations,
            retrieved_count=len(results),
            context_tokens=ctx.total_tokens,
        )

    def stream_ask(
        self,
        query: str,
        top_k: int = 6,
        mode: str = "hybrid",
        temperature: float = 0.5,
        citations_callback: Optional[Callable[[List[SourceCitation]], None]] = None,
    ) -> Iterator[str]:
        """Stream generated answer tokens in real time while notifying citations callback."""
        if not self.llm_provider:
            yield "[Error: No LLM provider configured]"
            return

        results = self.retrieve(query, top_k=top_k, mode=mode)
        ctx = self.synthesizer.synthesize(results)

        if citations_callback:
            citations_callback(ctx.citations)

        if not ctx.formatted_context:
            yield "No relevant documents or code chunks were found in the index."
            return

        prompt = f"Context from local files:\n\n{ctx.formatted_context}\n\nUser Question:\n{query}\n\nAnswer:"
        for token in self.llm_provider.stream_generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
        ):
            yield token
