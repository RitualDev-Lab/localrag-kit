"""Retrieval, ranking, and context synthesis layer for LocalRAG-Kit."""

from localrag.retrieval.orchestrator import (
    SYSTEM_PROMPT,
    RAGOrchestrator,
    RAGResponse,
)
from localrag.retrieval.rrf import reciprocal_rank_fusion
from localrag.retrieval.synthesizer import (
    ContextSynthesizer,
    SourceCitation,
    SynthesizedContext,
)

__all__ = [
    "SYSTEM_PROMPT",
    "ContextSynthesizer",
    "RAGOrchestrator",
    "RAGResponse",
    "SourceCitation",
    "SynthesizedContext",
    "reciprocal_rank_fusion",
]
