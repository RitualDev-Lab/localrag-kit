"""Retrieval, ranking, and context synthesis layer for LocalRAG-Kit."""

from localrag.retrieval.rrf import reciprocal_rank_fusion
from localrag.retrieval.synthesizer import (
    ContextSynthesizer,
    SourceCitation,
    SynthesizedContext,
)
from localrag.retrieval.orchestrator import (
    RAGOrchestrator,
    RAGResponse,
    SYSTEM_PROMPT,
)

__all__ = [
    "reciprocal_rank_fusion",
    "ContextSynthesizer",
    "SourceCitation",
    "SynthesizedContext",
    "RAGOrchestrator",
    "RAGResponse",
    "SYSTEM_PROMPT",
]
