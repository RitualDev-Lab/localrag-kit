"""Context packing, source attribution, and token budget management for RAG prompts."""

from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from localrag.core.chunkers.base import BaseChunker
from localrag.storage.sqlite_store import SearchResult


class SourceCitation(BaseModel):
    """Citation provenance for a retrieved chunk."""
    source_index: int = Field(description="1-based citation index (e.g. 1 for [Source #1])")
    relative_path: str = Field(description="Relative path of file")
    start_line: int = Field(description="Starting line in source file")
    end_line: int = Field(description="Ending line in source file")
    section_title: Optional[str] = Field(default=None, description="Section or symbol name")
    score: float = Field(description="Relevance or RRF fusion score")
    text_snippet: str = Field(description="Quoted text content")

    @property
    def label(self) -> str:
        sec = f" | {self.section_title}" if self.section_title else ""
        return f"[Source #{self.source_index}: {self.relative_path}:{self.start_line}-{self.end_line}{sec}]"


class SynthesizedContext(BaseModel):
    """Packed context ready for LLM prompt injection."""
    formatted_context: str
    citations: List[SourceCitation]
    total_tokens: int


class ContextSynthesizer:
    """Deduplicates and packs search results into a clean prompt context block within token limits."""

    def __init__(self, max_token_budget: int = 3500):
        self.max_token_budget = max_token_budget

    def synthesize(self, results: List[SearchResult]) -> SynthesizedContext:
        """Filter, deduplicate, and assemble top results into numbered context blocks."""
        if not results:
            return SynthesizedContext(formatted_context="", citations=[], total_tokens=0)

        citations: List[SourceCitation] = []
        blocks: List[str] = []
        seen_line_ranges: Set[Tuple[str, int, int]] = set()
        accumulated_tokens = 0

        source_idx = 1
        for res in results:
            chunk = res.chunk
            meta = chunk.metadata

            # Avoid exact duplicate file/line overlaps
            line_key = (meta.relative_path, meta.start_line, meta.end_line)
            if line_key in seen_line_ranges:
                continue
            seen_line_ranges.add(line_key)

            chunk_tokens = meta.estimated_tokens or BaseChunker.estimate_tokens(chunk.text)
            if accumulated_tokens + chunk_tokens > self.max_token_budget and citations:
                # Token budget reached, skip remaining lower-rank chunks
                break

            citation = SourceCitation(
                source_index=source_idx,
                relative_path=meta.relative_path,
                start_line=meta.start_line,
                end_line=meta.end_line,
                section_title=meta.section_title,
                score=res.score,
                text_snippet=chunk.text.strip(),
            )
            citations.append(citation)

            # Build readable source header and code block
            sec_header = f" | Section: {citation.section_title}" if citation.section_title else ""
            block = (
                f"--- [Source #{citation.source_index}] File: {citation.relative_path} "
                f"(Lines {citation.start_line}-{citation.end_line}{sec_header}) ---\n"
                f"{citation.text_snippet}\n"
            )
            blocks.append(block)

            accumulated_tokens += chunk_tokens
            source_idx += 1

        formatted_text = "\n".join(blocks).strip()
        return SynthesizedContext(
            formatted_context=formatted_text,
            citations=citations,
            total_tokens=accumulated_tokens,
        )
