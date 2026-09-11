"""Structure-aware Markdown chunker respecting heading hierarchies and code fences."""

import re

from localrag.core.chunkers.base import BaseChunker
from localrag.core.chunkers.sliding_window import SlidingWindowChunker
from localrag.core.models import Chunk, ChunkMetadata, Document
from localrag.utils.hasher import generate_chunk_id


class MarkdownChunker(BaseChunker):
    """Chunks markdown documents by heading sections while keeping code blocks intact."""

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        min_chunk_tokens: int = 30,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.fallback_chunker = SlidingWindowChunker(
            chunk_size=max_chunk_tokens,
            chunk_overlap=50,
            min_chunk_size=min_chunk_tokens,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        content = document.content
        if not content or not content.strip():
            return []

        lines = content.splitlines()
        total_lines = len(lines)

        # Track heading hierarchy and code block fences
        sections: list[tuple[str, int, int, str]] = []  # (heading_path, start_line, end_line, text)
        current_heading_stack: list[tuple[int, str]] = []  # [(level, title)]
        current_lines: list[str] = []
        current_start_line = 1
        in_code_fence = False

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Check for code fence toggle
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence

            match = heading_pattern.match(stripped) if not in_code_fence else None

            if match:
                # Flush previous section
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        heading_path = " > ".join(title for _, title in current_heading_stack)
                        sections.append((heading_path, current_start_line, idx - 1, sec_text))
                    current_lines = []

                # Update heading stack
                level = len(match.group(1))
                title = match.group(2).strip()

                # Pop higher or equal headings
                while current_heading_stack and current_heading_stack[-1][0] >= level:
                    current_heading_stack.pop()

                current_heading_stack.append((level, title))
                current_start_line = idx

            current_lines.append(line)

        # Flush final section
        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                heading_path = " > ".join(title for _, title in current_heading_stack)
                sections.append((heading_path, current_start_line, total_lines, sec_text))

        # Convert sections into Chunks (sub-chunking if any section exceeds max tokens)
        chunks: list[Chunk] = []

        for heading_path, start_line, end_line, sec_text in sections:
            tokens = self.estimate_tokens(sec_text)
            if tokens <= self.max_chunk_tokens:
                chunk_id = generate_chunk_id(
                    document.metadata.relative_path,
                    start_line,
                    end_line,
                    sec_text,
                )
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    file_path=document.metadata.file_path,
                    relative_path=document.metadata.relative_path,
                    file_type=document.metadata.file_type,
                    start_line=start_line,
                    end_line=end_line,
                    char_count=len(sec_text),
                    estimated_tokens=tokens,
                    section_title=heading_path or None,
                )
                chunks.append(Chunk(text=sec_text, metadata=meta))
            else:
                # Sub-chunk the oversized section using sliding window
                sub_doc = Document(
                    metadata=document.metadata,
                    content=sec_text,
                )
                sub_chunks = self.fallback_chunker.chunk(sub_doc)
                for sc in sub_chunks:
                    # Offset line numbers relative to parent section
                    sc.metadata.start_line += start_line - 1
                    sc.metadata.end_line += start_line - 1
                    sc.metadata.section_title = heading_path or None
                    sc.metadata.chunk_id = generate_chunk_id(
                        document.metadata.relative_path,
                        sc.metadata.start_line,
                        sc.metadata.end_line,
                        sc.text,
                    )
                    chunks.append(sc)

        # If no headings existed, fall back to sliding window
        if not chunks:
            return self.fallback_chunker.chunk(document)

        return chunks
