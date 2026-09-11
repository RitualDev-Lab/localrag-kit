"""Token-aware sliding window chunker with exact line number preservation."""

import re
from typing import List, Optional
from localrag.core.chunkers.base import BaseChunker
from localrag.core.models import Chunk, ChunkMetadata, Document
from localrag.utils.hasher import generate_chunk_id


class SlidingWindowChunker(BaseChunker):
    """Chunks text using a sliding window of paragraphs/sentences with token boundaries."""

    def __init__(
        self,
        chunk_size: int = 400,       # Approximate target tokens per chunk
        chunk_overlap: int = 60,     # Overlap tokens between consecutive chunks
        min_chunk_size: int = 40,    # Ignore smaller residual chunks
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        content = document.content
        if not content or not content.strip():
            return []

        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        # Precompute character line index: map character position -> line number (1-indexed)
        line_start_offsets = [0]
        curr_offset = 0
        for line in lines:
            curr_offset += len(line)
            line_start_offsets.append(curr_offset)

        def offset_to_line(char_pos: int) -> int:
            """Binary search line number for character offset."""
            import bisect
            idx = bisect.bisect_right(line_start_offsets, char_pos)
            return max(1, idx)

        # Split text into paragraphs or logical line chunks
        paragraphs = []
        pattern = re.compile(r"(\n\s*\n+)")
        last_end = 0
        for m in pattern.finditer(content):
            start, end = m.span()
            if start > last_end:
                paragraphs.append((content[last_end:start], last_end, start))
            last_end = end
        if last_end < len(content):
            paragraphs.append((content[last_end:], last_end, len(content)))

        # If paragraphs are too few or long, break down by lines
        units = []
        for p_text, p_start, p_end in paragraphs:
            p_tokens = self.estimate_tokens(p_text)
            if p_tokens > self.chunk_size:
                # Sub-split long paragraph by sentences or lines
                sub_lines = p_text.splitlines(keepends=True)
                sub_offset = p_start
                for s_line in sub_lines:
                    units.append((s_line, sub_offset, sub_offset + len(s_line)))
                    sub_offset += len(s_line)
            else:
                units.append((p_text, p_start, p_end))

        # Build overlapping chunks from units
        chunks: List[Chunk] = []
        curr_units = []
        curr_tokens = 0
        i = 0

        while i < len(units):
            u_text, u_start, u_end = units[i]
            u_tokens = self.estimate_tokens(u_text)

            curr_units.append(units[i])
            curr_tokens += u_tokens

            if curr_tokens >= self.chunk_size or i == len(units) - 1:
                # Assemble chunk text
                chunk_text = "".join(u[0] for u in curr_units).strip()
                if chunk_text and self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                    start_char = curr_units[0][1]
                    end_char = curr_units[-1][2]
                    start_line = offset_to_line(start_char)
                    end_line = offset_to_line(max(start_char, end_char - 1))

                    chunk_id = generate_chunk_id(
                        document.metadata.relative_path,
                        start_line,
                        end_line,
                        chunk_text,
                    )

                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        file_path=document.metadata.file_path,
                        relative_path=document.metadata.relative_path,
                        file_type=document.metadata.file_type,
                        start_line=start_line,
                        end_line=end_line,
                        char_count=len(chunk_text),
                        estimated_tokens=self.estimate_tokens(chunk_text),
                    )
                    chunks.append(Chunk(text=chunk_text, metadata=meta))

                # If at end, terminate
                if i == len(units) - 1:
                    break

                # Backtrack to implement overlap
                overlap_tokens = 0
                backtrack_count = 0
                for j in range(len(curr_units) - 1, -1, -1):
                    overlap_tokens += self.estimate_tokens(curr_units[j][0])
                    backtrack_count += 1
                    if overlap_tokens >= self.chunk_overlap:
                        break

                # Advance window
                step = max(1, len(curr_units) - backtrack_count)
                i = (i - len(curr_units) + 1) + step
                curr_units = []
                curr_tokens = 0
                continue

            i += 1

        # Fallback if no chunk produced (e.g. content was short)
        if not chunks and content.strip():
            chunk_text = content.strip()
            chunk_id = generate_chunk_id(document.metadata.relative_path, 1, len(lines), chunk_text)
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=document.metadata.file_path,
                relative_path=document.metadata.relative_path,
                file_type=document.metadata.file_type,
                start_line=1,
                end_line=len(lines),
                char_count=len(chunk_text),
                estimated_tokens=self.estimate_tokens(chunk_text),
            )
            chunks.append(Chunk(text=chunk_text, metadata=meta))

        return chunks
