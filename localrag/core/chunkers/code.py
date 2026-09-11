"""Structure-aware source code chunker preserving function and class boundaries."""

import re

from localrag.core.chunkers.base import BaseChunker
from localrag.core.chunkers.sliding_window import SlidingWindowChunker
from localrag.core.models import Chunk, ChunkMetadata, Document
from localrag.utils.hasher import generate_chunk_id

# Regex patterns matching declaration headers across popular languages
CODE_BOUNDARY_PATTERNS = [
    # Python: def, async def, class
    re.compile(r"^(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\("),
    re.compile(r"^class\s+([a-zA-Z0-9_]+)\s*(?:\(|:)"),
    # JS/TS: function, class, exported const fn
    re.compile(r"^(?:export\s+)?(?:async\s+)?function(?:\s+([a-zA-Z0-9_]+)|\s*\()"),
    re.compile(r"^(?:export\s+)?class\s+([a-zA-Z0-9_]+)"),
    re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\("),
    # Go: func (r *Receiver) MethodName() or func FunctionName()
    re.compile(r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\("),
    re.compile(r"^type\s+([a-zA-Z0-9_]+)\s+(?:struct|interface)"),
    # Rust: fn, pub fn, struct, impl
    re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)"),
    re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?struct\s+([a-zA-Z0-9_]+)"),
    re.compile(r"^impl(?:\s+<[^>]+>)?\s+(?:[a-zA-Z0-9_]+(?:\s+for\s+)?)?([a-zA-Z0-9_]+)"),
    # Java / C# / C++: public/private method/class
    re.compile(r"^(?:public|private|protected|static|final|native|synchronized|\s)*class\s+([a-zA-Z0-9_]+)"),
]


class CodeChunker(BaseChunker):
    """Chunks code files by top-level functions, classes, and logical declaration blocks."""

    def __init__(
        self,
        max_chunk_tokens: int = 450,
        min_chunk_tokens: int = 40,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.fallback_chunker = SlidingWindowChunker(
            chunk_size=max_chunk_tokens,
            chunk_overlap=40,
            min_chunk_size=min_chunk_tokens,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        content = document.content
        if not content or not content.strip():
            return []

        lines = content.splitlines()
        total_lines = len(lines)

        # First pass: find boundary split lines
        boundaries: list[tuple[int, str]] = []  # [(line_idx, symbol_name)]

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "/*", "*")):
                continue

            for pat in CODE_BOUNDARY_PATTERNS:
                match = pat.match(line)
                if match:
                    symbol = match.group(1) if match.groups() and match.group(1) else stripped[:40]
                    boundaries.append((idx, symbol))
                    break

        # If no identifiable function/class boundaries found, use sliding window
        if not boundaries:
            return self.fallback_chunker.chunk(document)

        # Slice code between consecutive boundaries
        chunks: list[Chunk] = []

        # Include preamble (imports, package declaration, top comments)
        first_bound_line = boundaries[0][0]
        if first_bound_line > 1:
            preamble_lines = lines[: first_bound_line - 1]
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text and self.estimate_tokens(preamble_text) >= self.min_chunk_tokens:
                chunk_id = generate_chunk_id(
                    document.metadata.relative_path,
                    1,
                    first_bound_line - 1,
                    preamble_text,
                )
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    file_path=document.metadata.file_path,
                    relative_path=document.metadata.relative_path,
                    file_type=document.metadata.file_type,
                    start_line=1,
                    end_line=first_bound_line - 1,
                    char_count=len(preamble_text),
                    estimated_tokens=self.estimate_tokens(preamble_text),
                    section_title="[Imports & Preamble]",
                )
                chunks.append(Chunk(text=preamble_text, metadata=meta))

        # Process each bounded block
        for i, (start_line, symbol) in enumerate(boundaries):
            end_line = (boundaries[i + 1][0] - 1) if (i + 1 < len(boundaries)) else total_lines
            block_lines = lines[start_line - 1 : end_line]
            block_text = "\n".join(block_lines).strip()

            if not block_text:
                continue

            tokens = self.estimate_tokens(block_text)

            if tokens <= self.max_chunk_tokens:
                chunk_id = generate_chunk_id(
                    document.metadata.relative_path,
                    start_line,
                    end_line,
                    block_text,
                )
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    file_path=document.metadata.file_path,
                    relative_path=document.metadata.relative_path,
                    file_type=document.metadata.file_type,
                    start_line=start_line,
                    end_line=end_line,
                    char_count=len(block_text),
                    estimated_tokens=tokens,
                    section_title=symbol,
                )
                chunks.append(Chunk(text=block_text, metadata=meta))
            else:
                # Sub-chunk very large classes or functions using sliding window
                sub_doc = Document(
                    metadata=document.metadata,
                    content=block_text,
                )
                sub_chunks = self.fallback_chunker.chunk(sub_doc)
                for sc in sub_chunks:
                    sc.metadata.start_line += start_line - 1
                    sc.metadata.end_line += start_line - 1
                    sc.metadata.section_title = symbol
                    sc.metadata.chunk_id = generate_chunk_id(
                        document.metadata.relative_path,
                        sc.metadata.start_line,
                        sc.metadata.end_line,
                        sc.text,
                    )
                    chunks.append(sc)

        return chunks
