"""Tests for text, code, and PDF parser components."""

import tempfile
from pathlib import Path
from localrag.core.models import FileMetadata, FileType
from localrag.core.parsers.code_parser import CodeParser
from localrag.core.parsers.text_parser import TextParser
from localrag.core.parsers import ParserRegistry


def test_text_parser():
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write("# Hello World\nThis is a sample markdown file.\nWith multiple lines.")
        f.flush()
        temp_path = f.name

    try:
        parser = TextParser()
        meta = FileMetadata(
            file_path=temp_path,
            relative_path="sample.md",
            file_type=FileType.MARKDOWN,
            sha256="abc",
        )
        doc = parser.parse(meta)
        assert "# Hello World" in doc.content
        assert doc.metadata.line_count == 3
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_html_tag_cleaning():
    with tempfile.NamedTemporaryFile("w+", suffix=".html", delete=False) as f:
        f.write("""
        <html>
            <head><style>body { color: red; }</style></head>
            <body>
                <script>console.log("noisy script");</script>
                <h1>Title</h1>
                <p>Paragraph text with &amp; entity.</p>
            </body>
        </html>
        """)
        f.flush()
        temp_path = f.name

    try:
        parser = TextParser()
        meta = FileMetadata(
            file_path=temp_path,
            relative_path="sample.html",
            file_type=FileType.HTML,
            sha256="abc",
        )
        doc = parser.parse(meta)
        assert "noisy script" not in doc.content
        assert "color: red" not in doc.content
        assert "Title" in doc.content
        assert "Paragraph text with & entity." in doc.content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_code_parser():
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as f:
        code_text = "import os\n\ndef add(a, b):\n    return a + b\n"
        f.write(code_text)
        f.flush()
        temp_path = f.name

    try:
        parser = CodeParser()
        meta = FileMetadata(
            file_path=temp_path,
            relative_path="math.py",
            file_type=FileType.CODE,
            sha256="abc",
            language="python",
        )
        doc = parser.parse(meta)
        assert "def add(a, b):" in doc.content
        assert doc.metadata.line_count == 4
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_parser_registry():
    registry = ParserRegistry()
    assert registry.get_parser(FileType.MARKDOWN) is not None
    assert registry.get_parser(FileType.CODE) is not None
    assert registry.get_parser(FileType.PDF) is not None
