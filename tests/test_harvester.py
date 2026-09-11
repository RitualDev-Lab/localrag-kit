"""Tests for recursive file harvesting and type categorization."""

import tempfile
from pathlib import Path
from localrag.core.harvester import FileHarvester
from localrag.core.models import FileType


def test_harvester_discovers_supported_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create nested file structure
        (root / "README.md").write_text("# Test Docs\nContent here.")
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("def hello():\n    return 'world'")
        (root / "src" / "utils.ts").write_text("export const add = (a: number, b: number) => a + b;")
        (root / "docs").mkdir()
        (root / "docs" / "page.html").write_text("<html><body><h1>Hello</h1></body></html>")
        
        # Ignored files
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("[core]")
        (root / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (root / "empty.txt").write_text("")  # Empty file (should be skipped)

        harvester = FileHarvester(root)
        files = harvester.harvest()

        rel_paths = {f.relative_path for f in files}

        assert "README.md" in rel_paths
        assert "src/app.py" in rel_paths
        assert "src/utils.ts" in rel_paths
        assert "docs/page.html" in rel_paths

        assert ".git/config" not in rel_paths
        assert "photo.png" not in rel_paths
        assert "empty.txt" not in rel_paths

        # Check metadata
        md_file = next(f for f in files if f.relative_path == "README.md")
        assert md_file.file_type == FileType.MARKDOWN
        assert md_file.line_count == 2

        py_file = next(f for f in files if f.relative_path == "src/app.py")
        assert py_file.file_type == FileType.CODE
        assert py_file.language == "python"
