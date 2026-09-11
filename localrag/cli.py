"""Command-line interface for LocalRAG-Kit."""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from localrag import __version__
from localrag.core.ingestion import IngestionPipeline


console = Console()


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="localrag",
        description="LocalRAG-Kit: 100% offline local document & codebase RAG engine",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: scan
    scan_parser = subparsers.add_parser("scan", help="Scan and analyze a directory without indexing")
    scan_parser.add_argument("path", nargs="?", default=".", help="Target folder path (default: current directory)")
    scan_parser.add_argument("--preview", action="store_true", help="Show preview of first 3 generated chunks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "scan":
        target_path = Path(args.path).resolve()
        if not target_path.exists():
            console.print(f"[bold red]Error:[/bold red] Target path does not exist: {target_path}")
            sys.exit(1)

        console.print(Panel(
            f"[bold cyan]LocalRAG-Kit File Harvester & Smart Chunker[/bold cyan]\n"
            f"[dim]Analyzing target folder:[/dim] [yellow]{target_path}[/yellow]",
            border_style="cyan"
        ))

        pipeline = IngestionPipeline(target_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Scanning & parsing files...", total=None)
            
            def on_progress(meta, current, total):
                progress.update(task, total=total, completed=current, description=f"[green]Processing {meta.relative_path}...")

            documents, stats = pipeline.process_directory(progress_callback=on_progress)

        # Print statistics table
        table = Table(title="Ingestion & Chunking Summary", border_style="bright_blue")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Files Scanned", str(stats.total_files_scanned))
        table.add_row("Files Successfully Parsed", str(stats.total_files_parsed))
        table.add_row("Total Chunks Produced", str(stats.total_chunks_produced))
        table.add_row("Total Characters", f"{stats.total_characters:,}")
        table.add_row("Estimated Tokens", f"{stats.total_estimated_tokens:,}")
        table.add_row("Failed / Skipped Files", str(stats.failed_files))

        console.print(table)

        if args.preview and documents:
            console.print("\n[bold magenta]Sample Generated Chunks Preview:[/bold magenta]")
            sample_count = 0
            for doc in documents:
                for chunk in doc.chunks:
                    sample_count += 1
                    console.print(Panel(
                        f"[bold yellow]{chunk.metadata.relative_path}[/bold yellow] "
                        f"[dim](Lines {chunk.metadata.start_line}-{chunk.metadata.end_line}, ~{chunk.metadata.estimated_tokens} tokens)[/dim]\n"
                        f"[italic cyan]Section: {chunk.metadata.section_title or 'N/A'}[/italic cyan]\n\n"
                        f"{chunk.text[:300]}...",
                        title=f"Chunk #{sample_count} [{chunk.metadata.chunk_id}]",
                        border_style="dim"
                    ))
                    if sample_count >= 3:
                        break
                if sample_count >= 3:
                    break


if __name__ == "__main__":
    main()
