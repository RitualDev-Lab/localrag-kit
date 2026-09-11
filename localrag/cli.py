"""Command-line interface for LocalRAG-Kit."""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from localrag import __version__
from localrag.core.ingestion import IngestionPipeline
from localrag.storage.sqlite_store import SQLiteStore


console = Console()


def get_default_db_path(target_path: Path) -> Path:
    """Return standard .localrag/index.db path relative to target folder."""
    return target_path / ".localrag" / "index.db"


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

    # Command: index
    index_parser = subparsers.add_parser("index", help="Index or incrementally update database for a directory")
    index_parser.add_argument("path", nargs="?", default=".", help="Target folder path (default: current directory)")
    index_parser.add_argument("--full", action="store_true", help="Force full re-index instead of incremental")
    index_parser.add_argument("--embed", action="store_true", help="Generate dense vector embeddings for semantic search")
    index_parser.add_argument("--embed-provider", default="auto", choices=["auto", "fast", "ollama", "openai"], help="Embedding backend (default: auto)")
    index_parser.add_argument("--embed-model", help="Embedding model name (e.g. nomic-embed-text, bge-m3)")

    # Command: search (keyword BM25 or Vector)
    search_parser = subparsers.add_parser("search", help="Search indexed database (BM25 keyword or vector semantic)")
    search_parser.add_argument("query", help="Search query string")
    search_parser.add_argument("--path", default=".", help="Workspace path containing .localrag/index.db")
    search_parser.add_argument("--mode", choices=["bm25", "vector"], default="bm25", help="Search algorithm (default: bm25)")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")

    # Command: ask (Grounded Hybrid RAG)
    ask_parser = subparsers.add_parser("ask", help="Ask a question grounded in local files with cited sources")
    ask_parser.add_argument("query", help="Question to ask")
    ask_parser.add_argument("--path", default=".", help="Workspace path containing .localrag/index.db")
    ask_parser.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid", help="Retrieval algorithm (default: hybrid)")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to synthesize (default: 5)")
    ask_parser.add_argument("--llm", default="auto", choices=["auto", "ollama", "mock", "openai"], help="LLM backend (default: auto)")
    ask_parser.add_argument("--model", help="LLM model name (e.g. qwen3.5:4b, llama3.2)")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Show index and database statistics")
    stats_parser.add_argument("--path", default=".", help="Workspace path containing .localrag/index.db")

    # Command: serve (FastAPI Web Server)
    serve_parser = subparsers.add_parser("serve", help="Start the local FastAPI REST & Web interface server")
    serve_parser.add_argument("path", nargs="?", default=".", help="Workspace path to serve (default: current directory)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--llm", default="auto", choices=["auto", "ollama", "mock", "openai"], help="LLM backend (default: auto)")
    serve_parser.add_argument("--model", help="LLM model name")
    serve_parser.add_argument("--embed", default="fast", choices=["auto", "fast", "ollama", "openai"], help="Embedding backend (default: fast)")
    serve_parser.add_argument("--open", action="store_true", help="Automatically open browser")

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

    elif args.command == "index":
        target_path = Path(args.path).resolve()
        if not target_path.exists():
            console.print(f"[bold red]Error:[/bold red] Target path does not exist: {target_path}")
            sys.exit(1)

        db_path = get_default_db_path(target_path)
        console.print(Panel(
            f"[bold cyan]LocalRAG-Kit Hybrid Indexer[/bold cyan]\n"
            f"[dim]Target workspace:[/dim] [yellow]{target_path}[/yellow]\n"
            f"[dim]Database path:[/dim] [green]{db_path}[/green]\n"
            f"[dim]Mode:[/dim] [{'magenta' if args.full else 'cyan'}]{'Full Rebuild' if args.full else 'Incremental Update'}[/]",
            border_style="cyan"
        ))

        pipeline = IngestionPipeline(target_path)
        embed_provider = None
        if args.embed:
            from localrag.providers import get_embedding_provider
            embed_provider = get_embedding_provider(
                provider_type=args.embed_provider,
                model=args.embed_model,
            )
            console.print(f"[dim]Embeddings enabled:[/dim] [cyan]{embed_provider.name}[/cyan] ({embed_provider.dimension} dims)")

        with SQLiteStore(db_path) as store:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[green]Indexing files...", total=None)

                def on_progress(meta, current, total):
                    progress.update(task, total=total, completed=current, description=f"[green]Indexing {meta.relative_path}...")

                stats = pipeline.index_to_store(
                    store,
                    incremental=not args.full,
                    embedding_provider=embed_provider,
                    progress_callback=on_progress,
                )

            table = Table(title="Indexing Summary", border_style="bright_blue")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Count", style="bold green")

            table.add_row("Total Files in Scope", str(stats.total_files_scanned))
            table.add_row("Files Indexed / Updated", str(stats.files_added_or_modified))
            table.add_row("Files Unchanged (Cached)", str(stats.files_unchanged))
            table.add_row("Files Deleted from Index", str(stats.files_deleted))
            table.add_row("New Chunks Stored", str(stats.chunks_indexed))

            console.print(table)
            console.print(f"[bold green][OK] Indexing complete![/bold green] Database stored at: {db_path}")

    elif args.command == "search":
        target_path = Path(args.path).resolve()
        db_path = get_default_db_path(target_path)
        if not db_path.exists():
            console.print(f"[bold red]Error:[/bold red] No index database found at: {db_path}")
            console.print("Run [yellow]localrag index[/yellow] first to create the index.")
            sys.exit(1)

        with SQLiteStore(db_path) as store:
            if args.mode == "vector":
                from localrag.providers import get_embedding_provider
                embed_provider = get_embedding_provider("fast")
                query_vec = embed_provider.embed_text(args.query)
                results = store.search_vector(query_vec, limit=args.limit)
            else:
                results = store.search_bm25(args.query, limit=args.limit)

            if not results:
                console.print(f"[yellow]No matching chunks found for query:[/yellow] '{args.query}'")
                return

            console.print(f"\n[bold green]Found {len(results)} matching chunks for:[/bold green] [italic cyan]'{args.query}'[/italic cyan] [dim](Mode: {args.mode})[/dim]\n")
            for i, r in enumerate(results, start=1):
                chunk = r.chunk
                console.print(Panel(
                    f"[bold yellow]{chunk.metadata.relative_path}[/bold yellow] "
                    f"[dim](Lines {chunk.metadata.start_line}-{chunk.metadata.end_line})[/dim] "
                    f"[green]Score: {r.score:.4f}[/green] [dim]({r.match_type})[/dim]\n"
                    f"[italic cyan]Section: {chunk.metadata.section_title or 'N/A'}[/italic cyan]\n\n"
                    f"{chunk.text}",
                    title=f"Result #{i} [{chunk.id}]",
                    border_style="cyan"
                ))

    elif args.command == "ask":
        target_path = Path(args.path).resolve()
        db_path = get_default_db_path(target_path)
        if not db_path.exists():
            console.print(f"[bold red]Error:[/bold red] No index database found at: {db_path}")
            console.print("Run [yellow]localrag index[/yellow] first to build the index.")
            sys.exit(1)

        from localrag.providers import get_embedding_provider, get_llm_provider
        from localrag.retrieval import RAGOrchestrator

        embed_provider = get_embedding_provider("fast")
        llm_provider = get_llm_provider(provider_type=args.llm, model=args.model)

        with SQLiteStore(db_path) as store:
            orchestrator = RAGOrchestrator(
                store=store,
                embedding_provider=embed_provider,
                llm_provider=llm_provider,
            )

            console.print(Panel(
                f"[bold cyan]LocalRAG Question Answerer[/bold cyan]\n"
                f"[dim]Query:[/dim] [yellow]{args.query}[/yellow]\n"
                f"[dim]Retrieval Mode:[/dim] [green]{args.mode}[/green] | [dim]LLM:[/dim] [cyan]{llm_provider.model_name}[/cyan]",
                border_style="cyan"
            ))

            citations_list = []

            def on_citations(citations):
                citations_list.extend(citations)

            console.print("\n[bold magenta]Generated Answer:[/bold magenta]")
            tokens = []
            for tok in orchestrator.stream_ask(
                query=args.query,
                top_k=args.top_k,
                mode=args.mode,
                citations_callback=on_citations,
            ):
                console.print(tok, end="")
                tokens.append(tok)
            console.print("\n")

            if citations_list:
                console.print("[bold cyan]Cited Local Sources:[/bold cyan]")
                for c in citations_list:
                    sec = f" ({c.section_title})" if c.section_title else ""
                    console.print(f"  [green]•[/green] [bold yellow][Source #{c.source_index}][/bold yellow] [white]{c.relative_path}:{c.start_line}-{c.end_line}[/white]{sec} [dim](score: {c.score:.4f})[/dim]")

    elif args.command == "stats":
        target_path = Path(args.path).resolve()
        db_path = get_default_db_path(target_path)
        if not db_path.exists():
            console.print(f"[bold red]Error:[/bold red] No database found at: {db_path}")
            sys.exit(1)

        with SQLiteStore(db_path) as store:
            stats = store.get_stats()
            size_mb = stats.database_size_bytes / (1024 * 1024)

            table = Table(title="LocalRAG Index Statistics", border_style="bright_blue")
            table.add_column("Statistic", style="bold cyan")
            table.add_column("Value", style="bold green")

            table.add_row("Database Location", stats.db_path)
            table.add_row("Total Files Indexed", str(stats.total_files))
            table.add_row("Total Chunks Stored", str(stats.total_chunks))
            table.add_row("Vectorized Chunks", str(stats.total_vectorized_chunks))
            table.add_row("Database File Size", f"{size_mb:.2f} MB ({stats.database_size_bytes:,} bytes)")

            console.print(table)

    elif args.command == "serve":
        target_path = Path(args.path).resolve()
        db_path = get_default_db_path(target_path)

        console.print(Panel(
            f"[bold cyan]LocalRAG-Kit FastAPI Server[/bold cyan]\n"
            f"[dim]Workspace:[/dim] [yellow]{target_path}[/yellow]\n"
            f"[dim]API URL:[/dim] [bold green]http://{args.host}:{args.port}/api[/bold green]\n"
            f"[dim]Interactive Docs:[/dim] [cyan]http://{args.host}:{args.port}/docs[/cyan]",
            border_style="cyan"
        ))

        import uvicorn
        from localrag.server.app import create_app

        app = create_app(
            workspace_path=target_path,
            db_path=db_path,
            embed_provider_name=args.embed,
            llm_provider_name=args.llm,
        )

        if args.open:
            import webbrowser
            webbrowser.open(f"http://{args.host}:{args.port}")

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
