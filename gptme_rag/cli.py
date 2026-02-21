import contextlib
import json
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax
from tqdm import tqdm

from .benchmark import RagBenchmark
from .embeddings import ModernBERTEmbedding
from .indexing.document import Document
from .indexing.indexer import Indexer
from .indexing.watcher import FileWatcher
from .query.context_assembler import ContextAssembler

console = Console()
logger = logging.getLogger(__name__)

# TODO: change this to a more appropriate location
default_persist_dir = Path.home() / ".cache" / "gptme" / "rag"


class ChunkMerger:
    @staticmethod
    def find_best_overlap(text1: str, text2: str) -> int:
        """Find the best overlap between the end of text1 and start of text2."""
        max_overlap = min(len(text1), len(text2))
        min_overlap = 20  # Minimum meaningful overlap

        for size in range(max_overlap, min_overlap - 1, -1):
            if text1[-size:] == text2[:size]:
                return size
        return 0

    @staticmethod
    def merge_chunks(chunks: list[Document]) -> str:
        """Merge multiple chunks into coherent text, removing overlaps."""
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0].content

        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.metadata.get("chunk_index", 0))

        # Start with first chunk
        merged_content = sorted_chunks[0].content

        # Merge subsequent chunks, removing overlaps
        for chunk in sorted_chunks[1:]:
            overlap_size = ChunkMerger.find_best_overlap(merged_content, chunk.content)
            if overlap_size > 0:
                merged_content += chunk.content[overlap_size:]
            else:
                merged_content += "\n" + chunk.content

        return merged_content

    @staticmethod
    def get_adjacent_chunks(doc: Document, indexer: Indexer) -> list[Document]:
        """Get adjacent chunks for a document."""
        if doc.doc_id is None:
            return []
        base_id = doc.doc_id.split("#chunk")[0]
        chunk_index = int(doc.metadata.get("chunk_index", 0))
        all_chunks = indexer.get_document_chunks(base_id)

        chunks = [doc]
        for chunk in all_chunks:
            if chunk.metadata.get("chunk_index") in [chunk_index - 1, chunk_index + 1]:
                chunks.append(chunk)

        return sorted(chunks, key=lambda x: x.metadata.get("chunk_index", 0))


class SearchOutputFormatter:
    def __init__(self, console: Console, raw: bool = False):
        self.console = console
        self.raw = raw

    def format_file(self, doc: Document, content: str) -> str:
        """Format content as a file."""
        source = doc.metadata.get("source", "unknown")
        return f'<file path="{source}">\n{content}\n</file>'

    def format_chunks(self, doc: Document, content: str) -> str:
        """Format content as chunks."""
        source = doc.metadata.get("source", "unknown")
        return f'<chunks path="{source}">\n{content}\n</chunks>'

    def _indent_content(self, content: str, indent: int = 0) -> str:
        """Format content without extra indentation for XML blocks."""
        return content

    def print_content(self, content: str, doc: Document):
        """Print content with optional syntax highlighting."""
        lexer = doc.metadata.get("extension", "").lstrip(".") or "text"
        self.console.print(
            content
            if self.raw
            else Syntax(content, lexer, theme="monokai", word_wrap=True)
        )

    def print_score(self, relevance: float):
        """Print relevance score."""
        self.console.print(f"\n[yellow]Relevance: {relevance:.2f}[/yellow]")

    def print_summary_header(self):
        """Print header for summary view."""
        self.console.print("\n[bold]Most Relevant Documents:[/bold]")

    def print_document_header(self, i: int, source: str):
        """Print document header in summary view."""
        self.console.print(f"\n[cyan]{i+1}. {source}[/cyan]")

    def print_preview(self, doc: Document):
        """Print document preview in summary view."""
        preview = doc.content[:200] + ("..." if len(doc.content) > 200 else "")
        lexer = doc.metadata.get("extension", "").lstrip(".") or "text"
        self.console.print("\n[bold]Preview:[/bold]")
        self.console.print(Syntax(preview, lexer, theme="monokai", word_wrap=True))

    def print_context_info(self, context):
        """Print context information."""
        self.console.print("\n[bold]Full Context:[/bold]")
        self.console.print(f"Total tokens: {context.total_tokens}")
        self.console.print(f"Documents included: {len(context.documents)}")
        self.console.print(f"Truncated: {context.truncated}")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """RAG implementation for gptme context management."""
    handler = RichHandler()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to index"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
@click.option(
    "--embedding-function",
    type=click.Choice(["modernbert", "default"]),
    default="modernbert",
    help="Embedding function to use (modernbert or default)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cpu",
    help="Device to run embeddings on (defaults to cpu)",
)
@click.option(
    "--force-recreate",
    is_flag=True,
    help="Force recreation of the collection",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Size of document chunks. Defaults based on model: ModernBERT-msmarco=512, ModernBERT-base=1000",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Overlap between chunks. Defaults based on model: ModernBERT-msmarco=50, ModernBERT-base=200",
)
def index(
    paths: list[Path],
    pattern: str,
    persist_dir: Path,
    embedding_function: str,
    device: str,
    force_recreate: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
):
    """Index documents in one or more directories."""
    if not paths:
        console.print("❌ No paths provided", style="red")
        return

    try:
        if embedding_function and not force_recreate:
            console.print(
                "[yellow]Warning:[/yellow] Changing embedding model may require recreating the collection. "
                "Use --force-recreate if you encounter dimension mismatch errors.",
                style="yellow",
            )

        indexer = Indexer(
            persist_directory=persist_dir,
            enable_persist=True,
            embedding_function=embedding_function,
            device=device,
            force_recreate=force_recreate,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Get existing files and their metadata from the index, using absolute paths
        existing_docs = indexer.get_all_documents()
        logger.debug("Found %d existing documents in index", len(existing_docs))

        existing_files = {}
        for doc in existing_docs:
            if "source" in doc.metadata:
                abs_path = os.path.abspath(doc.metadata["source"])
                last_modified = doc.metadata.get("last_modified")
                if last_modified:
                    try:
                        # Parse ISO format timestamp to float
                        existing_files[abs_path] = datetime.fromisoformat(
                            last_modified
                        ).timestamp()
                    except ValueError:
                        logger.warning(
                            "Invalid last_modified format: %s", last_modified
                        )
                        existing_files[abs_path] = 0
                else:
                    existing_files[abs_path] = 0
                # logger.debug("Existing file: %s", abs_path)  # Too spammy

        logger.debug("Loaded %d existing files from index", len(existing_files))

        # First, collect all documents and filter for new/modified
        all_documents = []
        with console.status("Collecting documents...") as status:
            for path in paths:
                if path.is_file():
                    status.update(f"Processing file: {path}")
                else:
                    status.update(f"Processing directory: {path}")

                documents = indexer.collect_documents(path)

                # Filter for new or modified documents
                filtered_documents = []
                for doc in documents:
                    source = doc.metadata.get("source")
                    if source:
                        # Resolve to absolute path for consistent comparison
                        abs_source = os.path.abspath(source)
                        doc.metadata["source"] = abs_source
                        current_mtime = os.path.getmtime(abs_source)

                        # Include if file is new or modified
                        if abs_source not in existing_files:
                            logger.debug("New file: %s", abs_source)
                            filtered_documents.append(doc)
                        # Round to microseconds (6 decimal places) for comparison
                        elif round(current_mtime, 6) > round(
                            existing_files[abs_source], 6
                        ):
                            logger.debug(
                                "Modified file: %s (current: %s, stored: %s)",
                                abs_source,
                                current_mtime,
                                existing_files[abs_source],
                            )
                            filtered_documents.append(doc)
                        else:
                            logger.debug("Unchanged file: %s", abs_source)

                all_documents.extend(filtered_documents)

        if not all_documents:
            console.print("No new or modified documents to index", style="yellow")
            return

        # Then process them with a progress bar
        n_files = len(set(doc.metadata.get("source", "") for doc in all_documents))
        n_chunks = len(all_documents)

        logger.info(f"Found {n_files} new/modified files to index ({n_chunks} chunks)")

        with tqdm(
            total=n_chunks,
            desc="Indexing documents",
            unit="chunk",
            disable=not sys.stdout.isatty(),
        ) as pbar:
            for progress in indexer.add_documents_progress(all_documents):
                pbar.update(progress)

        console.print(
            f"✅ Successfully indexed {n_files} files ({n_chunks} chunks)",
            style="green",
        )
    except Exception as e:
        console.print(f"❌ Error indexing directory: {e}", style="red")
        if logger.isEnabledFor(logging.DEBUG):
            console.print_exception()


@cli.command()
@click.argument("query")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@click.option("--n-results", "-n", default=5, help="Number of results to return")
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
@click.option("--max-tokens", default=4000, help="Maximum tokens in context window")
@click.option(
    "--format",
    type=click.Choice(["summary", "full"]),
    default="summary",
    help="Output format: summary (preview only) or full (complete content)",
)
@click.option(
    "--expand",
    type=click.Choice(["none", "adjacent", "file"]),
    default="none",
    help="Context expansion: none (matched chunks), adjacent (with neighboring chunks), file (entire file)",
)
@click.option("--raw", is_flag=True, help="Skip syntax highlighting")
@click.option(
    "--score",
    is_flag=True,
    help="Output relevance scores",
)
@click.option("--explain", is_flag=True, help="Show scoring explanations")
@click.option(
    "--weights",
    type=click.STRING,
    help="Custom scoring weights as JSON string, e.g. '{\"recency_boost\": 0.3}'",
)
@click.option(
    "--embedding-function",
    type=click.Choice(["modernbert", "default"]),
    help="Embedding function to use (modernbert or default)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to run embeddings on (cuda or cpu)",
)
@click.option(
    "--filter",
    "-f",
    multiple=True,
    help="Filter results by path pattern (glob). Can be specified multiple times.",
)
def search(
    query: str,
    paths: list[Path],
    n_results: int,
    persist_dir: Path,
    max_tokens: int,
    score: bool,
    format: str,
    expand: str,
    raw: bool,
    explain: bool,
    weights: str | None,
    embedding_function: str | None,
    device: str | None,
    filter: tuple[str, ...],
):
    """Search the index and assemble context."""
    paths = [path.resolve() for path in paths]

    # Hide ChromaDB output during initialization and search
    with console.status("Initializing..."):
        # Parse custom weights if provided
        scoring_weights = None
        if weights:
            try:
                scoring_weights = json.loads(weights)
            except json.JSONDecodeError as e:
                console.print(f"❌ Invalid weights JSON: {e}", style="red")
                return
            except Exception as e:
                console.print(f"❌ Error parsing weights: {e}", style="red")
                return

        # Redirect stdout to suppress ChromaDB output (using context manager for safety)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            # Initialize indexer with explicit arguments
            # Always use ModernBERT by default for better results
            indexer = Indexer(
                persist_directory=persist_dir,
                enable_persist=True,
                scoring_weights=scoring_weights,
                embedding_function=(
                    "modernbert" if embedding_function is None else embedding_function
                ),
                device=device or "cpu",
            )
            assembler = ContextAssembler(max_tokens=max_tokens)

            # Combine paths and filters for search
            search_paths = list(paths)
            if filter:
                # If no paths were specified but filters are present,
                # search from root and apply filters
                if not paths:
                    search_paths = [Path(".")]
                logger.debug(f"Using path filters: {filter}")

            if explain:
                documents, distances, explanations = indexer.search(
                    query,
                    n_results=n_results,
                    paths=search_paths,
                    path_filters=filter,
                    explain=True,
                )
            else:
                documents, distances, _ = indexer.search(
                    query, n_results=n_results, paths=search_paths, path_filters=filter
                )

    if not documents:
        console.print("No results found", style="yellow")
        return

    # Debug info in verbose mode
    logger.debug(f"Found {len(documents)} documents")
    for doc in documents:
        logger.debug(f"Document: {doc.doc_id}, source: {doc.metadata.get('source')}")

    # Assemble context window
    context = assembler.assemble_context(documents, user_query=query)

    def get_expanded_content(doc: Document, expand: str, indexer: Indexer) -> str:
        """Get content based on expansion mode.

        When expand='file' is used, the content is read directly from the filesystem
        to ensure freshness. The chunks are verified against the current file content
        to ensure they match, preventing display of outdated content.

        When expand='adjacent', neighboring chunks are retrieved from the index
        and merged.
        """
        logger.debug(f"Expanding content with mode: {expand}")
        logger.debug(f"Document ID: {doc.doc_id}")
        source = doc.metadata.get("source")
        logger.debug(f"Source: {source}")

        if expand == "file":
            if not source or not Path(source).is_file():
                logger.warning(f"Source file not found: {source}")
                return doc.content

            try:
                # Read fresh content directly from filesystem
                content = Path(source).read_text()

                # Verify that chunk content exists in current file
                if doc.content not in content:
                    logger.warning(f"Chunk content not found in current file: {source}")
                    return doc.content

                return content
            except Exception as e:
                logger.error(f"Error reading file {source}: {e}")
                return doc.content

        chunks = [doc]
        if expand == "adjacent":
            chunks = ChunkMerger.get_adjacent_chunks(doc, indexer)
            logger.debug(f"Found {len(chunks)} adjacent chunks")

        return ChunkMerger.merge_chunks(chunks)

    # Initialize output formatter
    formatter = SearchOutputFormatter(console, raw)

    # Handle full format with expanded context
    if format == "full":
        for i, doc in enumerate(documents):
            # Show relevance info first
            if distances and score:
                formatter.print_score(1 - distances[i])

            # Get and format content
            content = get_expanded_content(doc, expand, indexer)
            formatted = (
                formatter.format_file(doc, content)
                if expand == "file"
                else formatter.format_chunks(doc, content)
            )

            # Display content
            formatter.print_content(formatted, doc)
            console.print()
        return

    # Show summary view
    formatter.print_summary_header()

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        formatter.print_document_header(i, source)

        # Show scoring explanation if requested
        if explain and explanations:
            explanation = explanations[i]
            console.print("\n[bold]Scoring Breakdown:[/bold]")

            # Show individual score components
            scores = explanation.get("scores", {})
            for factor, score in scores.items():
                # Color code the scores
                if score > 0:
                    score_color = "green"
                    sign = "+"
                elif score < 0:
                    score_color = "red"
                    sign = ""
                else:
                    score_color = "yellow"
                    sign = " "

                # Print score and explanation
                console.print(
                    f"  {factor:15} [{score_color}]{sign}{score:>6.3f}[/{score_color}] | {explanation['explanations'][factor]}"
                )

            # Show total score
            total = explanation["total_score"]
            console.print(f"\n  {'Total':15} [bold blue]{total:>7.3f}[/bold blue]")
        else:
            # Just show the base relevance score
            if distances and score:
                formatter.print_score(1 - distances[i])

        # Display preview
        formatter.print_preview(doc)

    # Show assembled context info
    formatter.print_context_info(context)


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to index"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
@click.option(
    "--ignore-patterns",
    "-i",
    multiple=True,
    default=[],
    help="Glob patterns to ignore",
)
@click.option(
    "--embedding-function",
    type=click.Choice(["modernbert", "default"]),
    help="Embedding function to use (modernbert or default)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to run embeddings on (cuda or cpu)",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Size of document chunks. Defaults based on model: ModernBERT-msmarco=512, ModernBERT-base=1000",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Overlap between chunks. Defaults based on model: ModernBERT-msmarco=50, ModernBERT-base=200",
)
def watch(
    directory: Path,
    pattern: str,
    persist_dir: Path,
    ignore_patterns: list[str],
    embedding_function: str | None,
    device: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
):
    """Watch directory for changes and update index automatically."""
    try:
        # Initialize indexer with explicit arguments
        indexer = Indexer(
            persist_directory=persist_dir,
            enable_persist=True,
            embedding_function=(
                "modernbert" if embedding_function is None else embedding_function
            ),
            device=device or "cpu",
            chunk_size=chunk_size,  # Now optional in Indexer
            chunk_overlap=chunk_overlap,  # Now optional in Indexer
        )

        # Initial indexing
        console.print(f"Performing initial indexing of {directory}")
        with console.status("Indexing..."):
            indexer.index_directory(directory, pattern)

        console.print("Starting file watcher...")

        try:
            # TODO: FileWatcher should use same gitignore patterns as indexer
            file_watcher = FileWatcher(
                indexer, [str(directory)], pattern, ignore_patterns
            )
            with file_watcher:
                console.print("Watching for changes. Press Ctrl+C to stop.")
                # Keep the main thread alive

                try:
                    signal.pause()
                except AttributeError:  # Windows doesn't have signal.pause
                    while True:
                        time.sleep(1)
        except KeyboardInterrupt:
            console.print("\nStopping file watcher...")

    except Exception as e:
        console.print(f"❌ Error watching directory: {e}", style="red")
        console.print_exception()


@cli.command()
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt",
)
def clean(persist_dir: Path, force: bool):
    """Clear the index by removing the persist directory."""
    try:
        if not persist_dir.exists():
            console.print("✅ Index directory does not exist", style="green")
            return

        if not force:
            if not click.confirm(
                f"Are you sure you want to delete the index at {persist_dir}?"
            ):
                console.print("Operation cancelled", style="yellow")
                return

        shutil.rmtree(persist_dir)
        console.print("✅ Successfully cleared the index", style="green")
    except Exception as e:
        console.print(f"❌ Error clearing index: {e}", style="red")
        if logging.getLogger().level <= logging.DEBUG:
            console.print_exception()


@cli.command()
def status():
    """Show the status of the index."""
    try:
        with console.status("Getting index status..."):
            indexer = Indexer(
                persist_directory=default_persist_dir, enable_persist=True
            )
            status = indexer.get_status()

        # Print basic information
        console.print("\n[bold]Index Status[/bold]")
        console.print(f"Collection: [cyan]{status['collection_name']}[/cyan]")
        console.print(f"Storage Type: [cyan]{status['storage_type']}[/cyan]")
        if "persist_directory" in status:
            console.print(
                f"Persist Directory: [cyan]{status['persist_directory']}[/cyan]"
            )

        # Print document statistics
        console.print("\n[bold]Document Statistics[/bold]")
        console.print(f"Total Documents: [green]{status['document_count']:,}[/green]")
        console.print(f"Total Chunks: [green]{status['chunk_count']:,}[/green]")

        # Print source statistics
        if status["source_stats"]:
            console.print("\n[bold]Source Statistics[/bold]")
            for ext, count in sorted(
                status["source_stats"].items(), key=lambda x: x[1], reverse=True
            ):
                ext_display = ext if ext else "no extension"
                percentage = (
                    (count / status["chunk_count"]) * 100
                    if status["chunk_count"]
                    else 0
                )
                console.print(
                    f"  {ext_display:12} [yellow]{count:4}[/yellow] chunks ([yellow]{percentage:4.1f}%[/yellow])"
                )

        # Print configuration
        console.print("\n[bold]Configuration[/bold]")
        console.print(
            f"Chunk Size: [blue]{status['config']['chunk_size']:,}[/blue] tokens"
        )
        console.print(
            f"Chunk Overlap: [blue]{status['config']['chunk_overlap']:,}[/blue] tokens"
        )
        if "embedding_model" in status["config"]:
            model_name = status["config"]["embedding_model"]
            if model_name == "ModernBERT":
                # Get more specific model info from the indexer
                if isinstance(indexer.embedding_function, ModernBERTEmbedding):
                    if indexer.embedding_function.is_msmarco:
                        model_name = "ModernBERT-msmarco (optimized for retrieval)"
                    else:
                        model_name = "ModernBERT-base (general purpose)"
            console.print(f"Embedding Model: [blue]{model_name}[/blue]")

    except Exception as e:
        console.print(f"❌ Error getting index status: {e}", style="red")
        if logging.getLogger().level <= logging.DEBUG:
            console.print_exception()


@cli.group()
def benchmark():
    """Run performance benchmarks."""
    pass


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to benchmark"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def indexing(directory: Path, pattern: str, persist_dir: Path | None):
    """Benchmark document indexing performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    with console.status("Running indexing benchmark..."):
        benchmark.run_indexing_benchmark(directory, pattern)

    benchmark.print_results()


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--queries",
    "-q",
    multiple=True,
    default=["test", "document", "example"],
    help="Queries to benchmark",
)
@click.option(
    "--n-results",
    "-n",
    default=5,
    help="Number of results per query",
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def search_benchmark(
    directory: Path,
    queries: list[str],
    n_results: int,
    persist_dir: Path | None,
):
    """Benchmark search performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    # First index the directory
    with console.status("Indexing documents..."):
        benchmark.run_indexing_benchmark(directory)

    # Then run search benchmark
    with console.status("Running search benchmark..."):
        benchmark.run_search_benchmark(list(queries), n_results)

    benchmark.print_results()


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--duration",
    "-d",
    default=5.0,
    help="Duration of the benchmark in seconds",
)
@click.option(
    "--updates-per-second",
    "-u",
    default=2.0,
    help="Number of updates per second",
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def watch_perf(
    directory: Path,
    duration: float,
    updates_per_second: float,
    persist_dir: Path | None,
):
    """Benchmark file watching performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    with console.status("Running file watching benchmark..."):
        benchmark.run_watch_benchmark(
            directory,
            duration=duration,
            updates_per_second=updates_per_second,
        )

    benchmark.print_results()


def main(args=None):
    """Entry point for the CLI."""
    return cli(args=args)


if __name__ == "__main__":
    main()
