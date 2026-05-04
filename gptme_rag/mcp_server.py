"""MCP (Model Context Protocol) server exposing gptme-rag to MCP-capable agents.

Wraps the existing ``Indexer`` engine behind an MCP stdio server so Claude Code,
Cursor, Codex, gptme, and any other MCP client can run searches against an
existing gptme-rag index without going through the CLI.

This is a v1 prototype — see ``gptme/gptme-rag#22`` for scope and roadmap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_results(
    documents: list,
    scores: list[float],
    max_chars_per_doc: int = 2000,
) -> list[dict[str, Any]]:
    """Convert (documents, scores) into a JSON-serializable list."""
    out: list[dict[str, Any]] = []
    for doc, score in zip(documents, scores):
        content = doc.content or ""
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "...[truncated]"
        out.append(
            {
                "score": 1.0
                - float(score),  # convert distance → similarity (higher = better)
                "source": str(doc.metadata.get("source", "")) if doc.metadata else "",
                "content": content,
                "metadata": dict(doc.metadata) if doc.metadata else {},
            }
        )
    return out


def build_server(persist_dir: Path | None = None) -> Any:
    """Build the MCP server instance.

    Imports ``mcp`` lazily so the rest of gptme-rag works without the optional
    ``[mcp]`` extra installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only when extra missing
        raise ImportError(
            "MCP server requires the optional `mcp` extra. "
            "Install with: pip install gptme-rag[mcp]"
        ) from e

    from gptme_rag.indexing.indexer import Indexer

    server = FastMCP("gptme-rag")
    _indexer_cache: dict[Path, Indexer] = {}

    def _get_indexer(override_persist_dir: str | None = None) -> Indexer:
        target = Path(override_persist_dir) if override_persist_dir else persist_dir
        if target is None:
            target = Path.home() / ".cache" / "gptme-rag" / "default"
        target = target.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if target not in _indexer_cache:
            _indexer_cache[target] = Indexer(
                persist_directory=target, enable_persist=True
            )
        return _indexer_cache[target]

    @server.tool()
    def rag_query(
        query: str,
        top_k: int = 5,
        persist_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the gptme-rag index for content relevant to ``query``.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return (default 5, capped at 50).
            persist_dir: Optional override for the index directory.

        Returns:
            List of result dicts with keys ``score``, ``source``, ``content``, ``metadata``.
        """
        top_k = max(1, min(int(top_k), 50))
        indexer = _get_indexer(persist_dir)
        documents, scores, _ = indexer.search(query=query, n_results=top_k)
        return _format_results(documents, scores)

    @server.tool()
    def rag_index_status(persist_dir: str | None = None) -> dict[str, Any]:
        """Return summary stats for the active gptme-rag index.

        Args:
            persist_dir: Optional override for the index directory.

        Returns:
            Dict with ``persist_dir``, ``document_count`` (unique source files),
            ``chunk_count`` (total ChromaDB chunks), ``embedding_model``.
        """
        indexer = _get_indexer(persist_dir)
        status = indexer.get_status()
        return {
            "persist_dir": str(indexer.persist_directory),
            "document_count": status["document_count"],  # unique source files
            "chunk_count": status["chunk_count"],  # raw ChromaDB chunks
            "embedding_model": indexer.embedding_model_name,
        }

    @server.tool()
    def rag_index_refresh(
        directory: str,
        pattern: str = "**/*.*",
        persist_dir: str | None = None,
    ) -> dict[str, Any]:
        """Re-index ``directory`` into the gptme-rag store.

        Args:
            directory: Directory to walk for documents.
            pattern: Glob pattern (default ``**/*.*``, matches files with extensions).
            persist_dir: Optional override for the index directory.

        Returns:
            Dict with ``directory``, ``pattern``, ``documents_before``,
            ``documents_after``, ``documents_indexed_delta`` (all in unique
            source-file counts, not raw chunk counts).
        """
        directory_path = Path(directory).resolve()
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        indexer = _get_indexer(persist_dir)
        before = indexer.get_status()["document_count"]  # unique source files
        indexer.index_directory(directory_path, glob_pattern=pattern)
        indexer.cache.clear()  # flush stale cached queries so next rag_query sees fresh index
        after = indexer.get_status()["document_count"]  # unique source files

        return {
            "directory": str(directory_path),
            "pattern": pattern,
            "documents_before": before,
            "documents_after": after,
            "documents_indexed_delta": after - before,
        }

    return server


def run(persist_dir: Path | None = None) -> None:
    """Run the MCP server over stdio (blocking).

    Used as the entry point for ``gptme-rag mcp``.
    """
    server = build_server(persist_dir=persist_dir)
    server.run(transport="stdio")
