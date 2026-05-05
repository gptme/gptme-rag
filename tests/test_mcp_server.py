"""Tests for the MCP server wrapper.

Most tests only require gptme-rag itself; the mcp package is only needed
for ``test_build_server_returns_fastmcp_instance`` which is skipped when
the optional ``mcp`` extra is not installed.
"""

from __future__ import annotations

import pytest


def test_build_server_returns_fastmcp_instance(tmp_path):
    """build_server should return a FastMCP server bound to gptme-rag tools."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    server = build_server(persist_dir=tmp_path / "index")
    assert server is not None
    # FastMCP exposes a `name` attribute
    assert getattr(server, "name", None) == "gptme-rag"


def test_format_results_truncates_long_content():
    """_format_results should respect max_chars_per_doc."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _format_results

    long_content = "x" * 5000
    fake_doc = SimpleNamespace(content=long_content, metadata={"source": "/tmp/foo.md"})
    out = _format_results([fake_doc], [0.42], max_chars_per_doc=100)

    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(1.0 - 0.42)  # distance → similarity
    assert out[0]["source"] == "/tmp/foo.md"
    assert out[0]["content"].endswith("...[truncated]")
    assert len(out[0]["content"]) <= 100 + len("...[truncated]")


def test_format_results_handles_missing_metadata():
    """_format_results should not crash when document.metadata is None."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _format_results

    fake_doc = SimpleNamespace(content="hello", metadata=None)
    out = _format_results([fake_doc], [0.1])

    assert out[0]["source"] == ""
    assert out[0]["metadata"] == {}


def test_assemble_context_deduplicates_by_source():
    """_assemble_context should keep only the best entry per source file."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    doc_a = SimpleNamespace(
        content="Content from file A.", metadata={"source": "/tmp/a.md"}
    )
    doc_b = SimpleNamespace(
        content="Content from file B.", metadata={"source": "/tmp/b.md"}
    )
    doc_a2 = SimpleNamespace(
        content="Another chunk from file A.", metadata={"source": "/tmp/a.md"}
    )

    # a.md appears twice — only the first (higher-ranked) entry should survive
    output = _assemble_context([doc_a, doc_b, doc_a2], [0.1, 0.3, 0.5], "test query")
    assert "file A." in output
    assert "file B." in output
    assert "Another chunk" not in output  # deduped out


def test_assemble_context_includes_header_and_counts():
    """_assemble_context should include a query header and document count."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    doc = SimpleNamespace(content="Hello world.", metadata={"source": "/tmp/x.md"})
    output = _assemble_context([doc], [0.2], "find hello")
    assert 'Retrieved Context for: "find hello"' in output
    assert "1 relevant document(s) found" in output
    assert "[80% relevant]" in output  # 1.0 - 0.2 = 0.8
    assert "Hello world." in output


def test_assemble_context_clamps_relevance_percentage():
    """_assemble_context should keep displayed relevance percentages sane."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    poor_match = SimpleNamespace(
        content="Low relevance.", metadata={"source": "/tmp/low.md"}
    )
    impossible_match = SimpleNamespace(
        content="High relevance.", metadata={"source": "/tmp/high.md"}
    )

    output = _assemble_context(
        [poor_match, impossible_match],
        [1.5, -0.5],
        "test",
    )
    assert "[0% relevant] low.md" in output
    assert "[100% relevant] high.md" in output


def test_assemble_context_respects_max_chars():
    """_assemble_context should truncate when max_context_chars is hit."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    # Two long docs — cap is tight enough to truncate after the first
    doc1 = SimpleNamespace(content="A" * 2000, metadata={"source": "/tmp/first.md"})
    doc2 = SimpleNamespace(content="B" * 2000, metadata={"source": "/tmp/second.md"})
    output = _assemble_context([doc1, doc2], [0.1, 0.2], "test", max_context_chars=1000)
    assert "first.md" in output
    assert "second.md" not in output
    assert "omitted" in output.lower()
    assert "1 more document" in output


def test_assemble_context_cap_uses_rendered_markdown_length():
    """_assemble_context should measure the actual rendered Markdown output."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    doc1 = SimpleNamespace(content="A", metadata={"source": "/tmp/first.md"})
    doc2 = SimpleNamespace(content="B" * 200, metadata={"source": "/tmp/second.md"})
    first_only = _assemble_context([doc1], [0.1], "test")
    omission = "*(1 more document(s) omitted — context limit reached)*\n"
    cap = len(first_only) + len(omission) + 5

    output = _assemble_context([doc1, doc2], [0.1, 0.2], "test", max_context_chars=cap)
    assert "first.md" in output
    assert "second.md" not in output
    assert len(output) <= cap


def test_assemble_context_handles_missing_source():
    """_assemble_context should not crash on docs without metadata."""
    from types import SimpleNamespace

    from gptme_rag.mcp_server import _assemble_context

    doc = SimpleNamespace(content="No metadata here.", metadata=None)
    output = _assemble_context([doc], [0.5], "test")
    assert "No metadata here." in output
    assert "(unknown)" in output


def test_cli_mcp_command_registered():
    """The `gptme-rag mcp` subcommand should be discoverable via Click."""
    from gptme_rag.cli import cli

    assert (
        "mcp" in cli.commands
    ), f"Expected 'mcp' command in CLI; got {sorted(cli.commands)}"


# ---------------------------------------------------------------------------
# End-to-end tests — exercise the registered MCP tools via FastMCP.call_tool.
# These tests build a real ChromaDB index in a tmp dir, so they're slower
# (~30s for the first embedding model load). Skipped without the mcp extra.
# ---------------------------------------------------------------------------


def _call_tool(server, name: str, args: dict):
    """Run a FastMCP tool synchronously, returning the structured dict result."""
    import asyncio

    _content, structured = asyncio.run(server.call_tool(name, args))
    return structured


def _seed_docs(docs_dir):
    """Write two small docs designed to make a 'gptme assistant' query rank
    one above the other, so we can validate ordering deterministically."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "hello.md").write_text(
        "# Hello\n\nThe quick brown fox jumps over the lazy dog.\n"
    )
    (docs_dir / "gptme.md").write_text(
        "# gptme\n\ngptme is a personal AI assistant for the terminal.\n"
    )


@pytest.mark.slow
def test_index_status_starts_empty(tmp_path):
    """rag_index_status should report zero counts before any indexing."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    server = build_server(persist_dir=tmp_path / "idx")
    result = _call_tool(server, "rag_index_status", {})

    assert result["document_count"] == 0
    assert result["chunk_count"] == 0
    assert result["persist_dir"].endswith("idx")
    assert result["embedding_model"]  # non-empty string


@pytest.mark.slow
def test_index_refresh_reports_delta(tmp_path):
    """rag_index_refresh should report before/after/delta in unique source files."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    docs = tmp_path / "docs"
    _seed_docs(docs)

    server = build_server(persist_dir=tmp_path / "idx")
    result = _call_tool(
        server,
        "rag_index_refresh",
        {"directory": str(docs), "pattern": "**/*.md"},
    )

    assert result["documents_before"] == 0
    assert result["documents_after"] == 2
    assert result["documents_indexed_delta"] == 2
    assert result["pattern"] == "**/*.md"
    assert result["directory"].endswith("docs")


@pytest.mark.slow
def test_index_refresh_rejects_missing_directory(tmp_path):
    """rag_index_refresh should raise ValueError on a non-directory path."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    server = build_server(persist_dir=tmp_path / "idx")
    with pytest.raises(Exception) as exc_info:
        _call_tool(
            server,
            "rag_index_refresh",
            {"directory": str(tmp_path / "does-not-exist")},
        )
    # FastMCP wraps ValueError; underlying message should mention the path.
    assert "does-not-exist" in str(exc_info.value)


@pytest.mark.slow
def test_query_returns_ranked_results(tmp_path):
    """rag_query should rank semantically relevant docs higher.

    This is the canonical end-to-end smoke test: index two docs, query for
    one of them, and verify the relevant doc comes first with a higher
    similarity score.
    """
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    docs = tmp_path / "docs"
    _seed_docs(docs)

    server = build_server(persist_dir=tmp_path / "idx")
    _call_tool(
        server,
        "rag_index_refresh",
        {"directory": str(docs), "pattern": "**/*.md"},
    )
    result = _call_tool(server, "rag_query", {"query": "gptme assistant", "top_k": 2})

    # FastMCP returns structured output as {"result": [...]} for list returns.
    hits = (
        result["result"] if isinstance(result, dict) and "result" in result else result
    )
    assert isinstance(hits, list)
    assert len(hits) == 2

    # Ordering: gptme.md is more relevant to "gptme assistant" than the lazy-dog doc.
    assert hits[0]["source"].endswith(
        "gptme.md"
    ), f"Expected gptme.md ranked first, got {hits[0]['source']}"
    assert hits[1]["source"].endswith("hello.md")
    assert (
        hits[0]["score"] > hits[1]["score"]
    ), f"Expected gptme.md score > hello.md score; got {hits[0]['score']} vs {hits[1]['score']}"

    # Result shape contract.
    for h in hits:
        assert set(h.keys()) >= {"score", "source", "content", "metadata"}
        assert isinstance(h["score"], float)
        assert isinstance(h["content"], str)
        assert isinstance(h["metadata"], dict)


@pytest.mark.slow
def test_query_top_k_clamped(tmp_path):
    """rag_query should clamp top_k to [1, 50] without raising."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from gptme_rag.mcp_server import build_server

    docs = tmp_path / "docs"
    _seed_docs(docs)

    server = build_server(persist_dir=tmp_path / "idx")
    _call_tool(
        server,
        "rag_index_refresh",
        {"directory": str(docs), "pattern": "**/*.md"},
    )

    # top_k=0 → clamped to 1, returns exactly 1 hit (2 docs indexed, clamp is the binding constraint)
    result = _call_tool(server, "rag_query", {"query": "gptme", "top_k": 0})
    hits = (
        result["result"] if isinstance(result, dict) and "result" in result else result
    )
    assert len(hits) == 1

    # top_k=999 → clamped to 50, but we only have 2 docs so just verify no error
    result = _call_tool(server, "rag_query", {"query": "gptme", "top_k": 999})
    hits = (
        result["result"] if isinstance(result, dict) and "result" in result else result
    )
    assert len(hits) <= 2  # only have 2 docs in the test corpus
