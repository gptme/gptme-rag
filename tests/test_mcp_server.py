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


def test_cli_mcp_command_registered():
    """The `gptme-rag mcp` subcommand should be discoverable via Click."""
    from gptme_rag.cli import cli

    assert (
        "mcp" in cli.commands
    ), f"Expected 'mcp' command in CLI; got {sorted(cli.commands)}"
