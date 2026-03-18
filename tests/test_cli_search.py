"""Tests for the CLI search command, focusing on --output-format json."""

import json

import pytest
from click.testing import CliRunner

from gptme_rag.cli import cli
from gptme_rag.indexing.document import Document
from gptme_rag.indexing.indexer import Indexer


@pytest.fixture
def populated_index(tmp_path):
    """Create and populate a temporary index for CLI testing."""
    indexer = Indexer(
        persist_directory=tmp_path / "index",
        enable_persist=True,
        chunk_size=200,
        chunk_overlap=20,
    )
    docs = [
        Document(
            content="Python is a high-level programming language known for readability.",
            metadata={"source": str(tmp_path / "python.txt"), "extension": ".txt"},
            doc_id="python",
        ),
        Document(
            content="Machine learning uses statistical methods to learn from data.",
            metadata={"source": str(tmp_path / "ml.txt"), "extension": ".txt"},
            doc_id="ml",
        ),
    ]
    indexer.add_documents(docs)
    return tmp_path / "index"


def test_search_json_output_structure(populated_index):
    """JSON output has the required top-level keys."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "Python programming",
            "--persist-dir",
            str(populated_index),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "query" in data
    assert "results" in data
    assert "total_results" in data
    assert "context" in data


def test_search_json_output_query_echoed(populated_index):
    """The query is echoed back in JSON output."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "Python programming",
            "--persist-dir",
            str(populated_index),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["query"] == "Python programming"


def test_search_json_output_result_fields(populated_index):
    """Each result has source, relevance, content, and metadata fields."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "Python",
            "--persist-dir",
            str(populated_index),
            "--n-results",
            "1",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total_results"] >= 1
    first = data["results"][0]
    assert "source" in first
    assert "relevance" in first
    assert "content" in first
    assert "metadata" in first
    # relevance is a float in [0, 1]
    assert isinstance(first["relevance"], float)
    assert 0.0 <= first["relevance"] <= 1.0


def test_search_json_output_context_info(populated_index):
    """Context info block includes total_tokens and truncated."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "machine learning",
            "--persist-dir",
            str(populated_index),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    ctx = data["context"]
    assert "total_tokens" in ctx
    assert "truncated" in ctx
    assert isinstance(ctx["total_tokens"], int)
    assert isinstance(ctx["truncated"], bool)


def test_search_json_no_results(tmp_path):
    """JSON output for an empty index returns empty results list."""
    # Create the index directory (CLI requires it to exist)
    Indexer(
        persist_directory=tmp_path / "empty",
        enable_persist=True,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "anything",
            "--persist-dir",
            str(tmp_path / "empty"),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["results"] == []
    assert data["total_results"] == 0


def test_search_json_output_is_valid_json(populated_index):
    """Output is valid JSON (no rich markup, no extra text)."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "data",
            "--persist-dir",
            str(populated_index),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Should parse without error
    data = json.loads(result.output)
    assert isinstance(data, dict)


def test_search_human_format_is_default(populated_index):
    """Default output format is human-readable (not JSON)."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["search", "Python", "--persist-dir", str(populated_index)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Human output should NOT be valid JSON at the top level
    try:
        json.loads(result.output)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "Default output should be human-readable, not JSON"
