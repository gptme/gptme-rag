"""Tests for the CLI search command, focusing on --json flag."""

import json
import subprocess
import sys

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
            "--json",
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
            "--json",
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
            "--json",
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
    """Context info block includes total_tokens, included_results, and truncated."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "search",
            "machine learning",
            "--persist-dir",
            str(populated_index),
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    ctx = data["context"]
    assert "total_tokens" in ctx
    assert "truncated" in ctx
    assert "results_in_context" in ctx
    assert isinstance(ctx["total_tokens"], int)
    assert isinstance(ctx["truncated"], bool)
    assert isinstance(ctx["results_in_context"], int)
    # results_in_context <= total_results (equals when not truncated)
    assert ctx["results_in_context"] <= data["total_results"]
    if not ctx["truncated"]:
        assert ctx["results_in_context"] == data["total_results"]


def test_search_json_no_results(tmp_path):
    """JSON output for an empty index returns empty results list with context key."""
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
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["results"] == []
    assert data["total_results"] == 0
    # context key must be present even when there are no results (schema consistency)
    assert "context" in data
    assert data["context"]["total_tokens"] == 0
    assert data["context"]["truncated"] is False
    assert data["context"]["results_in_context"] == 0


def test_search_json_format_flag_warns(populated_index):
    """--format is silently ignored when --json is set, with a stderr warning."""
    # Use subprocess so we can capture real stderr separately from stdout
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gptme_rag.cli",
            "search",
            "Python",
            "--persist-dir",
            str(populated_index),
            "--json",
            "--format",
            "full",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # stdout should be valid JSON
    data = json.loads(result.stdout)
    assert "results" in data
    # Warning should appear on stderr
    assert "Warning" in result.stderr
    assert "--format" in result.stderr


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
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Should parse without error
    data = json.loads(result.output)
    assert isinstance(data, dict)


def test_search_json_expand_truncated_consistency(populated_index):
    """When --expand is used, truncated is always False (all results are returned)."""
    runner = CliRunner()
    for expand_mode in ("adjacent", "file"):
        result = runner.invoke(
            cli,
            [
                "search",
                "Python",
                "--persist-dir",
                str(populated_index),
                "--json",
                "--expand",
                expand_mode,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        ctx = data["context"]
        # truncated must be False when expand is active — all results are returned
        assert ctx["truncated"] is False, (
            f"--expand {expand_mode}: truncated={ctx['truncated']} but "
            f"results_in_context={ctx['results_in_context']} == total_results={data['total_results']}"
        )
        # results_in_context must equal total_results in expand mode
        assert ctx["results_in_context"] == data["total_results"]


def test_search_human_format_is_default(populated_index):
    """Default output is human-readable (not JSON)."""
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
