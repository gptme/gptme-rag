"""Integration tests for SmartRAGCache with Indexer search operations."""

import time

import pytest

from gptme_rag.indexing.document import Document
from gptme_rag.indexing.indexer import Indexer


@pytest.fixture
def temp_indexer(tmp_path):
    """Create a temporary indexer for testing."""
    indexer = Indexer(
        persist_directory=tmp_path / "test_index",
        collection_name="test_collection",
        enable_persist=True,
        embedding_function="minilm",  # Use default for speed
    )
    return indexer


@pytest.fixture
def indexed_documents(temp_indexer):
    """Index sample documents for testing."""
    docs = [
        Document(
            content="Python is a programming language",
            metadata={"source": "python.md", "type": "doc"},
        ),
        Document(
            content="JavaScript is also a programming language",
            metadata={"source": "javascript.md", "type": "doc"},
        ),
        Document(
            content="Machine learning is a subset of AI",
            metadata={"source": "ml.md", "type": "doc"},
        ),
    ]
    temp_indexer.add_documents(docs)
    return temp_indexer


def test_cache_miss_then_hit(indexed_documents):
    """Test that first search is cache miss, second is cache hit."""
    indexer = indexed_documents

    # Clear cache statistics
    indexer.cache.clear()

    # First search - should be cache miss
    query = "programming language"
    docs1, dist1, _ = indexer.search(query, n_results=2)

    assert indexer.cache.stats["misses"] == 1
    assert indexer.cache.stats["hits"] == 0
    assert len(docs1) > 0

    # Second search with same query - should be cache hit
    docs2, dist2, _ = indexer.search(query, n_results=2)

    assert indexer.cache.stats["hits"] == 1
    assert indexer.cache.stats["misses"] == 1

    # Results should be identical
    assert len(docs1) == len(docs2)
    assert docs1[0].content == docs2[0].content
    assert dist1 == dist2


def test_cache_different_queries(indexed_documents):
    """Test that different queries create different cache entries."""
    indexer = indexed_documents
    indexer.cache.clear()

    # First query
    docs1, _, _ = indexer.search("programming", n_results=2)
    assert indexer.cache.stats["misses"] == 1

    # Different query - should be another cache miss
    docs2, _, _ = indexer.search("machine learning", n_results=2)
    assert indexer.cache.stats["misses"] == 2
    assert indexer.cache.stats["hits"] == 0

    # Results should be different
    assert docs1[0].content != docs2[0].content


def test_cache_different_parameters(indexed_documents):
    """Test that different search parameters create different cache entries."""
    indexer = indexed_documents
    indexer.cache.clear()

    query = "programming"

    # Search with n_results=1
    docs1, _, _ = indexer.search(query, n_results=1)
    assert len(docs1) == 1
    assert indexer.cache.stats["misses"] == 1

    # Same query but n_results=2 - should be cache miss
    docs2, _, _ = indexer.search(query, n_results=2)
    assert len(docs2) == 2
    assert indexer.cache.stats["misses"] == 2

    # Original n_results=1 again - should be cache hit
    docs3, _, _ = indexer.search(query, n_results=1)
    assert indexer.cache.stats["hits"] == 1


def test_cache_ttl_expiry(indexed_documents):
    """Test that cache entries expire after TTL."""
    # Create indexer with 1-second TTL for testing
    indexer = indexed_documents
    indexer.cache.ttl_seconds = 1
    indexer.cache.clear()

    query = "programming"

    # First search - cache miss
    docs1, _, _ = indexer.search(query, n_results=2)
    assert indexer.cache.stats["misses"] == 1

    # Immediate second search - cache hit
    docs2, _, _ = indexer.search(query, n_results=2)
    assert indexer.cache.stats["hits"] == 1

    # Wait for TTL expiry
    time.sleep(1.1)

    # Search again - should be cache miss due to expiry
    docs3, _, _ = indexer.search(query, n_results=2)
    assert indexer.cache.stats["misses"] == 2


def test_cache_explain_mode_bypassed(indexed_documents):
    """Test that explain mode bypasses cache."""
    indexer = indexed_documents
    indexer.cache.clear()

    query = "programming"

    # First search with explain=True - should not use cache
    docs1, dist1, expl1 = indexer.search(query, n_results=2, explain=True)
    assert expl1 is not None  # Explanations should be returned
    assert indexer.cache.stats["misses"] == 0  # Cache not checked
    assert indexer.cache.stats["hits"] == 0

    # Second search with explain=False - should be cache miss
    docs2, dist2, expl2 = indexer.search(query, n_results=2, explain=False)
    assert expl2 is None
    assert indexer.cache.stats["misses"] == 1


def test_cache_performance_improvement(indexed_documents):
    """Test that cache provides performance improvement."""
    indexer = indexed_documents
    indexer.cache.clear()

    query = "programming language"

    # First search - measure time (cache miss)
    start1 = time.time()
    docs1, _, _ = indexer.search(query, n_results=2)
    time1 = time.time() - start1

    # Second search - measure time (cache hit)
    start2 = time.time()
    docs2, _, _ = indexer.search(query, n_results=2)
    time2 = time.time() - start2

    # Cache hit should be significantly faster
    # Allow some variance, but cache should be at least 2x faster
    assert (
        time2 < time1 * 0.5
    ), f"Cache hit ({time2:.4f}s) not faster than miss ({time1:.4f}s)"

    # Results should be identical
    assert docs1[0].content == docs2[0].content


def test_cache_statistics(indexed_documents):
    """Test cache statistics tracking."""
    indexer = indexed_documents
    indexer.cache.clear()

    query = "programming"

    # Perform multiple searches
    indexer.search(query, n_results=2)  # Miss
    indexer.search(query, n_results=2)  # Hit
    indexer.search(query, n_results=2)  # Hit
    indexer.search("different query", n_results=2)  # Miss

    stats = indexer.cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    # Calculate hit_rate from raw stats
    hit_rate = (
        stats["hits"] / (stats["hits"] + stats["misses"])
        if (stats["hits"] + stats["misses"]) > 0
        else 0
    )
    assert hit_rate == 0.5
    # Calculate entry_count
    entry_count = len(indexer.cache.cache)
    assert entry_count == 2


def test_cache_with_path_filters(indexed_documents):
    """Test cache with path_filters parameter."""
    indexer = indexed_documents
    indexer.cache.clear()

    query = "programming"

    # Search with path filter
    docs1, _, _ = indexer.search(query, n_results=2, path_filters=("*.md",))
    assert indexer.cache.stats["misses"] == 1

    # Same query, same filter - cache hit
    docs2, _, _ = indexer.search(query, n_results=2, path_filters=("*.md",))
    assert indexer.cache.stats["hits"] == 1

    # Same query, different filter - cache miss
    docs3, _, _ = indexer.search(query, n_results=2, path_filters=("*.py",))
    assert indexer.cache.stats["misses"] == 2


def test_cache_memory_limit(temp_indexer):
    """Test that cache respects memory limits."""
    indexer = temp_indexer

    # Set very small memory limit (1KB)
    indexer.cache.max_memory_bytes = 1024
    indexer.cache.clear()

    # Index many documents
    docs = [
        Document(
            content=f"Document {i} with some content " * 100,  # Large content
            metadata={"source": f"doc{i}.md"},
        )
        for i in range(10)
    ]
    indexer.add_documents(docs)

    # Perform searches - cache should evict entries to stay under limit
    for i in range(10):
        indexer.search(f"Document {i}", n_results=1)

    # Check that cache stayed within memory limit
    stats = indexer.cache.stats
    # Check total_size_bytes instead of memory_usage_mb
    assert stats["total_size_bytes"] <= 1024
    assert stats["evictions"] > 0  # Some evictions should have occurred
