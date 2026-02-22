"""Tests for the file watcher functionality."""

import logging
import time

from gptme_rag.indexing.watcher import FileWatcher

logger = logging.getLogger(__name__)


def test_file_watcher_basic(tmp_path, indexer):
    """Test basic file watching functionality."""
    test_file = tmp_path / "test.txt"

    with FileWatcher(indexer, [str(tmp_path)], update_delay=0):
        # Create a new file
        test_file.write_text("Initial content")
        time.sleep(1)  # Wait for the watcher to process

        # Verify file was indexed
        results, _, _ = indexer.search("Initial content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == test_file.name

        # Modify the file
        test_file.write_text("Updated content")
        time.sleep(1)  # Wait for the watcher to process

        # Verify update was indexed
        results, _, _ = indexer.search("Updated content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == test_file.name


def test_file_watcher_pattern_matching(tmp_path, indexer):
    """Test that pattern matching works correctly."""
    with FileWatcher(indexer, [str(tmp_path)], pattern="*.txt", update_delay=0):
        # Create files with different extensions
        txt_file = tmp_path / "test.txt"
        py_file = tmp_path / "test.py"

        txt_file.write_text("Text file content")
        py_file.write_text("Python file content")

        time.sleep(1)  # Wait for the watcher to process

        # Verify only txt file was indexed
        results, _, _ = indexer.search("file content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == txt_file.name


def test_file_watcher_ignore_patterns(tmp_path, indexer):
    """Test that ignore patterns work correctly."""
    with FileWatcher(
        indexer, [str(tmp_path)], ignore_patterns=["*.ignore"], update_delay=0
    ):
        # Create an ignored file and a normal file
        ignored_file = tmp_path / "test.ignore"
        normal_file = tmp_path / "test.txt"

        ignored_file.write_text("Should be ignored")
        normal_file.write_text("Should be indexed")

        time.sleep(1)  # Wait for the watcher to process

        # Verify only normal file was indexed
        results, _, _ = indexer.search("Should be")
        assert len(results) == 1
        assert results[0].metadata["filename"] == normal_file.name


def test_file_watcher_move(tmp_path, indexer):
    """Test handling of file moves."""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "destination.txt"

    def wait_for_index(
        content: str, filename: str | None = None, retries: int = 15, delay: float = 0.3
    ) -> bool:
        """Wait for content to appear in index with retries."""
        logger.info(f"Waiting for content to be indexed: {content}")
        for attempt in range(retries):
            results, _, _ = indexer.search(content)
            if len(results) == 1:
                if filename is None or results[0].metadata["filename"] == filename:
                    logger.info(f"Found content after {attempt + 1} attempts")
                    return True
            logger.debug(f"Attempt {attempt + 1}: content not found, waiting {delay}s")
            time.sleep(delay)
        logger.warning(f"Content not found after {retries} attempts: {content}")
        return False

    with FileWatcher(indexer, [str(tmp_path)], update_delay=0):
        # Create source file and wait for it to be indexed
        src_file.write_text("Test content")
        assert wait_for_index("Test content", src_file.name), "Source file not indexed"

        # Move the file
        src_file.rename(dst_file)

        # Wait for the moved file to be indexed at new location
        assert wait_for_index("Test content", dst_file.name), "Moved file not indexed"

        # Final verification
        results, _, _ = indexer.search("Test content")
        assert len(results) == 1, "Expected exactly one result"
        assert (
            results[0].metadata["filename"] == dst_file.name
        ), "Wrong filename in metadata"


def test_file_watcher_delete(tmp_path, indexer):
    """Test that deleting a file removes it from the index."""
    test_file = tmp_path / "deleteme.txt"

    with FileWatcher(indexer, [str(tmp_path)], update_delay=0):
        # Create and verify file is indexed
        test_file.write_text("Unique deletable content")
        time.sleep(1)

        results, _, _ = indexer.search("Unique deletable content")
        assert len(results) == 1, "File should be indexed"
        assert results[0].metadata["filename"] == test_file.name

        # Delete the file
        test_file.unlink()
        time.sleep(1)

        # Clear cache so we get fresh results
        indexer.cache.clear()

        # Verify file was removed from index
        results, _, _ = indexer.search("Unique deletable content")
        assert len(results) == 0, "Deleted file should be removed from index"


def test_file_watcher_delete_one_of_many(tmp_path, indexer):
    """Test that deleting one file doesn't affect other indexed files."""
    file_a = tmp_path / "keep.txt"
    file_b = tmp_path / "remove.txt"

    with FileWatcher(indexer, [str(tmp_path)], update_delay=0):
        # Create two files with very different content
        file_a.write_text("Alpha bravo charlie delta")
        file_b.write_text("Xray yankee zulu whiskey")
        time.sleep(1)

        # Verify both are indexed
        results_a, _, _ = indexer.search("Alpha bravo charlie delta")
        assert len(results_a) >= 1
        results_b, _, _ = indexer.search("Xray yankee zulu whiskey")
        assert len(results_b) >= 1

        # Delete only file_b
        file_b.unlink()
        time.sleep(1)
        indexer.cache.clear()

        # file_a should still be in the index
        results_a, _, _ = indexer.search("Alpha bravo charlie delta")
        assert len(results_a) == 1, "Kept file should still be indexed"
        assert results_a[0].metadata["filename"] == file_a.name

        # file_b source should not appear in any results
        all_docs = indexer.list_documents(group_by_source=True)
        sources = [doc.metadata.get("source", "") for doc in all_docs]
        assert not any(
            file_b.name in s for s in sources
        ), "Deleted file should not appear in index"


def test_file_watcher_batch_updates(tmp_path, indexer):
    """Test handling of multiple rapid updates."""
    test_file = tmp_path / "test.txt"

    def verify_content(content: str, timeout: float = 5.0) -> bool:
        """Verify content appears in index within timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            results, _, _ = indexer.search(content, n_results=1)
            if results and content in results[0].content:
                return True
            time.sleep(0.5)
        logger.debug(f"Content not found within timeout: {content}")
        return False

    with FileWatcher(indexer, [str(tmp_path)], update_delay=0.5):
        # Wait for watcher to initialize
        time.sleep(1.0)

        # Test a few updates with longer delays
        for i in range(3):
            content = f"Content version {i}"
            test_file.write_text(content)
            time.sleep(1.0)  # Wait between updates
            if not verify_content(content):
                logger.error(f"Failed to verify content: {content}")
                raise AssertionError(f"Content not found: {content}")

        # Verify final state
        results, _, _ = indexer.search("Content version")
        assert len(results) == 1, "Expected exactly one result"
        assert "version 2" in results[0].content
