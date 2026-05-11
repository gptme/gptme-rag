"""Streaming document processor for efficient handling of large documents."""

import logging
from pathlib import Path
from collections.abc import Generator

import tiktoken

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents in a streaming fashion to manage memory efficiently."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunks: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            max_chunks: Maximum number of chunks to process (None for unlimited)
            encoding_name: Name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.encoding = tiktoken.get_encoding(encoding_name)

    def _token_byte_offsets(self, text: str, tokens: list[int]) -> list[int]:
        """Compute starting byte offset in ``text`` for each token.

        Works by decoding progressively longer token prefixes and measuring
        the encoded length, so it handles multi-byte UTF-8 characters and
        BPE edge cases correctly.

        Args:
            text: The original full text.
            tokens: The tokenized representation of ``text``.

        Returns:
            A list of byte offsets, one per token, where position ``i`` is
            the byte offset of token ``tokens[i]`` in ``text.encode('utf-8')``.
        """
        offsets: list[int] = []
        byte_pos = 0
        for token in tokens:
            offsets.append(byte_pos)
            byte_pos += len(self.encoding.decode([token]).encode("utf-8"))
        return offsets

    def process_text(
        self,
        text: str,
        metadata: dict | None = None,
        original_text: str | None = None,
    ) -> Generator[dict, None, None]:
        """Process text into chunks with metadata.

        Args:
            text: Text to process
            metadata: Optional metadata to include with each chunk
            original_text: Original full text for computing byte offsets
                (defaults to ``text`` when not provided).

        Yields:
            Dict containing chunk text, metadata, and ``byte_start``/``byte_end``
            offsets in the source text (UTF-8 byte positions, 0-indexed).
        """
        # Skip empty text
        if not text.strip():
            return

        source = original_text or text
        source_bytes = len(source.encode("utf-8"))

        try:
            # Encode text to tokens
            tokens = self.encoding.encode(text)

            # Pre-compute token-level byte offsets in the source text
            token_offsets = self._token_byte_offsets(source, tokens)

            # Calculate total chunks
            total_chunks = max(1, self.estimate_chunks(len(tokens)))

            # If text is short enough, yield as single chunk
            if len(tokens) <= self.chunk_size:
                yield {
                    "text": source,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": 0,
                        "token_count": len(tokens),
                        "total_chunks": 1,
                        "chunk_start": 0,
                        "chunk_end": len(tokens),
                        "byte_start": 0,
                        "byte_end": source_bytes,
                    },
                }
                return

            # Process text in chunks based on tokens
            chunk_start = 0
            chunk_count = 0

            while chunk_start < len(tokens):
                # Calculate chunk end
                chunk_end = min(chunk_start + self.chunk_size, len(tokens))

                # Get chunk tokens and decode
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_text = self.encoding.decode(chunk_tokens)

                # Compute byte range from pre-computed token offsets
                chunk_byte_start = token_offsets[chunk_start]
                if chunk_end < len(tokens):
                    chunk_byte_end = token_offsets[chunk_end]
                else:
                    chunk_byte_end = source_bytes

                # Log chunk info
                logger.debug(
                    f"Creating chunk {chunk_count} with {len(chunk_tokens)} tokens, "
                    f"{len(chunk_text)} chars (bytes {chunk_byte_start}-{chunk_byte_end}), "
                    f"{chunk_text[:50]}..."
                )

                # Create chunk metadata
                yield {
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": chunk_count,
                        "token_count": len(chunk_tokens),
                        "total_chunks": total_chunks,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "is_chunk": True,
                        "byte_start": chunk_byte_start,
                        "byte_end": chunk_byte_end,
                    },
                }

                # Calculate next chunk start
                if chunk_end == len(tokens):
                    # If we've reached the end, we're done
                    break

                # Move forward by at least one token, considering overlap
                next_start = chunk_start + max(1, self.chunk_size - self.chunk_overlap)
                chunk_start = min(next_start, len(tokens) - 1)
                chunk_count += 1

                # Check max chunks limit
                if self.max_chunks and chunk_count >= self.max_chunks:
                    break

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            # Yield the entire text as a single chunk if processing fails
            yield {
                "text": source,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": 0,
                    "token_count": len(text),
                    "total_chunks": 1,
                    "chunk_start": 0,
                    "chunk_end": len(text),
                    "error": str(e),
                    "byte_start": 0,
                    "byte_end": source_bytes,
                },
            }

    def is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary.

        Args:
            file_path: Path to the file

        Returns:
            True if the file appears to be binary
        """
        # Check file extension first
        binary_extensions = {".db", ".sqlite3", ".bin", ".pyc", ".so", ".dll", ".exe"}
        if file_path.suffix.lower() in binary_extensions:
            return True

        # Read first chunk and check for null bytes
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                return b"\x00" in chunk
        except OSError:
            return True

    def process_file(
        self,
        file_path: str | Path,
        metadata: dict | None = None,
        encoding: str = "utf-8",
    ) -> Generator[dict, None, None]:
        """Process a file in chunks.

        Args:
            file_path: Path to the file
            metadata: Optional metadata to include with each chunk
            encoding: File encoding

        Yields:
            Dict containing chunk text and metadata
        """
        path = Path(file_path)

        # Skip binary files
        if self.is_binary_file(path):
            return

        file_metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "file_type": path.suffix.lstrip("."),
            **(metadata or {}),
        }

        try:
            # Read the entire file content
            content = path.read_text(encoding=encoding)

            # Skip empty files
            if not content.strip():
                return

            # Record source byte size for provenance tracking
            file_metadata["source_bytes"] = len(content.encode("utf-8"))

            # Process the content
            yield from self.process_text(content, file_metadata)
        except UnicodeDecodeError:
            # Skip files that can't be decoded as text
            return

    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_chunks(self, token_count: int) -> int:
        """Estimate the number of chunks for a given token count.

        Args:
            token_count: Number of tokens

        Returns:
            Estimated number of chunks
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        effective_chunk_size = self.chunk_size - self.chunk_overlap
        chunks = (token_count + effective_chunk_size - 1) // effective_chunk_size

        if self.max_chunks:
            chunks = min(chunks, self.max_chunks)

        return chunks

    def get_optimal_chunk_size(
        self,
        target_chunks: int,
        token_count: int,
        min_chunk_size: int = 100,
    ) -> int:
        """Calculate optimal chunk size to achieve target number of chunks.

        Args:
            target_chunks: Desired number of chunks
            token_count: Total number of tokens
            min_chunk_size: Minimum chunk size

        Returns:
            Optimal chunk size
        """
        if target_chunks <= 0:
            raise ValueError("Target chunks must be positive")

        # Calculate base chunk size
        base_size = token_count // target_chunks

        # Adjust for overlap
        if self.chunk_overlap > 0:
            # Solve: (token_count) / (chunk_size - overlap) = target_chunks
            adjusted_size = (token_count // target_chunks) + self.chunk_overlap
        else:
            adjusted_size = base_size

        # Ensure minimum size
        return max(adjusted_size, min_chunk_size)
