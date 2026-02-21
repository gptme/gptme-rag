import logging
from dataclasses import dataclass

import tiktoken

from ..indexing.document import Document

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Represents an assembled context window with relevant documents."""

    content: str
    documents: list[Document]
    total_tokens: int
    truncated: bool = False


class ContextAssembler:
    """Assembles context windows from relevant documents."""

    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model(model)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))

    def _format_document(self, doc: Document) -> str:
        """Format a document for inclusion in the context window."""
        return doc.format_xml()

    def assemble_context(
        self,
        documents: list[Document],
        system_prompt: str | None = None,
        user_query: str | None = None,
    ) -> ContextWindow:
        """
        Assemble a context window from documents, staying within token limit.

        Documents should be pre-sorted by relevance.
        Duplicate documents will be filtered out and a warning will be logged.
        """
        total_tokens = 0
        included_docs: list[Document] = []
        context_parts = []
        truncated = False

        # Add system prompt if provided
        if system_prompt:
            system_tokens = self._count_tokens(system_prompt)
            if system_tokens < self.max_tokens:
                total_tokens += system_tokens
                context_parts.append(system_prompt)

        # Reserve tokens for user query if provided
        query_tokens = 0
        if user_query:
            query_tokens = self._count_tokens(user_query)
            total_tokens += query_tokens

        # Add documents until we hit token limit
        seen_contents: set[str] = set()
        for doc in documents:
            formatted_doc = self._format_document(doc)
            doc_tokens = self._count_tokens(formatted_doc)

            # check if document content is duplicate (O(1) lookup instead of O(n))
            if doc.content in seen_contents:
                logger.warning(f"Duplicate document found: {doc.metadata['source']}")
                continue

            if total_tokens + doc_tokens > self.max_tokens:
                truncated = True
                break

            total_tokens += doc_tokens
            context_parts.append(formatted_doc)
            included_docs.append(doc)
            seen_contents.add(doc.content)

        # Add user query at the end if provided
        if user_query:
            context_parts.append(f"User query: {user_query}")

        return ContextWindow(
            content="\n\n".join(context_parts),
            documents=included_docs,
            total_tokens=total_tokens,
            truncated=truncated,
        )
