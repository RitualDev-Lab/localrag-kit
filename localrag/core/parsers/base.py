"""Abstract Base Class for file content parsers."""

from abc import ABC, abstractmethod

from localrag.core.models import Document, FileMetadata, FileType


class BaseParser(ABC):
    """Interface for parsing a physical file into an in-memory Document."""

    @abstractmethod
    def can_handle(self, file_type: FileType) -> bool:
        """Return True if this parser supports the given FileType."""

    @abstractmethod
    def parse(self, metadata: FileMetadata) -> Document:
        """Extract text content and produce a Document object."""
