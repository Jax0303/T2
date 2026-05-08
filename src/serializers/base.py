# SPDX-License-Identifier: MIT
"""Abstract base class for table serializers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.io.table_schema import Table


class SerializerBase(ABC):
    """Every serializer must implement serialize and parse."""

    @abstractmethod
    def serialize(self, table: Table) -> str:
        """Convert a Table object to a string representation."""

    @abstractmethod
    def parse(self, text: str) -> Table:
        """Reconstruct a Table object from its string representation.

        The reconstructed Table may lose information depending on the
        format (e.g., Markdown cannot preserve merged cells).
        """
