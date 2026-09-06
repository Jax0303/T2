"""Index-unit serialization for the cell sentence (S3 / S3c / mt2net templates).

:mod:`.caption` renders one data cell as one sentence carrying the table's title
slot, its row header path, its column header path and the value; :mod:`.templates`
holds the named templates; :mod:`.base` the ``Chunk`` envelope and path helpers.
"""
from __future__ import annotations

from . import caption  # noqa: F401
from .base import Chunk, TableView, fmt_value, join_path, leaf  # noqa: F401
