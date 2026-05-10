# SPDX-License-Identifier: MIT
"""HTML serializer — preserves rowspan / colspan."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from src.io.table_schema import Cell, HeaderNode, Table
from src.serializers.base import SerializerBase


class HtmlSerializer(SerializerBase):
    """Serialize / parse tables as HTML <table> elements."""

    # ------------------------------------------------------------------
    # serialize
    # ------------------------------------------------------------------
    def serialize(self, table: Table) -> str:
        """Convert *table* to an HTML ``<table>`` string.

        * Header cells (``is_header=True``) become ``<th>``.
        * ``row_span`` / ``col_span`` are emitted as attributes.
        * Covered cells (span == 0) are skipped (they are implied by the
          spanning origin cell).
        """
        parts: list[str] = ["<table>"]
        for row in table.cells:
            parts.append("  <tr>")
            for cell in row:
                if cell.row_span == 0 and cell.col_span == 0:
                    continue  # covered by a merge
                tag = "th" if cell.is_header else "td"
                attrs: list[str] = []
                if cell.row_span > 1:
                    attrs.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attrs.append(f'colspan="{cell.col_span}"')
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                parts.append(f"    <{tag}{attr_str}>{cell.value}</{tag}>")
            parts.append("  </tr>")
        parts.append("</table>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # parse
    # ------------------------------------------------------------------
    def parse(self, text: str) -> Table:
        """Reconstruct a Table from an HTML ``<table>`` string.

        Returns a Table whose ``cells`` grid is fully expanded — spanned
        positions contain placeholder ``Cell(value="", row_span=0, col_span=0)``.
        Header trees are **not** reconstructed (set to default virtual roots).
        """
        soup = BeautifulSoup(text, "html5lib")
        table_tag = soup.find("table")
        if not isinstance(table_tag, Tag):
            return Table(cells=[])

        rows_tags = table_tag.find_all("tr")
        if not rows_tags:
            return Table(cells=[])

        # First pass: figure out the grid dimensions.
        n_rows = len(rows_tags)
        n_cols = 0
        for tr in rows_tags:
            col_count = 0
            for td in tr.find_all(["td", "th"]):
                col_count += int(td.get("colspan", 1))
            n_cols = max(n_cols, col_count)

        # Initialise empty grid.
        grid: list[list[Cell | None]] = [[None] * n_cols for _ in range(n_rows)]
        merged: list[tuple[int, int, int, int]] = []

        for r_idx, tr in enumerate(rows_tags):
            c_idx = 0
            for td in tr.find_all(["td", "th"]):
                # Skip cells already filled by a previous rowspan.
                while c_idx < n_cols and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                if c_idx >= n_cols:
                    break
                rs = int(td.get("rowspan", 1))
                cs = int(td.get("colspan", 1))
                is_hdr = td.name == "th"
                value = td.get_text()

                grid[r_idx][c_idx] = Cell(
                    value=value, row_span=rs, col_span=cs, is_header=is_hdr,
                )
                if rs > 1 or cs > 1:
                    merged.append((r_idx, c_idx, r_idx + rs - 1, c_idx + cs - 1))

                # Fill covered cells.
                for dr in range(rs):
                    for dc in range(cs):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r_idx + dr, c_idx + dc
                        if rr < n_rows and cc < n_cols:
                            grid[rr][cc] = Cell(
                                value="", row_span=0, col_span=0, is_header=False,
                            )
                c_idx += cs

        # Replace any remaining None with empty cells.
        cells: list[list[Cell]] = []
        for row in grid:
            cells.append([c if c is not None else Cell(value="") for c in row])

        return Table(cells=cells, merged_cells=merged)
