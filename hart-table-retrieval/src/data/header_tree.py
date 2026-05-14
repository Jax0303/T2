"""
HiTab header tree.

HiTab data structure:
- table["top_root"]: column-header tree with children_dict (recursive)
- table["left_root"]: row-header tree with children_dict (recursive)
- table["data"]: 2D array of {"value": x} cells
- leaf nodes have line_idx pointing to the row/col they map to.
"""

from typing import List, Optional


class TreeNode:
    def __init__(self, value: str, line_idx: Optional[int] = None):
        self.value = value
        self.line_idx = line_idx
        self.children: List["TreeNode"] = []

    def __repr__(self):
        return f"TreeNode({self.value!r}, line_idx={self.line_idx}, children={len(self.children)})"


def _parse_hitab_node(d: dict) -> TreeNode:
    """Parse a HiTab top_root / left_root recursive node dict."""
    node = TreeNode(str(d.get("value", d.get("name", ""))), d.get("line_idx"))
    for child in d.get("children_dict", []):
        node.children.append(_parse_hitab_node(child))
    return node


class HeaderTree:
    def __init__(self):
        self.top_root: Optional[TreeNode] = None
        self.left_root: Optional[TreeNode] = None
        self._top_paths: List[List[str]] = []
        self._left_paths: List[List[str]] = []
        self._data: List[list] = []
        self._row_num = 0
        self._col_num = 0
        self._top_leaf_to_col: dict = {}  # line_idx -> col_idx
        self._left_leaf_to_row: dict = {}  # line_idx -> row_idx
        self._top_paths_by_col: dict = {}  # col_idx -> path
        self._left_paths_by_row: dict = {}  # row_idx -> path

    def build_tree(self, table_data: dict):
        top_root_data = table_data.get("top_root")
        left_root_data = table_data.get("left_root")

        if top_root_data:
            self.top_root = _parse_hitab_node(top_root_data)
        if left_root_data:
            self.left_root = _parse_hitab_node(left_root_data)

        self._data = table_data.get("data", [])
        self._row_num = len(self._data)
        self._col_num = len(self._data[0]) if self._data else 0

        # Extract paths
        if self.top_root:
            self._top_paths = self._extract_paths(self.top_root)
            self._build_leaf_index(self.top_root, self._top_leaf_to_col, self._top_paths_by_col)
        if self.left_root:
            self._left_paths = self._extract_paths(self.left_root)
            self._build_leaf_index(self.left_root, self._left_leaf_to_row, self._left_paths_by_row)

        return self

    def _extract_paths(self, root: TreeNode) -> List[List[str]]:
        """Extract root-to-leaf paths, excluding the root sentinel."""
        paths = []

        def dfs(node, current):
            # Skip sentinel root values
            if node.value not in ("<TOP>", "<LEFT>", "<ROOT>"):
                current = current + [node.value]

            if not node.children:
                if current:
                    paths.append(current)
            else:
                for c in node.children:
                    dfs(c, current)

        dfs(root, [])
        return paths

    def _build_leaf_index(self, root: TreeNode, leaf_to_idx: dict, paths_by_idx: dict):
        """Build line_idx -> path mapping for leaves."""

        def dfs(node, current):
            if node.value not in ("<TOP>", "<LEFT>", "<ROOT>"):
                current = current + [node.value]

            if not node.children and node.line_idx is not None:
                leaf_to_idx[node.line_idx] = node.line_idx
                paths_by_idx[node.line_idx] = current
            else:
                for c in node.children:
                    dfs(c, current)

        dfs(root, [])

    def get_all_paths(self) -> List[List[str]]:
        """Return all root-to-leaf paths from both top and left header trees.

        For HART, we return all combinations: each top-path joined with each left-path.
        But to keep manageable, return the union of top and left paths individually.
        """
        # Return union of top-only and left-only paths (deduped)
        seen = set()
        results = []
        for p in self._top_paths + self._left_paths:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                results.append(p)
        return results

    def get_top_paths(self) -> List[List[str]]:
        return self._top_paths

    def get_left_paths(self) -> List[List[str]]:
        return self._left_paths

    def get_path_for_cell(self, row_idx: int, col_idx: int) -> List[str]:
        """Combined left + top path for a data cell."""
        top_path = self._top_paths_by_col.get(col_idx, [])
        left_path = self._left_paths_by_row.get(row_idx, [])
        return list(left_path) + list(top_path)

    def get_cells_under_path(self, path: List[str]) -> List[str]:
        """Get all data cell values reachable under the given header path.

        Match path either as a prefix of a top-header path (then collect all cells
        in those columns) or as a prefix of a left-header path (rows).
        """
        if not path:
            return []

        cells = []

        # Try as top path prefix
        for col_idx, col_path in self._top_paths_by_col.items():
            if len(col_path) >= len(path) and col_path[: len(path)] == path:
                for row in self._data:
                    if col_idx < len(row):
                        cell = row[col_idx]
                        val = cell.get("value") if isinstance(cell, dict) else cell
                        if val not in (None, ""):
                            cells.append(str(val))

        # Try as left path prefix
        for row_idx, row_path in self._left_paths_by_row.items():
            if len(row_path) >= len(path) and row_path[: len(path)] == path:
                if row_idx < len(self._data):
                    for cell in self._data[row_idx]:
                        val = cell.get("value") if isinstance(cell, dict) else cell
                        if val not in (None, ""):
                            cells.append(str(val))

        return cells

    def get_depth(self) -> int:
        all_paths = self._top_paths + self._left_paths
        if all_paths:
            return max(len(p) for p in all_paths)
        return 1

    def get_spanning_cell_ratio(self) -> float:
        """Ratio of header internal-nodes (spanning cells) to total header cells."""
        if not (self.top_root or self.left_root):
            return 0.0

        total_header = 0
        spanning = 0

        def count(node):
            nonlocal total_header, spanning
            if node.value not in ("<TOP>", "<LEFT>", "<ROOT>"):
                total_header += 1
                if node.children:
                    spanning += 1
            for c in node.children:
                count(c)

        if self.top_root:
            count(self.top_root)
        if self.left_root:
            count(self.left_root)

        return spanning / max(total_header, 1)

    def get_leaf_headers(self) -> List[str]:
        """Get bottom-level column headers (leaf names from top_root)."""
        if not self.top_root:
            return [f"Col_{i}" for i in range(self._col_num)]

        result = [""] * self._col_num
        for col_idx, path in self._top_paths_by_col.items():
            if 0 <= col_idx < self._col_num and path:
                result[col_idx] = path[-1]
        return result

    def get_left_leaf_headers(self) -> List[str]:
        """Get row-leaf headers (left side)."""
        if not self.left_root:
            return [f"Row_{i}" for i in range(self._row_num)]
        result = [""] * self._row_num
        for row_idx, path in self._left_paths_by_row.items():
            if 0 <= row_idx < self._row_num and path:
                result[row_idx] = path[-1]
        return result

    def print_tree(self, root: Optional[TreeNode] = None, indent: int = 0):
        if root is None:
            print("--- TOP ---")
            if self.top_root:
                self.print_tree(self.top_root, 0)
            print("--- LEFT ---")
            if self.left_root:
                self.print_tree(self.left_root, 0)
            return

        prefix = "  " * indent
        marker = f" (line_idx={root.line_idx})" if root.line_idx is not None else ""
        print(f"{prefix}{root.value}{marker}")
        for c in root.children:
            self.print_tree(c, indent + 1)
