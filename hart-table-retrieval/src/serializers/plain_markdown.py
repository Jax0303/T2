from typing import List, Tuple

from .base import BaseSerializer


def _cell_value(c) -> str:
    if isinstance(c, dict):
        v = c.get("value", "")
        return "" if v is None else str(v)
    return "" if c is None else str(c)


class PlainMarkdownSerializer(BaseSerializer):
    def serialize(self, table_data: dict, header_tree=None) -> List[Tuple[str, dict]]:
        table_id = table_data.get("table_id", table_data.get("uid", "unknown"))
        title = table_data.get("title", "")

        data = table_data.get("data", [])

        if header_tree is not None:
            col_names = header_tree.get_leaf_headers()
            row_names = header_tree.get_left_leaf_headers()
        else:
            col_names = []
            row_names = []

        col_num = len(data[0]) if data else 0
        if not col_names:
            col_names = [f"Col_{i}" for i in range(col_num)]
        if not row_names:
            row_names = [f"Row_{i}" for i in range(len(data))]

        lines = []
        if title:
            lines.append(f"# {title}")
            lines.append("")

        if col_names:
            header_row = ["row"] + [str(c) for c in col_names[:col_num]]
            lines.append("| " + " | ".join(header_row) + " |")
            lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")

        for i, row in enumerate(data):
            row_label = str(row_names[i]) if i < len(row_names) else f"Row_{i}"
            cells = [row_label] + [_cell_value(c).replace("|", "\\|") for c in row]
            lines.append("| " + " | ".join(cells) + " |")

        md_text = "\n".join(lines)

        metadata = {"table_id": table_id, "path": [], "depth": 0}
        return [(md_text, metadata)]

    def get_name(self) -> str:
        return "plain_markdown"
