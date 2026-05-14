import json
from typing import List, Tuple

from .base import BaseSerializer


def _cell_value(c):
    if isinstance(c, dict):
        return c.get("value", "")
    return c


class JsonKeyValueSerializer(BaseSerializer):
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

        rows = []
        for i, row in enumerate(data):
            row_dict = {}
            if row_names and i < len(row_names) and row_names[i]:
                row_dict["__row__"] = row_names[i]
            for j, cell in enumerate(row):
                col_name = col_names[j] if j < len(col_names) else f"Col_{j}"
                row_dict[col_name] = _cell_value(cell)
            rows.append(row_dict)

        result = {"title": title, "rows": rows} if title else {"rows": rows}
        json_text = json.dumps(result, ensure_ascii=False, default=str)

        metadata = {"table_id": table_id, "path": [], "depth": 0}
        return [(json_text, metadata)]

    def get_name(self) -> str:
        return "json_kv"
