from typing import List, Tuple

from .base import BaseSerializer


class HeaderPathSerializer(BaseSerializer):
    def serialize(self, table_data: dict, header_tree) -> List[Tuple[str, dict]]:
        table_id = table_data.get("table_id", table_data.get("uid", "unknown"))
        title = table_data.get("title", "")

        paths = header_tree.get_all_paths()
        if not paths:
            text = title or "empty table"
            return [(text, {"table_id": table_id, "path": [], "depth": 0})]

        results = []
        for path in paths:
            cells = header_tree.get_cells_under_path(path)
            path_str = " > ".join(path)
            cells_str = ", ".join(cells[:200]) if cells else ""  # cap to avoid huge text

            if title:
                text = f"{title} [{path_str}] {cells_str}"
            else:
                text = f"[{path_str}] {cells_str}"

            metadata = {"table_id": table_id, "path": path, "depth": len(path)}
            results.append((text, metadata))

        return results
        if not results:
            return [(title or "empty", {"table_id": table_id, "path": [], "depth": 0})]
        return results

    def get_name(self) -> str:
        return "header_path"
