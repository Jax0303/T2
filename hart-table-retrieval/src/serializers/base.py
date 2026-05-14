from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSerializer(ABC):
    @abstractmethod
    def serialize(self, table_data: dict, header_tree) -> List[Tuple[str, dict]]:
        """
        테이블을 직렬화하여 (text, metadata) 튜플의 리스트로 반환.
        - single-vector: 리스트 길이 1
        - multi-vector (HART): 리스트 길이 = 헤더 경로 수
        metadata: table_id, path, depth
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
