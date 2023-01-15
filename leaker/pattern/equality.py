"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import Iterable, List

from ..api import Dataset, LeakagePattern


class QueryEquality(LeakagePattern[List[int]]):
    """
    The query equality (qeq) leakage pattern leaking the equality of queries.
    """
    def leak(self, keywords: Iterable[str], dataset: Dataset) -> List[List[int]]:
        return [[q == qp for qp in keywords] for q in keywords]
