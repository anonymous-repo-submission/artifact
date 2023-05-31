"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Generic, TypeVar, Union, Tuple, Optional

from .range_database import RangeDatabase
from .dataset import Dataset

T = TypeVar("T", covariant=True)


class LeakagePattern(ABC, Generic[T]):
    """A leakage pattern, that is, a function from a sequence of queries or values to some specific leakage type."""

    @abstractmethod
    def leak(self, queries: Union[Iterable[int], Iterable[str], Iterable[Tuple[int, int]]],
             dataset: Optional[Union[Dataset, RangeDatabase]] = None) -> List[T]:
        """
        Calculates the leakage on the given data set and queries.

        Parameters
        ----------
        queries : Union[Iterable[int], Iterable[str], Iterable[Tuple[int, int]]]
            the values or queries to leak on
        dataset : Optional[Union[Dataset, RangeDatabase]]
            the data set or range DB to calculate the leakage on

        Returns
        -------
        leak : List[T]
            the leakage
        """
        raise NotImplementedError

    def __call__(self, queries: Union[Iterable[int], Iterable[str], Iterable[Tuple[int, int]]],
                 dataset: Optional[Union[Dataset, RangeDatabase]] = None) -> List[T]:
        return self.leak(queries, dataset)
