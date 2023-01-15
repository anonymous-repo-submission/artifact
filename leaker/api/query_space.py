"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber, Abdelkarim Kati

"""
from abc import ABC, abstractmethod
from logging import getLogger
from random import sample
from typing import Collection, Set, List, Iterator, Tuple, Any, Dict, Union, Optional

import numpy as np

from .constants import Selectivity
from .dataset import Dataset, KeywordQueryLog
from .range_database import RangeDatabase
from .attack import QuerySequence
from ..util import DummyIdDict

log = getLogger(__name__)


class QuerySpace(ABC, Collection):
    """
    A query space can be used to select a query sequence of arbitrary length (up to the size of the query space).

    It can be used to select multiple instances of queries.
    """

    @abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def select(self, n: int) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, entry) -> bool:
        raise NotImplementedError


class KeywordQuerySpace(QuerySpace):
    __query_log: KeywordQueryLog
    __space: Union[List[Set[Tuple[str, int]]], List[QuerySequence]]
    __allow_repetition: bool
    __is_query_seq: bool  # whether just a query sequence is the source of the query space
    __use_qlog: bool
    __sample_queries: bool  # whether to sample queries or just attack the sequence

    def __init__(self, *args):
        self.__is_query_seq = isinstance(args[0], int)
        if self.__is_query_seq:
            self.query_seq_space(*args)
        else:
            self.dataset_space(*args)

    def query_seq_space(self, size: int, query_log: KeywordQueryLog = None, sample_queries: bool = True):
        """
        Creates and populates the query space using solely information from the query sequence.

        Parameters
        ----------
        size : int
            the desired size of the query space
        query_log : QeryLog
            the query log of users
        sample_queries: bool
            whether to sample queries or just attack the sequence
        """
        self.__space: List[QuerySequence] = []
        self.__allow_repetition = True
        self.__use_qlog: bool = query_log is not None
        self.__query_log = query_log
        self.__sample_queries = sample_queries

        if self.__sample_queries:
            for i, query_seq in enumerate(self._query_candidates(size, query_log)):
                self.__space.append(query_seq)
        else:
            if not self.__use_qlog:
                log.warning(f"Non-Sampling option requires a query log.")
            for i in query_log.user_ids():
                queries = query_log.keywords_list(i, True)
                qseq = QuerySequence(transition_matrix=[],
                                     num_states=len(set(queries)), query_list=queries,
                                     keyword_to_state={kw: s for s, kw in enumerate(set(queries))},
                                     alt_state_map=DummyIdDict(), original_transition_matrix=[])
                self.__space.append(qseq)  # We don't sample & attack the actual sequence

    def resample(self) -> None:
        """Resample distribution"""
        if self.__is_query_seq:
            num_states = self.__space[0].original_transition_matrix.shape[0]

            self.query_seq_space(num_states, self.__query_log, self.__sample_queries)

    def dataset_space(self, full: Dataset, known: Dataset, selectivity: Selectivity, size: int,
                      query_log: KeywordQueryLog = None,
                      allow_repetition: bool = False):
        """
        Creates and populates the query space using information from the database.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        selectivity : Selectivity
            the selectivity of the keywords to use
        size : int
            the desired size of the query space
        query_log : QeryLog
            the query log of users
        allow_repetition : bool
            whether repetitions are allowed when drawing query sequences
        """
        self.__query_log = None
        self.__space: List[Set[Tuple[str, int]]] = []
        self.__use_qlog: bool = query_log is not None
        self.__sample_queries = False

        for i, candidate_keywords in enumerate(self._candidates(full, known, query_log)):
            if len(candidate_keywords) < size:
                log.warning(f"Set of candidate keywords with length {len(candidate_keywords)} at position {i} smaller "
                            f"than configured query space size of {size}. Requested selectivity ignored.")
                self.__space.append(candidate_keywords)
                continue
            if selectivity == Selectivity.High:
                self.__space.append(set(sorted(candidate_keywords, key=lambda item: full.selectivity(item[0]),
                                               reverse=True)[:size]))
            elif selectivity == Selectivity.Low:
                self.__space.append(set(sorted(candidate_keywords, key=lambda item: full.selectivity(item[0]))[:size]))
            elif selectivity == Selectivity.PseudoLow:
                self.__space.append(set(sorted(filter(lambda item: 10 <= full.selectivity(item[0]), candidate_keywords),
                                               key=lambda item: full.selectivity(item[0]))[:size]))
            elif selectivity == Selectivity.Independent:
                self.__space.append(set(sample(population=candidate_keywords, k=size)))

        self.__allow_repetition = allow_repetition

    @classmethod
    def create(cls, full: Dataset, known: Dataset, selectivity: Selectivity, size: int, query_log: KeywordQueryLog,
               allow_repetition: bool = False) -> 'KeywordQuerySpace':
        """
        Creates a query space with data of a dataset.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        selectivity : Selectivity
            the selectivity of the keywords to use
        size : int
            the desired size of the query space
        query_log : QeryLog
            the query log of users
        allow_repetition : bool
            whether repetitions are allowed when drawing query sequences

        Returns
        -------
        create : QuerySpace
            the created query space
        """
        return cls(full, known, selectivity, size, query_log, allow_repetition)

    def _get_space(self) -> Iterator[Set[Tuple[str, int]]]:
        yield from self.__space

    def get_query_log(self) -> KeywordQueryLog:
        return self.__query_log

    def get_full_sequence(self, user_id: Optional[int] = None) -> QuerySequence:
        if user_id is None:
            return self.__space[0]
        else:
            return self.__space[user_id]

    def sample_queries(self) -> bool:
        return self.__sample_queries

    def select(self, n: int, from_original=False) -> Iterator[List[str]]:
        """
        Selects a query sequence of the desired length.

        Parameters
        ----------
        n : int
            the length of the query sequence
        from_original : bool
            If the query space was transformed, but sampling should occur from the original

        Returns
        -------
        select : Iterator[List[str]]
            the selected queries per query space
        """
        # choice is expecting sequences with the same ordering here, hence the list(...)
        length = n
        for i, space in enumerate(self.__space):
            if len(space) == 0:
                log.warning(f"Encountered empty space at position {i + 1} of {len(self.__space)}. "
                            f"Less evaluations than anticipated are performed.")
                continue
            if len(space) < n and not self.__allow_repetition:
                log.warning(
                    f"Encountered insufficiently large query space with size {len(space)} at position {i + 1} of"
                    f" {len(self.__space)}.")
                length = len(space)

            if self.__is_query_seq:  # Have to stay in the same order
                if self.sample_queries():
                    seq_sample = [None] * length
                    '''Start in a "non-ending" state'''
                    if not from_original:
                        trans_mat = space.transition_matrix
                    else:
                        trans_mat = space.original_transition_matrix
                    num_states = trans_mat.shape[0]
                    non_zero = np.nonzero(np.sum(trans_mat, axis=1) - trans_mat.diagonal())[0]
                    ix = np.random.choice(len(non_zero), 1)[0]
                    seq_sample[0] = non_zero[ix]
                    for j in range(1, length):
                        probs = trans_mat[seq_sample[j - 1]]
                        seq_sample[j] = np.random.choice(num_states, 1, p=probs)[0]

                    self.__space[i] = QuerySequence(transition_matrix=space.transition_matrix,
                                                    num_states=space.num_states,
                                                    query_list=seq_sample, keyword_to_state=space.keyword_to_state,
                                                    alt_state_map=space.alt_state_map, original_transition_matrix=
                                                    space.original_transition_matrix)
                    yield seq_sample
                else:
                    seq = [space.keyword_to_state[q] for q in space.query_list]

                    yield seq

            else:
                space = list(space)
                p = np.array(list(map(lambda item: float(item[1]), space)))
                p /= p.sum()
                yield np.random.choice(list(map(lambda item: item[0], space)), length, p=p,
                                       replace=self.__allow_repetition)

    def __set__(self, user_id: int, qseq: [QuerySequence]):
        self.__space[user_id] = qseq

    def __len__(self) -> int:
        return len(self.__space)

    def __iter__(self) -> Iterator[Iterator[str]]:
        for space in self.__space:
            yield from map(lambda item: item[0], iter(space))

    def __contains__(self, keyword: object) -> bool:
        return any([keyword in dict(space) for space in self.__space])

    @classmethod
    def is_multi_user(cls) -> bool:
        """Return True if multiple users are considered, False if a single user or queries aggregated from all users
        are considered."""
        return False

    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        """
        Returns one or multiple sets of keyword candidates for populating the query space. Multiple sets can be used to,
        e.g., yield queries of individual users. Keyword candidates consist of the keyword and their frequency/weights
        for query selection.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        query_log : QeryLog
            the query log of users
        """
        raise NotImplementedError

    @classmethod
    def _query_candidates(cls, size: int, query_log: KeywordQueryLog) -> Iterator[QuerySequence]:
        """
        Returns one or multiple sets of keyword candidates for populating the query space. Multiple sets can be used to,
        e.g., yield queries of individual users. Keyword candidates consist of the keyword and their frequency/weights
        for query selection. This method solely uses query sequence information

        Parameters
        ----------
        queries: QuerySequence
        query_log : QeryLog
            the query log of users
        """
        return cls._candidates(None, None, query_log)


class RangeQuerySpace(QuerySpace):
    """
    A class to represent a QuerySpace for range queries.
    :param n: An upper bound on the number of queries to return. Returns all if = -1.
    The query space is re-created after each sampling if resample is set True
    """
    __queries: List[List[Tuple[int, int]]]
    __db: RangeDatabase
    __allow_repetition: bool
    __allow_empty: bool
    __kwargs: Dict[str, Any]
    __n: int
    __resample: bool

    def __init__(self, db: RangeDatabase, n: int = -1, allow_repetition: bool = True, allow_empty: bool = True,
                 resample: bool = True, **kwargs):
        self.__allow_repetition = allow_repetition
        self.__allow_empty = allow_empty
        self.__kwargs = kwargs
        self.__db = db
        self.__n = n
        self.__resample = resample

        self.__queries = self.gen_queries(self.__db, self.__n, self.__allow_repetition, self.__allow_empty,
                                          **self.__kwargs)

    @classmethod
    def create(cls, db: RangeDatabase, allow_repetition: bool = True, allow_empty: bool = True, **kwargs) \
            -> 'RangeQuerySpace':
        return cls(db, allow_repetition, allow_empty, **kwargs)

    def get_size(self) -> int:
        """
        Return the number of queries.
        :return: Number of queries
        """
        return self.__len__()

    def select(self, n: int = -1) -> Iterator[List[Tuple[int, int]]]:
        """
        Return n queries from the query space for each of its users.
        :param n: The number of queries to return. Returns all if = -1.
        :return: The queries
        """
        for queries in self.__queries:
            if n == -1 or n >= len(queries):
                res = queries
            else:
                res = sample(population=queries, k=n)
            yield res

        if self.__resample:
            self.__queries = self.gen_queries(self.__db, self.__n, self.__allow_repetition, self.__allow_empty,
                                              **self.__kwargs)

    @classmethod
    @abstractmethod
    def gen_queries(cls, db: RangeDatabase, n: int, allow_repetition: bool = False, allow_empty: bool = False,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """This implements the actual query space, creating a sequence of n queries according to a distribution,
        possibly sampled for multiple users"""
        raise NotImplementedError

    def __len__(self):
        return len(self.__queries)

    def __iter__(self):
        return iter(self.__queries)

    def __contains__(self, item):
        return item in self.__queries
