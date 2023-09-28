"""
This file implement the MarkovStationary Markov Attack.

For License information see the LICENSE file.

"""
from collections import Counter
from logging import getLogger
from typing import TypeVar, List, Any, Set, Union, Dict

import numpy as np
from bidict import bidict

from leaker.api import Extension, KeywordQueryAttack, LeakagePattern, QuerySequence
from leaker.pattern import QueryEquality
from .util import calc_stationary_dist, trans_matrix_from_seq

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class MarkovStationary(KeywordQueryAttack):

    def __init__(self, known: Union[QuerySequence, List[List[str]]], **kwargs):
        super().__init__(known, **kwargs)
        if isinstance(known, list):
            self.__known = known
        else:
            self.__known = [known]

    @classmethod
    def name(cls) -> str:
        return "MarkovStationary"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [QueryEquality]

    def recover(self, queries: List[int]) -> List[str]:
        log.info(f"Running {self.name()}")
        cntr = Counter(queries)
        num_states_user = len(cntr.keys())

        recovered = ["" for q in queries]
        best_dist = np.inf

        for u_seq in self.__known:
            if isinstance(u_seq, QuerySequence):
                num_states_adv = u_seq.num_states
            else:
                num_states_adv = len(set(u_seq))

            if num_states_adv < num_states_user:
                continue  # skip adv knowledge again if more user states than adv states are observed
            if isinstance(u_seq, QuerySequence):
                t_mat_adv = u_seq.original_transition_matrix
                keyword_to_state = u_seq.keyword_to_state
            else:
                t_mat_adv, keyword_to_state = trans_matrix_from_seq(u_seq, num_states_adv)


            stationary_dist = calc_stationary_dist(t_mat_adv)

            res: Dict[int, int] = {}
            big_s: Set[int] = set()
            dist = 0
            for i, freq in cntr.items():
                res_set = np.argsort(np.abs(freq / len(queries) - stationary_dist))
                for r in res_set:
                    if r not in big_s:
                        res[i] = r
                        big_s.add(r)
                        dist += abs(freq / len(queries) - stationary_dist[r])
                        break

            if dist < best_dist:
                best_dist = dist

                keyword_to_state = bidict(keyword_to_state)

                recovered = ["" for q in queries]
                for i, q in enumerate(queries):
                    if q in res.keys():
                        if res[q] in keyword_to_state.inverse.keys():
                            recovered[i] = keyword_to_state.inverse[res[q]]

        return recovered
