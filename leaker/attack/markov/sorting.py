"""
This file implement the MarkovSorting Markov Attack.

For License information see the LICENSE file.

"""
from collections import Counter
from logging import getLogger
from typing import TypeVar, List, Any, Set, Union, Dict

import numpy as np
from bidict import bidict

from leaker.api import Extension, KeywordQueryAttack, LeakagePattern, QuerySequence
from leaker.pattern import QueryEquality
from .util import calc_stationary_dist, trans_matrix_from_seq, transform, print_stats, pancake

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class MarkovSorting(KeywordQueryAttack):
    _transform: bool
    _pancake: bool

    def __init__(self, known: Union[QuerySequence, List[List[str]]], **kwargs):
        super().__init__(known, **kwargs)
        if isinstance(known, list):
            self.__known = known
        else:
            self.__known = [known]
        self._transform = False
        self._pancake = False

    @classmethod
    def name(cls) -> str:
        return "MarkovSorting"

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

            if num_states_adv < num_states_user and not self._transform:
                continue  # skip adv knowledge again if more user states than adv states are observed
            if isinstance(u_seq, QuerySequence):
                t_mat_adv = u_seq.original_transition_matrix
                keyword_to_state = u_seq.keyword_to_state
            else:
                t_mat_adv, keyword_to_state = trans_matrix_from_seq(u_seq, num_states_adv)

            if self._transform:
                t_mat_adv, alt_map = transform(t_mat_adv)
                num_states_adv = len(t_mat_adv[0])
                # if num_states_adv < num_states_user:
                #    continue

            stationary_dist = calc_stationary_dist(t_mat_adv)
            #TODO - not yet implemented
            if self._pancake:

                alt_state_map, obs_traces, kwd_to_dummy_st = pancake(queries, t_mat_adv)
                num_states_adv = len(set(obs_traces))
                t_mat_adv, keyword_to_state = trans_matrix_from_seq(obs_traces, num_states_adv)
                cntr = Counter(obs_traces)
                recovered = ["" for q in obs_traces]
                stationary_dist = calc_stationary_dist(t_mat_adv)
                if num_states_adv < num_states_user:
                    continue

            print_stats(t_mat_adv)

            import sys
            original_stdout = sys.stdout
            with open('stats.txt', 'a+') as f:
                f.write(f"{self.name()}  stats: ")
                sys.stdout = f  # Change the standard output to the file we created.
                print_stats(t_mat_adv)
                sys.stdout = original_stdout  # Reset the standard output to its original value

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
                        if self._transform:
                            res[q] = alt_map[res[q]]
                        if self._pancake:
                            # here add the needed changes for pancake integration
                            continue
                        if res[q] in keyword_to_state.inverse.keys():
                            recovered[i] = keyword_to_state.inverse[res[q]]

        return recovered


class TransformedMarkovSorting(MarkovSorting):
    def __init__(self, known: Union[QuerySequence, List[List[str]]]):
        super().__init__(known)
        self._transform = True

    @classmethod
    def name(cls) -> str:
        return "TransformedMarkovSorting"

class PancakeMarkovSorting(MarkovSorting):
    def __init__(self, known: Union[QuerySequence, List[List[str]]]):
        super().__init__(known)
        self._pancake = True

    @classmethod
    def name(cls) -> str:
        return "PancakeMarkovSorting"