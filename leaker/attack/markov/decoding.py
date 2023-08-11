"""
This file implement the Markov Decoding  Attack.

For License information see the LICENSE file.

"""
from logging import getLogger
from typing import TypeVar, List, Any, Union

import numpy as np
from bidict import bidict
from hmmlearn import hmm

from leaker.api import Extension, KeywordQueryAttack, LeakagePattern, QuerySequence
from leaker.pattern import QueryEquality
from .util import calc_stationary_dist, trans_matrix_from_seq, print_stats
from scipy.stats import binom

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class MarkovDecoding(KeywordQueryAttack):
    __ep: float
    _binomial: bool

    def __init__(self, known: Union[QuerySequence, List[List[str]]], ep: float = 0.0005):
        super().__init__(known)
        if isinstance(known, list):
            self.__known = known
        else:
            self.__known = [known]
        self.__ep = ep
        self._binomial = False

    @classmethod
    def name(cls) -> str:
        return "MarkovDecoding"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [QueryEquality]

    def recover(self, queries: List[int]) -> List[str]:
        log.info(f"Running {self.name()} with {len(set(queries))}")
        num_states_user = len(set(queries))
        state_map = bidict({k: v for k, v in enumerate(set(queries))})
        token_sequence = [state_map.inverse[q] for q in queries]

        sample_hist = np.histogram(token_sequence, num_states_user, (0, num_states_user))[0] \
                      / float(len(token_sequence))
        unique_queries = len(sample_hist)

        recovered = ["" for q in queries]
        best_prob = -np.inf

        for u_seq in self.__known:
            if isinstance(u_seq, QuerySequence):
                num_states_adv = u_seq.num_states
            else:
                num_states_adv = len(set(u_seq))

            if num_states_adv < num_states_user :
                continue  # skip adv knowledge again if more user states than adv states are observed
            if isinstance(u_seq, QuerySequence):
                t_mat_adv = u_seq.original_transition_matrix
                keyword_to_state = u_seq.keyword_to_state
            else:
                t_mat_adv, keyword_to_state = trans_matrix_from_seq(u_seq, num_states_adv)


            stationary_dist = calc_stationary_dist(t_mat_adv)
            print_stats(t_mat_adv)

            big_o = [[None] * unique_queries for _ in range(num_states_adv)]

            hw = []

            for i in range(num_states_adv):
                cnt = 0
                for j in range(unique_queries):
                    if not self._binomial:
                        if abs(sample_hist[j] - stationary_dist[i]) <= self.__ep:
                            cnt += 1
                            big_o[i][j] = 1 - abs(sample_hist[j] - stationary_dist[i])
                        else:
                            big_o[i][j] = 0.00001
                    else:
                        big_o[i][j] = max(0.00001, binom.pmf(sample_hist[j] * len(token_sequence), len(token_sequence), stationary_dist[i]))
                hw.append(cnt)

            log.info(f"HW {len(set(queries))}: {min(hw), np.mean(hw), max(hw)}")

            big_o = np.array([np.divide(big_o[i], np.sum(big_o[i])) for i in range(num_states_adv)])

            my_hmm = hmm.MultinomialHMM(n_components=num_states_adv)
            my_hmm.n_features = unique_queries
            my_hmm.transmat_ = t_mat_adv
            my_hmm.startprob_ = np.full(num_states_adv, 1.0 / num_states_adv)
            my_hmm.emissionprob_ = big_o

            log_prob, predicted_states = my_hmm.decode(np.array([token_sequence]).T, algorithm="viterbi")

            if log_prob > best_prob:
                best_prob = log_prob
                keyword_to_state = bidict(keyword_to_state)

                recovered = []
                for s in predicted_states:
                    if s in keyword_to_state.inverse.keys():
                        recovered.append(keyword_to_state.inverse[s])
                    else:
                        recovered.append("")

        return recovered


class BinomialMarkovDecoding(MarkovDecoding):
    def __init__(self, known: Union[QuerySequence, List[List[str]]], ep: float = 0.0005):
        super().__init__(known, ep)
        self._binomial = True

    @classmethod
    def name(cls) -> str:
        return "BinomialMarkovDecoding"