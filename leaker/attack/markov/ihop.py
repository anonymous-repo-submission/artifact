"""
This file implement the Markov query dependant IHOP  Attack.

For License information see the LICENSE file.

"""
import random
import numpy as np
from bidict import bidict
from logging import getLogger
from typing import TypeVar, List, Any, Dict, Union
from scipy.optimize import linear_sum_assignment as hungarian

from leaker.api import Extension, KeywordQueryAttack, LeakagePattern, QuerySequence
from leaker.pattern import QueryEquality
from .util import trans_matrix_from_seq, calc_stationary_dist

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


# Auxiliary info based function to generate [tilde(V,f,F)]
def markov_aux_and_mapping(aux, ep):
    nkw = aux['num_states']  # Number of keywords known by the adv
    markov_aux = (aux['freq_tra_mat'] + ep / nkw) / (
            1 + 2 * ep / nkw)  # Transition matrix based on keyword frequencies in the aux data
    return markov_aux


def compute_markov_obs(token_sequence, n_tokens):
    markov_obs = np.zeros((n_tokens, n_tokens))
    nquery_per_tok = np.zeros(n_tokens)

    mj = np.histogram2d(token_sequence[1:], token_sequence[:-1], bins=(range(n_tokens + 1), range(n_tokens + 1)))[0] / (
            len(token_sequence) - 1)  # we normalized by (len(token_sequence) - 1) bcs ρ(tk_j) is not absolute

    for j in range(n_tokens):
        nquery_per_tok[j] = np.sum(mj[:, j])
        if np.sum(mj[:, j]) > 0:
            markov_obs[:, j] = mj[:, j] / np.sum(mj[:, j])

    return nquery_per_tok, markov_obs


def update_coefficients(token_sequence, token_map, aux, ep):
    # Observations
    nquery_per_tok, markov_obs = compute_markov_obs(token_sequence, len(token_map))
    markov_obs_counts = markov_obs * nquery_per_tok

    # Auxiliary info
    markov_aux = markov_aux_and_mapping(aux, ep)
    stationary_dist_aux = calc_stationary_dist(markov_aux)

    def _compute_coef_matrix(free_keywords, free_tokens, fixed_keywords, fixed_tokens):
        cost_matrix = np.zeros((len(free_keywords), len(free_tokens)))

        cost_matrix -= markov_obs_counts[np.ix_(free_tokens, free_tokens)].diagonal() * np.log(
            np.array([markov_aux[np.ix_(free_keywords, free_keywords)].diagonal()]).T)

        ss_from_others = (markov_aux[np.ix_(free_keywords, free_keywords)] *
                          (np.ones((len(free_keywords), len(free_keywords))) - np.eye(len(free_keywords)))) @ \
                         stationary_dist_aux[free_keywords]

        ss_from_others = ss_from_others / (
                np.sum(stationary_dist_aux[free_keywords]) - stationary_dist_aux[free_keywords])

        counts_from_others = markov_obs_counts[np.ix_(free_tokens, free_tokens)].sum(axis=1) - markov_obs_counts[
            np.ix_(free_tokens, free_tokens)].diagonal()

        cost_matrix -= counts_from_others * np.log(np.array([ss_from_others]).T)

        for tag, kw in zip(fixed_tokens, fixed_keywords):
            cost_matrix -= markov_obs_counts[free_tokens, tag] * np.log(np.array([markov_aux[free_keywords, kw]]).T)
            cost_matrix -= markov_obs_counts[tag, free_tokens] * np.log(np.array([markov_aux[kw, free_keywords]]).T)

        return cost_matrix

    return _compute_coef_matrix


class MarkovIHOP(KeywordQueryAttack):
    __pfree: float
    __niters: int
    __aux: Dict

    def __init__(self, known: Union[QuerySequence, List[List[str]]], pfree: float = 0.1, niters: int = 10000,
                 ep: float = 1e-20):
        super().__init__(known)
        if isinstance(known, list):
            self.__known = known
        else:
            self.__known = [known]

        self.__pfree = pfree
        self.__niters = niters
        self.__ep = ep

    @classmethod
    def name(cls) -> str:
        return "MarkovIHOP"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [QueryEquality]

    def recover(self, queries: List[int]) -> List[str]:
        log.info(f"Running {self.name()}")
        state_map = bidict({k: v for k, v in enumerate(set(queries))})
        token_sequence = [state_map.inverse[q] for q in queries]

        unq_qrs = sorted(set(token_sequence), key=token_sequence.index)
        token_map = [(k, v) for k, v in enumerate(unq_qrs)]

        pct_free = self.__pfree  # the given % of free tokens
        niters = self.__niters  # The number of iteration
        ntok = len(token_map)  # nbr of unique query tokens

        recovered = ["" for q in queries]
        best_cost = np.inf

        for u_seq in self.__known:
            if isinstance(u_seq, QuerySequence):
                num_states_adv = u_seq.num_states
            else:
                num_states_adv = len(set(u_seq))

            if num_states_adv < ntok :
                continue
            if isinstance(u_seq, QuerySequence):
                t_mat_adv = u_seq.original_transition_matrix
                keyword_to_state = u_seq.keyword_to_state
            else:
                t_mat_adv, keyword_to_state = trans_matrix_from_seq(u_seq, num_states_adv)


            aux = dict()
            aux['num_states'] = num_states_adv
            aux['freq_tra_mat'] = t_mat_adv.T


            compute_coef_matrix = update_coefficients(token_sequence, token_map, aux, self.__ep)

            known_tokens, known_keywords = [], []
            unknown_tokens = [i for i in range(ntok) if i not in known_tokens]
            unknown_keywords = [i for i in range(ntok) if i not in known_keywords]

            # First mapping:
            c_matrix_original = compute_coef_matrix(free_keywords=unknown_keywords, free_tokens=unknown_tokens,
                                                    fixed_keywords=known_keywords, fixed_tokens=known_tokens)

            try:
                row_ind, col_ind = hungarian(c_matrix_original)

                predictions = {token: rep for token, rep in zip(known_tokens, known_keywords)}

                for j, i in zip(col_ind, row_ind):
                    predictions[unknown_tokens[j]] = unknown_keywords[i]

                if len(predictions) != len(unknown_tokens):
                    log.info(
                        f"we have a rectangular cost matrix with {1 - len(predictions) / len(unknown_tokens)} "
                        f"% of unmapped tokens.")
                    unpredicted_tok = [token for token in unknown_tokens if token not in predictions.keys()]
                    unpredicted_key = [keyword for keyword in unknown_keywords if keyword not in predictions.values()]
                    if len(unpredicted_key) != 0:
                        for _ in unpredicted_tok:
                            predictions[_] = random.choice(unpredicted_key)
                    else:

                        unknown_tokens = [token for token in unknown_tokens if token not in unpredicted_tok]
                        token_sequence = [token for token in token_sequence if token in unknown_tokens]
                # Iterate using co-occurrence:
                n_free = int(pct_free * len(unknown_tokens))
                if not n_free > 1:
                    log.warning(f"n_free too low.")
                    continue

                for k in range(niters):
                    random_unknown_tokens = list(np.random.permutation(unknown_tokens))
                    new_free_tokens = random_unknown_tokens[:n_free]
                    new_fixed_tokens = random_unknown_tokens[n_free:] + known_tokens

                    # Free tokens that have not been assigned by hungarian in case of rectangular cost matrix
                    new_fixed_tokens = [token for token in new_fixed_tokens if token in predictions.keys()]
                    new_free_tokens = [token for token in new_free_tokens if token not in new_fixed_tokens]

                    new_fixed_keywords = [predictions[token] for token in new_fixed_tokens]
                    new_free_keywords = [rep for rep in unknown_keywords if rep not in new_fixed_keywords]

                    c_matrix = compute_coef_matrix(new_free_keywords, new_free_tokens, new_fixed_keywords,
                                                   new_fixed_tokens)

                    row_ind, col_ind = hungarian(c_matrix)
                    cost = c_matrix[row_ind, col_ind].sum()
                    for j, i in zip(col_ind, row_ind):
                        predictions[new_free_tokens[j]] = new_free_keywords[i]
            except ValueError:
                log.warning(f"Invalid values in cost matrix.")
                continue

            if cost < best_cost:
                predicted_states = [predictions[token] for token in token_sequence]
                keyword_to_state = bidict(keyword_to_state)
                best_cost = cost

                recovered = []
                for s in predicted_states:
                    if s in keyword_to_state.inverse.keys():
                        recovered.append(keyword_to_state.inverse[s])
                    else:
                        recovered.append("")

        return recovered


