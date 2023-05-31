"""
This file implements Markov Chain transformations and Pancake as countermeasures.
the Pancake countermeasure was implemented by Simon Oya in his paper IHOP
https://github.com/simon-oya/USENIX22-ihop-code/blob/master/defense.py

For License information see the LICENSE file.

"""

from functools import reduce
import numpy as np
from typing import List, Dict, Tuple
from bidict import bidict
from sklearn.preprocessing import normalize
from ..util import calc_stationary_dist
import matplotlib.pyplot as plt


def pancake_params(nkw, transition_matrix):
    replicas_per_kw = np.ceil(transition_matrix * nkw)
    replicas_per_kw = np.append(replicas_per_kw, 2 * nkw - np.sum(replicas_per_kw)).astype(np.int64)

    prob_reals = np.append(transition_matrix, 0)
    if replicas_per_kw[-1] == 0:
        prob_dummies = np.append(replicas_per_kw[:-1] / nkw - prob_reals[:-1], 0)
    else:
        prob_dummies = replicas_per_kw / nkw - prob_reals

    return prob_reals, prob_dummies, replicas_per_kw


def pancake(query_seq, transition_matrix, keyword_to_state):
    """Compute the parameter of Pancake given stationary distribution matrix, returns replicated/dummy state and
    updated mappings"""

    num_states_adv = len(transition_matrix[0])
    stationary_dist = calc_stationary_dist(transition_matrix)
    prob_reals, prob_dummies, replicas_per_kw = pancake_params(num_states_adv, stationary_dist)

    obs_traces = []
    permutation = np.random.permutation(2 * num_states_adv)
    aux = [0] + list(np.cumsum(replicas_per_kw, dtype=int))
    kw_id_to_replica = [tuple(permutation[aux[i]: aux[i + 1]]) for i in range(len(aux) - 1)]

    nq = len(query_seq)
    perm = np.random.permutation(3 * nq)
    separation = np.random.binomial(3 * nq, 0.5)
    indices_real_slots, indices_dummy_slots = perm[:separation], perm[separation:]
    indices_real_slots.sort()
    indices_for_each_true_message = []
    real_slots_copy = indices_real_slots.copy()
    for i in range(0, 3 * nq, 3):
        try:
            index = next(filter(lambda x: real_slots_copy[x] >= i, range(len(real_slots_copy))))
        except StopIteration:
            break
        indices_for_each_true_message.append(real_slots_copy[index])
        real_slots_copy = real_slots_copy[(index + 1):]

    trace_no_replicas = -np.ones(3 * nq, dtype=int)
    trace_no_replicas[indices_dummy_slots] = np.random.choice(num_states_adv + 1, len(indices_dummy_slots),
                                                              replace=True, p=prob_dummies)
    trace_no_replicas[indices_real_slots] = np.random.choice(num_states_adv + 1, len(indices_real_slots),
                                                             replace=True,
                                                             p=prob_reals)
    trace_no_replicas[indices_for_each_true_message] = query_seq[:len(indices_for_each_true_message)]

    for kw_id in trace_no_replicas:

        obs_traces.append(np.random.choice(kw_id_to_replica[kw_id]))  # leakage containing dummies & replication

    alt_state_map = {kw: s for s, kwl in enumerate(kw_id_to_replica[:-1]) for kw in
                     kwl}  # Mapping all replicated states into their original state s.

    for s, kw in enumerate(kw_id_to_replica[-1]):  # For newly created dummies, a new plaintext state is created
        alt_state_map[kw] = num_states_adv + s

    '''Add these new plaintext states to the keyword to state map'''
    kwd_to_dummy_st = {f"dummy{s}": num_states_adv + s for s, _ in enumerate(kw_id_to_replica[-1])}

    for kw, st in keyword_to_state.items():  # We need to include the previous map as well
        kwd_to_dummy_st[kw] = st

    return alt_state_map, obs_traces, kwd_to_dummy_st


def print_stats(t_mat):
    num_states = t_mat.shape[0]
    no_not_changed = []
    hw = []
    mins = []
    means = []
    maxs = []
    for i in range(num_states):

        try:
            no_not_changed.append(len(np.where(np.logical_and(t_mat[i] > 0.0, t_mat[i] < 0.2))[0]))
        except ValueError:
            pass
        try:
            mins.append(np.min(t_mat[i][np.nonzero(t_mat[i])]))
        except ValueError:
            pass
        try:
            hw.append(len(np.nonzero(t_mat[i])[0]))
        except ValueError:
            pass
        means.append(np.mean(t_mat[i]))
        maxs.append(np.max(t_mat[i]))

    print(
        f"({np.mean(mins)}, {np.mean(means)}, {np.mean(maxs)}). Avg no. not changed: {np.mean(no_not_changed)}. Avg. HW: {np.mean(hw)}")


def transform(transition_mat: np.ndarray, c: float = 0.2) -> Tuple[np.ndarray, Dict[int, int]]:
    """Transforms a given transition matrix, returns new matrix and Dictionary of new state to original state"""
    print(f"Before transform:")
    print_stats(transition_mat)
    num_states = transition_mat.shape[0]
    alt_states_cnt = 0  # the total number of replicated states.
    alt_states: Dict[int, Dict[int, List[int]]] = dict()
    state_map: Dict[int, int] = dict()

    for s in range(num_states):
        alt_states[s] = dict()
        e_in = np.nonzero(transition_mat.T[s])[0]  # incoming edges list for state s.


        eta = np.max(transition_mat.T[s])  # maximum edge value for all incoming edges into state s.


        v = int(np.ceil(eta / c))  # number of generated duplicates for state s.
        if v > 0:
            alt_states[s][-1] = set()  # Empty set, dummy value reserved for case v=0
            new_states = np.array(range(alt_states_cnt, alt_states_cnt + v))
            for new_state in new_states:
                state_map[new_state] = s  # Mapping all new replicated states into their original state s.
            alt_states_cnt += v

            for e in e_in:
                eta_p = transition_mat.T[s, e]  # for incoming edge e of the list of incoming edges e_in
                v_p = int(np.ceil(eta_p / c))  # compute the number of duplicates needed for this edges based on cte c.
                alt_states[s][e] = list(np.random.choice(new_states, v_p, replace=False))

        else:  # No incoming edges - keep state as is -> alternatives of state s will be alt_states_cnt
            new_states = {alt_states_cnt}
            state_map[next(iter(new_states))] = s
            alt_states_cnt += 1
            alt_states[s][-1] = new_states  # mark this fact via entry -1

    # Build new transition matrix
    t_mat = np.zeros((alt_states_cnt, alt_states_cnt))

    for s in alt_states:
        for e in alt_states[s]:
            if e == -1:
                continue
            alternatives = reduce(lambda x, y: x.union(y), alt_states[e].values(), set())

            for a in alternatives:
                if len(alt_states[s][e]) == 1:
                    o_v = transition_mat[e, s]
                    for k in alt_states[s].values():
                        if isinstance(k, set):
                            continue
                        t_mat[a, k] = o_v
                else:
                    for a_p in alt_states[s][e]:
                        t_mat[a, a_p] = c

    perm = np.random.permutation(alt_states_cnt)

    final_t_mat = np.zeros((alt_states_cnt, alt_states_cnt))
    final_state_map: Dict[int, int] = dict()
    for i in range(alt_states_cnt):
        final_state_map[perm[i]] = state_map[i]
        for j in range(alt_states_cnt):
            final_t_mat[perm[i], perm[j]] = t_mat[i, j]

    final_t_mat = np.array([np.array(final_t_mat[i]) for i in range(alt_states_cnt)])

    trans_mat = normalize(final_t_mat, axis=1, norm='l1')  # Bc. rounding error
    """
    # plot the stats for any trans matrix to print the stationary dist:
    # I put it here, so that's why I had the c in there. But we can remove c and print on any matrix If you want,
    # I should make it a method in util/debug and push it.
    stationary_dist = calc_stationary_dist(trans_mat)
    _ = plt.bar(range(alt_states_cnt), sorted(stationary_dist, reverse=True))
    plt.title(f"Stat dist after transform with c={c}")
    plt.show()
    plt.clf()
    """
    return trans_mat, final_state_map


def remove_all_except_keywords(transition_matrix, keyword_to_state, keywords_to_keep):
    """Take a transition matrix and remove all nodes that do not occur in keywords_to_keep
    (keywords that appear in the training set), then return a new trained and normalized transition matrix
     and new mapping from keywords to state taking into account the deleted nodes. """

    new_keyword_to_state: Dict[str, int] = dict()
    new_state_to_old_state: Dict[int, int] = dict()
    i = 0
    for kw, st in keyword_to_state.items():
        if kw in keywords_to_keep:
            new_keyword_to_state[kw] = i
            new_state_to_old_state[i] = st
            i += 1

    new_state_count = i
    tmp_t_mat = np.zeros((new_state_count, new_state_count))
    for i in range(new_state_count):
        for j in range(new_state_count):
            tmp_t_mat[i, j] = transition_matrix[new_state_to_old_state[i]][new_state_to_old_state[j]]

    tmp_t_mat = np.array([np.array(tmp_t_mat[i]) for i in range(new_state_count)])

    new_keyword_to_state = bidict(new_keyword_to_state)

    # We might have created rows with all cells = 0 => remove these states until this condition does not hold again.
    proceed = True
    while proceed:
        proceed = False
        final_keyword_to_state = dict()
        final_state_to_old_state = dict()
        k = 0
        for i in range(new_state_count):
            if not np.sum(tmp_t_mat[i]) == 0.0:
                final_keyword_to_state[new_keyword_to_state.inverse[i]] = k
                final_state_to_old_state[k] = i
                k += 1
            else:
                proceed = True

        new_state_count = k
        final_t_mat = np.zeros((new_state_count, new_state_count))
        for i in range(new_state_count):
            for j in range(new_state_count):
                final_t_mat[i, j] = tmp_t_mat[final_state_to_old_state[i]][final_state_to_old_state[j]]

        final_t_mat = np.array([np.array(final_t_mat[i]) for i in range(new_state_count)])
        tmp_t_mat = final_t_mat
        new_keyword_to_state = bidict(final_keyword_to_state)

    trans_mat = normalize(final_t_mat, axis=1, norm='l1')  # L1 norm in a vector is the abs(value_i)/sum(values)
    return trans_mat, final_keyword_to_state
