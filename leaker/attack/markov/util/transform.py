"""
This file implements Markov Chain transformations and Pancake as countermeasures.
the Pancake countermeasure was implemented by Simon Oya in his paper IHOP
https://github.com/simon-oya/USENIX22-ihop-code/blob/master/defense.py

For License information see the LICENSE file.

"""

import numpy as np
from typing import Dict
from bidict import bidict
from sklearn.preprocessing import normalize


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
