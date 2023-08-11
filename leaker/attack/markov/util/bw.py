#!/usr/bin/env python2.7

"""
This file interfaces the Baum Welch Algorithm.

For License information see the LICENSE file.

"""
import os
from tempfile import TemporaryDirectory
import hidden_markov
import numpy as np
import random

# NOTE: Documentation for the hidden_markov library can be found at https://hidden-markov.readthedocs.io/en/latest/
# it is surprisingly difficult to find via google :)

# if you want to visualize the Baum Welch generated hmm, include this library
# from model_viz import visualize

# function baum_welch_from_file is a wrapper function that allows the user to run the Baum Welch algorithm on
# a sequence stored in a .txt file and not passed directly as a list. It first reads the sequence
# from the file and formats it so that baum_welch can be called on the sequence.
# NOTE: this function is not used in the experiment, but can be helpful for running the Baum Welch
# algorithm on the same sequence to see how different parameters affect the algorithm
from bidict import bidict
from sklearn.preprocessing import normalize

from leaker.api.constants import CACHE_DIRECTORY


def baum_welch_from_file(file_name, dir_name, prefix, em_path, start_path, trans_path):
    # 1. read observation txt file into a list
    f = open(file_name, 'r')
    samples_list = f.read().split('\n')
    samples_list = filter(None, samples_list)  # remove blank strings from the list (ie at the end)
    total_obs = []
    obs_seq = []
    for sample in samples_list:
        fi = open(prefix + sample, 'r')
        obs = fi.read().split('\n')
        obs = filter(None, obs)
        total_obs.append(list(obs))
        fi.close()
    f.close()
    t, unique_obs = baum_welch(total_obs, dir_name, em_path, start_path, trans_path)
    return t, unique_obs


# function baum_welch runs the Baum Welch algorithm on a passed sequence or list of sequences
# the user can choose to provide an emissions probability matrix (probability that observation i
# will be seen from state j), a transition probability matrix (probability that state i will follow
# state j), or a start probability matrix (probability that the markov chain will begin at state i) that
# will seed the Baum Welch algorithm. If these parameters are not provided, random matrices will be created
# Parameters:
#   samples_list: a list of sequences (ex: a list of observed leakage sequences)
#   dir_name: a path to a directory where the results of the Baum Welch algorithm will be written
#   em_path: path to a file containing the emissions probability matrix. Must be written in a form that can
#           automatically interpreted as a np.matrix (ie 1 0; .5 .5)
#   start_path: path to a file containing the start probability matrix. Must be written in a form that can
#           automatically interpreted as a np.matrix (ie 0 0 1 0)
#   trans_path: path to a file containing the transition probability matrix. Must be written in a form that can
#           automatically interpreted as a np.matrix (ie 1 0; .5 .5)
#   em_identity: boolean value to determine if an identity matrix should be used for the emissions' matrix. If True and
#                an em_path is provided, the em_path will not be used and the identity matrix will be used instead
#   num_states: the value of num_states will be used to determine the number of states the hmm will be initialized with
#               if non-positive, the number of unique observations in samples_list will be used
def baum_welch(samples_list, dir_name, em_path=None, start_path=None, trans_path=None, em_identity=False,
               num_iter=10000, true_num_states=-1):
    total_obs = []
    obs_seq = []
    # the baum welch algorithm used requires that the observed sequences (obs_seq) be formatted a specific way - specifically
    # as a list of tuples where each tuple is one sequence of observations.
    # total_obs is used to determine the number of unique values in samples_list. I am currently using the numbers
    # of unique values as the number of states in the markov model because we are working under the
    # assumption that each leaked observation can only come from one plaintext query
    for sample in samples_list:
        total_obs.extend(list(sample))
        obs_seq.append(tuple(sample))

    unique_obs = list(set(total_obs))

    if true_num_states < 0:
        num_states = len(unique_obs)
    else:
        num_states = true_num_states


    # load the emissions, transition, and start probability matrices, or generate random
    # matrices if files are not included

    if em_path:
        f = open(em_path, 'r')
        emission_prob = np.matrix(f.read())
        f.close()
    else:  # create a random emissions matrix
        emission_prob = np.zeros((num_states, len(unique_obs)))
        for i in range(0, num_states):
            em_prob = []
            em_sum = 0
            for j in range(0, len(unique_obs)):
                em_prob.append(random.random())
                em_sum += em_prob[j]
            em_prob[:] = [prob / em_sum for prob in em_prob]  # normalize so each row sums to 1
            emission_prob[i] = em_prob

    if start_path:
        f = open(start_path, 'r')
        start_prob = np.matrix(f.read())
        f.close()
    else:  # create a random start matrix
        start_prob = []
        start_sum = 0
        for i in range(0, num_states):
            start_prob.append(random.random())
            start_sum += start_prob[i]
        start_prob[:] = [prob / start_sum for prob in start_prob]  # normalize so probabilities sum to 1

    if trans_path:
        f = open(trans_path, 'r')
        transition_prob = np.matrix(f.read())
        f.close()
    else:  # create a random transition matrix
        transition_prob = np.zeros((num_states, num_states))
        for i in range(0, num_states):
            t_prob = []
            t_sum = 0
            for j in range(0, num_states):
                t_prob.append(random.random())
                t_sum += t_prob[j]
            t_prob[:] = [prob / t_sum for prob in t_prob]  # normalize so each row sumes to 1
            transition_prob[i] = t_prob

    start_prob = np.matrix(start_prob)
    transition_prob = np.matrix(transition_prob)
    emission_prob = np.matrix(emission_prob)
    if em_identity:
        # override the created emissions matrix with an identity matrix
        emission_prob = np.matrix(np.identity(num_states))

    # initialize the hmm (does not run BW yet) - the library requires that the hmm be seeded/initialized
    # with a list of possible states, possible observations, and some start, transition, and emissions
    # probability matrices

    if true_num_states > 0:
        states = list(range(1, true_num_states + 1))
    else:
        states = unique_obs

    model = hidden_markov.hmm(states, states, start_prob, transition_prob, emission_prob)

    times_seen = [1 for i in range(len(samples_list))]
    # times_seen indicated how many times a specific sequence in samples_list has occurred
    # if you want to weight a specific observed sequence more than others, its corresponding
    # spot in times_seen can be given a higher value

    # run the Baum Welch algorithm using the initilizded hmm as the start matrix
    e, t, s = model.train_hmm(obs_seq, num_iter, times_seen)
    # save model to file
    # emissions matrix
    e_file = open(dir_name + '/emissions_matrix.txt', 'w')
    e_iter = e.getA()
    for i in range(0, e_iter.shape[0]):
        # print row
        for val in e_iter[i]:
            e_file.write(str(val) + ' ')
        if i < e_iter.shape[0] - 1:
            e_file.write(';')
    e_file.close()

    # transition matrix
    t_file = open(dir_name + '/transition_matrix.txt', 'w')
    t_iter = t.getA()
    for i in range(0, t_iter.shape[0]):
        for val in t_iter[i]:
            t_file.write(str(val) + ' ')
        if i < t_iter.shape[0] - 1:
            t_file.write('\n')
    t_file.close()

    # start matrix
    s_file = open(dir_name + '/start_matrix.txt', 'w')
    s_iter = s.getA()
    for i in range(0, s_iter.shape[0]):
        # print row
        for val in s_iter[i]:
            s_file.write(str(val) + ' ')
        if i < s_iter.shape[0] - 1:
            s_file.write('\n')
    s_file.close()

    # if you want to see the markov model generated by the algorithm, uncomment this
    # works best for small models (about 4 states)
    # change the second parameter to states, if using a differnt number of states
    # than unique_obs
    # visualize(unique_obs, unique_obs, e.getA(), t.getA(), dir_name)
    return t, unique_obs


def trans_matrix_from_seq(keyword_list, n, keyword_to_state=None):
    if keyword_to_state is None:
        keyword_to_state = {}
        max_state = 0
    elif len(keyword_to_state.keys()) == 0:
        max_state = 0
    else:
        max_state = max(keyword_to_state.values()) + 1

    unique_keywords = list(set(keyword_list))

    # count is the number of keywords
    size = min(n, len(unique_keywords))

    samp_tmp = random.sample(range(len(unique_keywords)), size)  # size random keywords

    samp = []
    for i in range(len(samp_tmp)):
        samp.append(unique_keywords[samp_tmp[i]])

    # Map keywords to states
    i = 0
    for j in range(len(samp)):
        if samp[j] not in keyword_to_state:
            keyword_to_state[samp[j]] = max_state + i
            i += 1

    sequence = []
    for keyword in keyword_list:
        if keyword in samp:
            sequence.append(keyword_to_state[keyword])

    state_map = bidict({k: v for k, v in enumerate(set(sequence))})

    with TemporaryDirectory(dir=CACHE_DIRECTORY) as tmp_dir:
        os.system(f"mkdir {tmp_dir}/bw_r")
        os.system(f"touch {tmp_dir}/em.txt")

        # run baum welch on the sequence
        t_matrix, state_order = baum_welch([[state_map.inverse[s] for s in sequence]], f'{tmp_dir}/bw_r', em_path=f'{tmp_dir}/em.txt',
                                           start_path=None, trans_path=None, em_identity=True, true_num_states=-1)
    t_matrix[np.isnan(t_matrix)] = 0

    tmp_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(t_matrix.shape[0]):
        if np.sum(t_matrix[i]) != 0.0:
            for j in range(t_matrix.shape[0]):
                tmp_matrix[state_map[i]][state_map[j]] = t_matrix[i, j]

    tmp_matrix = np.array([np.array(tmp_matrix[i]) for i in range(n)])

    tmp_matrix = normalize(tmp_matrix, axis=1, norm='l1')

    return tmp_matrix, keyword_to_state
