"""
This file implements linear algebra helper functions.

For License information see the LICENSE file.

"""
import numpy as np

from scipy.linalg import eig


def calc_stationary_dist(transition_mat: np.ndarray) -> np.ndarray:
    s, u = eig(transition_mat.T)
    evec1 = u[:, np.isclose(s, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real

    stationary[stationary < 0.0] = 1e-20

    return stationary
