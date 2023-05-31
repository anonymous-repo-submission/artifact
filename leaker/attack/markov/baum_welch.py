"""
This file implement the Baum Welch Attack.

For License information see the LICENSE file.

"""
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import TypeVar, List, Any, Dict, Union

import numpy as np
import networkx as nx
from bidict import bidict

from networkx.algorithms import isomorphism
from sklearn.preprocessing import normalize

from leaker.api import Extension, KeywordQueryAttack, LeakagePattern, QuerySequence
from leaker.api.constants import CACHE_DIRECTORY
from leaker.attack.markov.util import trans_matrix_from_seq
from leaker.pattern import QueryEquality
from .util import baum_welch

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class MarkovBaumWelch(KeywordQueryAttack):
    __num_states: int
    __t_matrix: np.ndarray
    __keyword_to_state: Dict[int, str]

    def __init__(self, known: Union[QuerySequence, List[List[str]]]):
        super().__init__(known)
        log.info(f"Setting up MarkovBW attack.")
        if isinstance(known, list):
            # Build trans matrix out of all other users except the one provided.
            # The users under attack need to be included in the full dataset
            size = len(set(kw for l in known for kw in l))
            t_mat = np.zeros((size, size))
            keyword_to_state = {}

            for keyword_list in known:
                t_mat_new, keyword_to_state = trans_matrix_from_seq(keyword_list, size, keyword_to_state)
                t_mat += t_mat_new

            self.__t_matrix = normalize(t_mat, axis=1, norm='l1')  # normalize rows to sum up to 1
            self.__num_states = size
            self.__keyword_to_state = keyword_to_state
        else:
            self.__num_states = known.num_states
            self.__t_matrix, self.__keyword_to_state = trans_matrix_from_seq(known.query_list, known.num_states)

        log.info(f"Setup done.")

    @classmethod
    def name(cls) -> str:
        return "MarkovBW"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [QueryEquality]

    @staticmethod
    def get_graph_stats(graph: np.ndarray) -> Dict[int, Dict[int, int]]:
        graph = graph.tolist()
        output = {}
        for n in range(len(graph)):
            output[n] = {}
            # sum out edges
            output[n][0] = sum(i ** 2 for i in graph[n])
            # sum in edges
            output[n][1] = sum(i ** 2 for i in np.array(graph)[:, n])
            # outdegree
            output[n][2] = np.count_nonzero(graph[n])
            # indegree
            output[n][3] = np.count_nonzero(np.array(graph)[:, n])
        return output

    @staticmethod
    def find_best_unused_match(base: np.ndarray, pattern: np.ndarray, unmatched1: List[int], unmatched2: List[int],
                               matched: Dict[int, int]) -> List[int]:
        base = base.tolist()
        pattern = pattern.tolist()
        curr = 99999
        match = []
        for node in unmatched1:
            for node2 in unmatched2:
                temp = 0
                for key in matched:
                    temp += (pattern[key][node] - base[matched[key]][node2]) * (
                            pattern[key][node] - base[matched[key]][node2])
                    temp += (pattern[node][key] - base[node2][matched[key]]) * (
                            pattern[node][key] - base[node2][matched[key]])
                if temp < curr:
                    match = [node, node2]
                    curr = temp
        return match

    @staticmethod
    def graph_matcher(base_graph: np.ndarray, pattern_graph: np.ndarray) -> Dict[int, int]:
        matches = {}
        num_states = len(pattern_graph)

        base_stats = MarkovBaumWelch.get_graph_stats(base_graph)
        pattern_stats = MarkovBaumWelch.get_graph_stats(pattern_graph)
        unmatched1 = [i for i in range(num_states)]
        unmatched2 = [i for i in range(len(base_graph))]

        # first layer of the matching / finding "ground truth" nodes
        curr = 99999
        ground_truth = []
        for node in unmatched1:
            for node2 in unmatched2:
                if (base_stats[node2][2] == pattern_stats[node2][2]) or (
                        base_stats[node2][3] == pattern_stats[node2][3] and float(base_stats[node2][2]) != 0.0 and
                        float(base_stats[node2][3]) != 0.0):
                    if ((abs(base_stats[node2][0] - pattern_stats[node][0]) / float(base_stats[node2][2])) + (
                            abs(base_stats[node2][1] - pattern_stats[node][1]) / float(base_stats[node2][3])) < curr):
                        ground_truth = [node, node2]
                        curr = (abs(base_stats[node2][0] - pattern_stats[node][0]) / float(base_stats[node2][2])) + (
                                abs(base_stats[node2][1] - pattern_stats[node][1]) / float(base_stats[node2][3]))

        if len(ground_truth) == 0:
            log.warning(f"No Ground Truth found!")
            return {}
        else:
            matches[ground_truth[0]] = ground_truth[1]
            unmatched1.remove(ground_truth[0])
            unmatched2.remove(ground_truth[1])

            while len(unmatched1) > 0:
                match = MarkovBaumWelch.find_best_unused_match(base_graph, pattern_graph, unmatched1, unmatched2,
                                                               matches)
                matches[match[0]] = match[1]
                unmatched1.remove(match[0])
                unmatched2.remove(match[1])

            return matches

    @staticmethod
    def my_edge_match(data1, data2):

        if data1[0]['weight'] == 0:
            return True
        if data2[0]['weight'] == 0:
            return True
        return abs(data1[0]['weight'] - data2[0]['weight']) < .3

    @staticmethod
    def build_graph_from_matrix(m, isFull):
        """
            This function is written to build a directed graph representing a hidden
            Markov model. Thus it takes two parameters, the transition matrix and the
            emissions matrix
            m: the graph matrix

            builds a directed graph representing the hidden markov model defined by the
            transition matrix and the emissions matrix

            returns: the generated graph
        """

        DG = nx.MultiDiGraph()
        r, c = np.shape(m)

        if isFull:
            for i in range(r):
                for j in range(c):
                    DG.add_edge(i, j, weight=m[i, j])
        else:
            for i in range(r):
                for j in range(c):
                    if m[i, j] != 0:
                        DG.add_edge(i, j, weight=m[i, j])


        return DG

    @staticmethod
    def subgraph_match(G1, G2, smallG2):
        """
            G1: The larger graph
            G2: The smaller graph

            Finds the subgraph of G1 that best matches G2

            returns: the mapping from G# to G#
        """


        matcher = isomorphism.MultiDiGraphMatcher(G1, G2, edge_match=lambda x, y: MarkovBaumWelch.my_edge_match(x, y))

        if matcher.subgraph_is_isomorphic():
            print('theyre isomorphic :)')
        else:
            print('not isomorphic :(')
        isos = matcher.subgraph_isomorphisms_iter()

        count = 0
        valid_isos = []
        for i in isos:
            log.info(f"doing {count}")
            newG1 = nx.relabel_nodes(G1, i)
            G2_nodes = smallG2.nodes()
            is_valid = True
            dist = 0.0
            for n in G2_nodes:

                G2_out_edges = smallG2.edges([n], data='weight')
                G1_out_edges = newG1.edges([n], data='weight')
                for e2 in G2_out_edges:
                    for e1 in G1_out_edges:
                        if e2[0] == e1[0] and e2[1] == e1[1] and abs(e2[2] - e1[2]) >= .3:
                            is_valid = False
                        elif e2[0] == e1[0] and e2[1] == e1[1]:
                            dist += abs(e2[2] - e1[2])

            if is_valid:
                valid_isos.append((i, dist))
                count += 1

        return valid_isos

    @staticmethod
    def subgraph_match_simple(base: np.ndarray, pattern: np.ndarray) -> Dict[int, int]:
        edges_b = set()
        states_b = len(base[0])

        edges_p = set()
        states_p = len(pattern[0])

        degrees_b = [len(np.nonzero(row)[0]) for row in base]

        degrees_p = [len(np.nonzero(row)[0]) for row in pattern]

        it_b = np.nditer(base, flags=['multi_index'])
        for elem in it_b:
            if elem > 0.0:
                edges_b.add(it_b.multi_index)

        it_p = np.nditer(pattern, flags=['multi_index'])
        for elem in it_p:
            if elem > 0.0:
                edges_p.add(it_p.multi_index)

        known = {np.argmax(degrees_p): np.argmax(degrees_b)}

        keep_running = True
        while keep_running:
            cand = {}
            for v_p, v_b in known.items():
                for v_pp in range(states_p):
                    if v_pp not in known.keys() and (v_p, v_pp) in edges_p:
                        cands_vpp = set(v_bp for v_bp in range(states_b) if (v_b, v_bp) in edges_b and
                                        (not (v_bp, v_b) in edges_b or (v_pp, v_p) in edges_p) and
                                        v_bp not in known.values())
                        if v_pp not in cand.keys():
                            cand[v_pp] = cands_vpp
                        else:
                            cand[v_pp] = cand[v_pp].union(cands_vpp)

            success = False
            for v, c in cand.items():
                if len(c) == 1:
                    known[v] = c.pop()
                    success = True

            if not success:
                open_vertices = cand.keys()
                if len(open_vertices) == 0:
                    keep_running = False
                else:
                    open_vertices_p = sorted(open_vertices, key=lambda v: degrees_p[v], reverse=True)
                    open_vertices_b = sorted(cand[open_vertices_p[0]], key=lambda v: degrees_b[v], reverse=True)

                    known[open_vertices_p[0]] = open_vertices_b[0]

        return known

    def recover(self, queries: List[int]) -> List[int]:
        log.info(f"Running BW Attack on queries {len(set(queries))}")
        with TemporaryDirectory(dir=CACHE_DIRECTORY) as tmp_dir:
            os.system(f"mkdir {tmp_dir}/bw_r")
            os.system(f"touch {tmp_dir}/em.txt")
            t_matrix2, state_order = baum_welch([queries], f'{tmp_dir}/bw_r', em_path=f'{tmp_dir}/em.txt',
                                                start_path=None, trans_path=None, em_identity=True, true_num_states=-1)

        log.info(f"BW Algo completed.")

        matcher = self.subgraph_match_simple(self.__t_matrix, t_matrix2)

        keyword_to_state = bidict(self.__keyword_to_state)

        res = []
        for i in queries:
            if i in matcher.keys():
                res.append(keyword_to_state.inverse[matcher[i]])
            else:
                res.append("")

        return res
