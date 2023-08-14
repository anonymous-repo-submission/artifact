"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber, Michael Yonli

"""

from abc import abstractmethod
from itertools import starmap
from logging import getLogger
from multiprocessing.pool import ThreadPool, Pool
from bidict import bidict
from typing import List, Union, Iterable, Tuple, Iterator, Optional, Type, Dict

from leaker.api import KeywordQuerySpace, AttackDefinition, QuerySequence, KeywordQueryLog
from leaker.attack.markov.util import trans_matrix_from_seq
from .errors import Error
from .param import EvaluationCase, DatasetSampler, QuerySelector, KeywordQueryScenario
from ..api import Attack, RangeAttack, KeywordAttack, Dataset, DataSink, RangeQuerySpace, RangeDatabase
from ..util.time import Stopwatch

log = getLogger(__name__)


class Evaluator:
    _evaluation_case: EvaluationCase
    _sinks: List[DataSink]
    _parallelism: int

    def __init__(self, evaluation_case: EvaluationCase, sinks: Union[DataSink, Iterable[DataSink]],
                 parallelism: int = 1):
        if isinstance(sinks, DataSink):
            self._sinks = [sinks]
        else:
            self._sinks = list(sinks)
        self._evaluation_case = evaluation_case
        self._parallelism = parallelism

    @abstractmethod
    def run(self):
        raise NotImplementedError


class KeywordAttackEvaluator(Evaluator):
    """
    A KeywordAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific dataset. It is capable
    of running multiple attacks in parallel to speed up the evaluation.

    Parameters
    ----------
    evaluation_case : EvaluationCase
        the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
    dataset_sampler : DatasetSampler
        the data set sampling settings, including the known data rate values
    query_selector : QuerySelector
        the policies for selecting the query sequence including the selectivity, the type and size of the query space
        and the number of queries
    sinks : Union[DataSink, Iterable[DataSink]]
        one or multiple data sinks to write performance data to
    parallelism : int
        the number of parallel threads to use in the evaluation
        default: 1
    """
    __dataset_sampler: DatasetSampler
    __query_selector: QuerySelector

    def __init__(self, evaluation_case: EvaluationCase, dataset_sampler: DatasetSampler, query_selector: QuerySelector,
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self.__dataset_sampler = dataset_sampler
        self.__query_selector = query_selector

    @staticmethod
    def __evaluate(dataset: Dataset, user: int, kdr: float, attack: KeywordAttack, queries: List[str]) -> \
            Tuple[str, int, float, float]:
        # recover queries using the given attack
        recovered = attack(dataset, queries)
        # count matches
        correct = [actual == recovered for actual, recovered in zip(queries, recovered)].count(True)

        return attack.name(), user, kdr, correct / len(queries)

    def __to_inputs(self, dataset: Dataset, kdr: float, known: Dataset) -> Iterator[Tuple[Dataset, int, float, Attack,
                                                                                          List]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data rate
        for i, queries in enumerate(self.__query_selector.select(dataset, known)):
            for attack in self._evaluation_case.attacks():
                yield dataset, i, kdr, attack.create(known), queries

    def __produce_input(self, pool: Optional[Pool] = None) \
            -> Iterator[Tuple[Dataset, int, float, KeywordAttack, List[str]]]:
        # yield all input tuples for __evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi threading pool
        datasets = self._evaluation_case.datasets()
        if pool is None:
            for inputs in starmap(self.__to_inputs, self.__dataset_sampler.sample(datasets)):
                yield from inputs
        else:
            for inputs in pool.starmap(self.__to_inputs, iterable=self.__dataset_sampler.sample(datasets, pool)):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""

        log.info(f"Running {self._evaluation_case.runs()} evaluation runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        reuse: bool = self.__dataset_sampler.reuse()

        stopwatch = Stopwatch()
        stopwatch.start()

        # Perform desired number of runs
        for run in range(1, self._evaluation_case.runs() + 1):
            log.info("######################################################################################")
            log.info(f"# RUN {run}")
            log.info("######################################################################################")

            sample_runs: int = 1
            if reuse:
                sample_runs: int = self._evaluation_case.runs()
                self.__dataset_sampler.set_reuse(True)

            for sample_run in range(1, sample_runs + 1):
                log.info(f"Starting evaluation {run}-{sample_run} with new queries")
                if self._parallelism == 1:
                    # do evaluation sequentially
                    performances: List[Tuple[str, int, float, float]] = []
                    for dataset, user, kdr, attack, queries in self.__produce_input():
                        performances.append(KeywordAttackEvaluator.__evaluate(dataset, user, kdr, attack, queries))

                else:
                    # create thread pool and do evaluation in parallel
                    with ThreadPool(processes=self._parallelism) as pool:
                        performances = pool.starmap(func=KeywordAttackEvaluator.__evaluate,
                                                    iterable=self.__produce_input(pool))
                        log.info("All computations completed.")

                for series, user, kdr, result in performances:
                    for sink in self._sinks:
                        sink.offer_data(series, user, kdr, result)

            log.info(f"RUN {run} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()


class KeywordQueryAttackEvaluator(Evaluator):
    """
       A KeywordQueryAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific
       QuerySequence. It is capable of running multiple attacks in parallel to speed up the evaluation.

       Parameters
       ----------
       evaluation_case : EvaluationCase
           the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
       queries : QuerySpace
           the policies for selecting the query sequence (query distribution)
       query_counts : List[int]
            the amount of queries the attack shall be evaluated on
       sinks : Union[DataSink, Iterable[DataSink]]
           one or multiple data sinks to write performance data to
       parallelism : int
           the number of parallel threads to use in the evaluation
           default: 1
       """

    __queries: KeywordQuerySpace
    __queries_n: List[int]

    def __init__(self, evaluation_case: EvaluationCase, queries: KeywordQuerySpace, query_counts: List[int],
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        if not queries.sample_queries():
            self.__queries_n = [len(list(queries.select(1))[0])]  # dummy length to get actual sequence length
        else:
            self.__queries_n = query_counts
        self.__queries = queries

    @staticmethod
    def _evaluate(queries: List[int], full_sequence: QuerySequence, attack: AttackDefinition, user: int,
                  keyword_to_state: bidict[str, int], alt_state_map: Dict[int, int]) \
            -> Tuple[str, int, float, int]:
        attack = attack.create(full_sequence)

        recovered = attack.recover(queries)

        queries = [alt_state_map[q] for q in queries]  # Un-transform possibly transformed queries

        # TODO: Remove any queries mapping to "dummy*"

        correct = [keyword_to_state.inverse[actual] == recovered for actual, recovered in zip(queries, recovered)]. \
            count(True)

        return attack.name(), len(queries), correct / len(queries), user

    def _to_inputs(self, query_count: int) -> Iterator[Tuple[List[int], QuerySequence, AttackDefinition, int,
                                                             bidict[str, int], Dict[int, int]]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data

        if self._evaluation_case.scenario() == KeywordQueryScenario.ARTIFICIAL:
            ref_seqs = list(self.__queries.select(10 ** 4, from_original=True))
            # sample adversarially known sequences for artificial eval without any transforms applied
            # to have another start state than at the user_queries

        qlog: KeywordQueryLog = self._evaluation_case.full_dataset()
        for i, user_queries in enumerate(self.__queries.select(query_count)):
            if qlog is None and self._evaluation_case.scenario() != KeywordQueryScenario.ARTIFICIAL \
                    and self._evaluation_case.scenario() != KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST:
                log.warning(f"Query-log specific KeywordQueryScenario without a supplied query log.")

            target_qlog = self.__queries.get_query_log()  # TODO: Ensure keywords match

            if self._evaluation_case.scenario() == KeywordQueryScenario.ALL_USERS:
                known_sequences = [[kw for kw in qlog.keywords_list(uid, True)]
                                   for uid in qlog.user_ids()]
            elif self._evaluation_case.scenario() == KeywordQueryScenario.OTHER_USERS:
                known_sequences = [[kw for kw in qlog.keywords_list(uid, True)]
                                   for uid in qlog.user_ids()
                                   if (uid != target_qlog.user_ids()[i])]
            elif self._evaluation_case.scenario() == KeywordQueryScenario.EXACT:
                known_sequences = [[kw for kw in qlog.keywords_list(target_qlog.user_ids()[i], True)]]
            elif self._evaluation_case.scenario() == KeywordQueryScenario.SPLIT \
                    or self._evaluation_case.scenario() == KeywordQueryScenario.SPLIT_SUB_KNOWN_DIST:
                known_sequences = [[kw for kw in qlog.keywords_list(target_qlog.user_ids()[i], True)]]
                # Take first half as training set
                known_sequences[0] = known_sequences[0][:len(known_sequences[0]) // 2]
                last = known_sequences[0][-1]
                while last not in known_sequences[0][:-2]:  # element last needs to appear
                    # one time before becoming the last element
                    known_sequences[0] = known_sequences[0][:-1]
                    last = known_sequences[0][-1]

                if self._evaluation_case.scenario() == KeywordQueryScenario.SPLIT_SUB_KNOWN_DIST:
                    # In this case, we learn a trans matrix on the train set and give it to the attacker
                    test_qseq = self.__queries.get_full_sequence(i)
                    log.info(f"New test mat has {test_qseq.num_states} states")
                    tmp_matrix, keyword_to_state = trans_matrix_from_seq(known_sequences[0],
                                                                         len(set(known_sequences[0])))

                    train_qseq = QuerySequence(transition_matrix=tmp_matrix, num_states=len(tmp_matrix[0]),
                                               query_list=[],
                                               keyword_to_state=keyword_to_state, alt_state_map=test_qseq.alt_state_map,
                                               original_transition_matrix=tmp_matrix)
                    known_sequences = train_qseq

                    log.info(f"New train mat has {len(keyword_to_state.keys())} states")

            elif self._evaluation_case.scenario() == KeywordQueryScenario.ARTIFICIAL:
                known_sequences = [ref_seqs[i]]
            elif self._evaluation_case.scenario() == KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST:
                known_sequences = self.__queries.get_full_sequence(i)

            for attack in self._evaluation_case.attacks():
                yield user_queries, known_sequences, attack, i, \
                      bidict(self.__queries.get_full_sequence(i).keyword_to_state), \
                      self.__queries.get_full_sequence(i).alt_state_map
        if self._evaluation_case.scenario() == KeywordQueryScenario.ARTIFICIAL or \
                self._evaluation_case.scenario() == KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST:
            self.__queries.resample()  # Get a new sample of the transition matrix of the specified distribution

    def _produce_input(self, query_count: int, pool: Optional[Pool] = None) -> Iterator[Tuple[List[int], QuerySequence,
                                                                                              AttackDefinition, int,
                                                                                              bidict[str, int], Dict[
                                                                                                  int, int]]]:
        # yield all input tuples for __evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi processing pool
        if pool is None:
            for inputs in map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs
        else:
            for inputs in pool.map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""
        log.info(f"Running {len(self.__queries_n)} query runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        stopwatch = Stopwatch()
        stopwatch.start()

        # Perform desired number of runs for each #queries
        for query_count in self.__queries_n:
            log.info("######################################################################################")
            log.info(f"# Queries {query_count}")
            log.info("######################################################################################")

            if self._parallelism == 1:
                # do evaluation sequentially
                for user_queries, sequence, attack, i, keyword_to_state, alt_state_map \
                        in self._produce_input(query_count):

                    attack_name, n_q, result, user = KeywordQueryAttackEvaluator._evaluate(user_queries, sequence,
                                                                                           attack, i, keyword_to_state,
                                                                                           alt_state_map)

                    if not self.__queries.sample_queries():
                        n_q = 1  # In this case we don't have an x-axis

                    for sink in self._sinks:
                        sink.offer_data(attack_name, user, n_q, result)
            else:
                # create pool and do evaluation of multiple runs in parallel
                with Pool(processes=self._parallelism) as pool:
                    results = pool.starmap(func=KeywordQueryAttackEvaluator._evaluate,
                                           iterable=self._produce_input(query_count, None))
                    log.info("All computations completed.")

                    for attack_name, n_q, result, user in results:
                        if not self.__queries.sample_queries():
                            n_q = 1  # In this case we don't have an x-axis
                        for sink in self._sinks:
                            sink.offer_data(attack_name, user, n_q, result)

            log.info(f"QUERY RUN {query_count} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()


class RangeAttackEvaluator(Evaluator):
    """
       A RangeAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific
       RangeDatabase. It is capable of running multiple attacks in parallel to speed up the evaluation.

       Parameters
       ----------
       evaluation_case : EvaluationCase
           the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
       range_queries : RangeQuerySpace
           the policies for selecting the query sequence (query distribution)
       query_counts : List[int]
            the amount of queries the attack shall be evaluated on
       sinks : Union[DataSink, Iterable[DataSink]]
           one or multiple data sinks to write performance data to
       normalize : bool
            whether to normalize the displayed reconstruction errors
            default: True
       parallelism : int
           the number of parallel threads to use in the evaluation
           default: 1
       """

    __normalize: bool
    __queries: RangeQuerySpace
    __queries_n: List[int]

    def __init__(self, evaluation_case: EvaluationCase, range_queries: RangeQuerySpace, query_counts: List[int],
                 sinks: Union[DataSink, Iterable[DataSink]], normalize: bool = True, parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self.__normalize = normalize
        self.__queries = range_queries
        self.__queries_n = query_counts

    @staticmethod
    def _evaluate(db: RangeDatabase, attack: RangeAttack, queries: List[Tuple[int, int]], error: Type[Error],
                  normalize: bool, user: int) \
            -> Tuple[str, float, float, int]:
        recovered = attack.recover(queries)

        return attack.name(), len(queries), error.calc_error(db, recovered, normalize), user

    def _to_inputs(self, query_count: int) -> Iterator[Tuple[RangeDatabase, RangeAttack, List[Tuple[int, int]],
                                                             Type[Error], bool, int]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data rate
        for db in self._evaluation_case.datasets():
            for attack in self._evaluation_case.attacks():
                for i, queries in enumerate(self.__queries.select(query_count)):
                    yield db, attack.create(db), queries, self._evaluation_case.error(), self.__normalize, i

    def _produce_input(self, query_count: int, pool: Optional[Pool] = None) -> Iterator[Tuple[RangeDatabase,
                                                                                              RangeAttack,
                                                                                              List[Tuple[int, int]],
                                                                                              Type[Error], bool, int]]:
        # yield all input tuples for __evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi processing pool
        if pool is None:
            for inputs in map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs
        else:
            for inputs in pool.map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""
        log.info(f"Running {len(self.__queries_n)} query runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        stopwatch = Stopwatch()
        stopwatch.start()

        # Perform desired number of runs for each #queries
        for query_count in self.__queries_n:
            log.info("######################################################################################")
            log.info(f"# Queries {query_count}")
            log.info("######################################################################################")

            if self._parallelism == 1:
                # do evaluation sequentially
                for db, attack, queries, error, normalize, user in self._produce_input(query_count):
                    attack_name, n_q, result, user = RangeAttackEvaluator._evaluate(db, attack, queries, error,
                                                                                    normalize, user)

                    for sink in self._sinks:
                        sink.offer_data(attack_name, user, n_q, result)
            else:
                # create pool and do evaluation of multiple runs in parallel
                with Pool(processes=self._parallelism) as pool:
                    results = pool.starmap(func=RangeAttackEvaluator._evaluate,
                                           iterable=self._produce_input(query_count, None))
                    log.info("All computations completed.")

                    for attack_name, n_q, result, user in results:
                        for sink in self._sinks:
                            sink.offer_data(attack_name, user, n_q, result)

            log.info(f"QUERY RUN {query_count} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()
