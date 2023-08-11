"""
For License information see the LICENSE file.


"""
import logging
import random
import sys
from typing import Callable
import numpy as np


from leaker.api import DataSink
from leaker.api.dataset import DummyKeywordQueryLogFromList
from leaker.attack import MarkovSorting, MarkovIHOP
from leaker.attack.markov import MarkovDecoding

from leaker.attack.query_space import ZipfZipfKeywordQuerySpace, FullUserQueryLogSpace
from leaker.evaluation import EvaluationCase
from leaker.evaluation.evaluator import KeywordQueryAttackEvaluator
from leaker.evaluation.param import KeywordQueryScenario

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('test_laa_eval.log', 'w', 'utf-8')
file.setFormatter(f)

log = logging.getLogger(__name__)

logging.basicConfig(handlers=[console, file], level=logging.INFO)


def init_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)


class EvaluatorTestSink(DataSink):
    __n: int
    __cb: Callable[[str, int, float, float, int], None]

    def __init__(self, callback: Callable[[str, int, float, float, int], None]):
        self.__n = 0
        self.__cb = callback

    def register_series(self, series_id: str, user_ids: int = 1) -> None:
        pass

    def offer_data(self, series_id: str, user_id: int, kdr: float, rr: float) -> None:
        self.__cb(series_id, kdr, rr, self.__n)
        self.__n += 1

    def flush(self) -> None:
        pass




def test_markov_attacks():
    init_rngs(5)

    num_states = 40

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        if series_id == "MarkovDecoding":
            assert 0.35 < rr
        else:
            assert 0.0 < rr

    verifier = EvaluatorTestSink(verif_cb)

    run = KeywordQueryAttackEvaluator(
        EvaluationCase([MarkovSorting, MarkovDecoding, MarkovIHOP.definition(pfree=.25, niters=10000)],
                       dataset=None, runs=1, scenario=KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST),
        ZipfZipfKeywordQuerySpace(num_states, None, True),
        query_counts=[5*10 ** 5],
        sinks=verifier,
        parallelism=8)
    run.run()


def test_qlog_markov_attacks():
    init_rngs(5)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        if series_id != "MarkovIHOP":
            assert 0.2 < rr


    qlog = DummyKeywordQueryLogFromList("test", ['kod', 'atpf', 'amp', 'type', 'locus', 'fcaall', 'anln', 'psbd', 'cud',
                                                 'lfy', 'tac', 'library', '0.49', '8009', '20', 'psba', 'py', 'fcaall',
                                                 'annotated', 'ndhd', 'snrk', 'tbr', 'fass', 'psbd', 'no_match', 'ndhd',
                                                 'knolle', 'ndhi', 'secrete', '49', '8009', '33', 'http',
                                                 'www.arabidopsis.org', 'servlets', 'tairobject', 'type', 'locus',
                                                 'http', 'www.arabidopsis.org', 'servlets', 'tairobject', 'type',
                                                 'locus', 'http', 'www.arabidopsis.org', 'servlets', 'tairobject',
                                                 'type', 'locus', 'http', 'www.arabidopsis.org', 'servlets',
                                                 'tairobject', 'type', 'locus', 'http', 'www.arabidopsis.org',
                                                 'servlets', 'tairobject', 'type', 'locus', 'http',
                                                 'www.arabidopsis.org', 'servlets', 'tairobject', 'type', 'locus',
                                                 'http', 'www.arabidopsis.org', 'servlets', 'tairobject', 'type',
                                                 'locus', 'http', 'www.arabidopsis.org', 'servlets', 'tairobject',
                                                 'type', 'germplasm', 'toz', 'http', 'arabidopsis.org', 'servlets',
                                                 'tairobject', 'type', 'locus', 'psbt', 'versailles', 'vb', 'ativd',
                                                 'attopii', '40', 'psaj', 'atmgl', 'atpb', 'aty', 'rbcl', 'atpdat',
                                                 'phya', 'bac', 'library'])

    num_states = len(qlog.keywords())

    verifier = EvaluatorTestSink(verif_cb)

    run = KeywordQueryAttackEvaluator(
        EvaluationCase([MarkovSorting, MarkovDecoding, MarkovIHOP.definition(pfree=.25, niters=10000)],
                       dataset=qlog, runs=1, scenario=KeywordQueryScenario.EXACT),
        FullUserQueryLogSpace(num_states, qlog, True),
        query_counts=[10 ** 4],
        sinks=verifier,
        parallelism=8)
    run.run()



