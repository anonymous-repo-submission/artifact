"""
For License information see the LICENSE file.



"""
import logging
import sys

from leaker.attack import MarkovSorting, MarkovIHOP
from leaker.attack.markov import MarkovDecoding, BinomialMarkovDecoding
from leaker.attack.query_space import ZipfKeywordQuerySpace, ZipfZipfKeywordQuerySpace, ErdosGraphKeywordQuerySpace, \
    UniformKeywordQuerySpace
from leaker.evaluation import EvaluationCase
from leaker.evaluation.evaluator import KeywordQueryAttackEvaluator
from leaker.evaluation.param import KeywordQueryScenario
from leaker.plotting import RangeMatPlotLibSink

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('keyword_query_attacks_eval.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

min_hw = 2
for qsp in [ZipfKeywordQuerySpace, ZipfZipfKeywordQuerySpace, ErdosGraphKeywordQuerySpace, UniformKeywordQuerySpace]:
    for num_states in [250, 500, 1000, 1500]:
        log.info(f"Running artificial eval for {num_states} states.")
        scen = KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST
        eva = KeywordQueryAttackEvaluator(
            EvaluationCase([BinomialMarkovDecoding, MarkovSorting, MarkovDecoding, MarkovIHOP.definition(pfree=.25, niters=10000)],
                           dataset=None, runs=30, scenario=scen),
            qsp(num_states),
            query_counts=[10 ** 3, 10 ** 4, 5 * 10 ** 4, 10 ** 5, 5 * 10 ** 5],
            sinks=RangeMatPlotLibSink(f"markovFINAL_{qsp}_{num_states}_{scen}.png", use_mean=False),
            parallelism=30)

        eva.run()


num_states = 500
for min_hw in [5, 100, 350, 450]:
    log.info(f"Running artificial eval for {num_states} states with min weight {min_hw}.")
    scen = KeywordQueryScenario.ARTIFICIAL_KNOWN_DIST
    eva = KeywordQueryAttackEvaluator(
        EvaluationCase([BinomialMarkovDecoding, MarkovSorting, MarkovDecoding, MarkovIHOP.definition(pfree=.25, niters=10000)],
                       dataset=None, runs=30, scenario=scen),
        ZipfZipfKeywordQuerySpace(num_states, min_hw=min_hw),
        query_counts=[10 ** 3, 10 ** 4, 5 * 10 ** 4, 10 ** 5, 5 * 10 ** 5],
        sinks=RangeMatPlotLibSink(f"markovFINAL_zipfzipf_{num_states}_{min_hw}_{scen}.png", use_mean=False),
        parallelism=30)

    eva.run()
