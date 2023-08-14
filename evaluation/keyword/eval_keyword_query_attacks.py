"""
For License information see the LICENSE file.



"""
import logging
import sys

from leaker.attack import MarkovSorting, FullUserQueryLogSpace, MarkovIHOP
from leaker.attack.markov import MarkovDecoding, BinomialMarkovDecoding
from leaker.evaluation import EvaluationCase
from leaker.evaluation.evaluator import KeywordQueryAttackEvaluator
from leaker.evaluation.param import KeywordQueryScenario
from leaker.plotting import RangeMatPlotLibSink
from leaker.whoosh_interface import WhooshBackend
from leaker.attack.query_space import FullUserSplitQueryLogSpace

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('keyword_query_attacks_eval.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)


backend = WhooshBackend()
file_description = "test"

'''
The aol_new Querylog used here to evaluate the AOL dataset can be indexed using the index_aol.py, 
same can be done for the TAIR Dataset using index_tait.py  
'''

for freq, freq_str in [(True, "infreq")]:
    q_log = backend.load_querylog(f"aol_new", pickle_description=file_description, min_user_count=22, max_user_count=27,
                              reverse=freq)  # 5 users attacked
    log.info(f"Loaded Log. {len(q_log.doc_ids())} searches performed for {len(q_log.keywords())} words.")
    for i in [5]:  # 2*i users in total
        q_log_adv = backend.load_querylog(f"aol_new", pickle_description=file_description, min_user_count=24 - i, max_user_count=25 + i,
                                          reverse=freq)
        log.info(f"Loaded Log. {len(q_log_adv.doc_ids())} searches performed for {len(q_log_adv.keywords())} words.")

        num_states = len(q_log_adv.keywords(remove_endstates=True))

        num_states_user = len(q_log.keywords(remove_endstates=True))

        log.info(f"num_states adv: {num_states} vs. attacked: {len(q_log.keywords(remove_endstates=True))}")

        for sampling in [True, False]:
            if sampling:
                sample_string = "sampled"
            else:
                sample_string = "not-sampled"

            for scen in [KeywordQueryScenario.ALL_USERS, KeywordQueryScenario.OTHER_USERS, KeywordQueryScenario.EXACT]:

                log.info(f"Running {freq_str}_{i}_{sample_string}_{scen}.")

                i_q_log = q_log
                if scen == KeywordQueryScenario.SPLIT:
                    i_q_log = q_log_adv
                    num_states = max([len(q_log_adv.keywords(i, remove_endstates=True)) for i in q_log_adv.user_ids()])
                    qsp = FullUserSplitQueryLogSpace(num_states, i_q_log, sampling)
                else:
                    num_states = max([len(q_log.keywords(i, remove_endstates=True)) for i in q_log.user_ids()])
                    qsp = FullUserQueryLogSpace(num_states, i_q_log, sampling)

                log.info(f"The max number of states is {num_states}. Number of states of each user are {[len(q_log.keywords(i, remove_endstates=True)) for i in q_log.user_ids()]}")

                eva = KeywordQueryAttackEvaluator(EvaluationCase([BinomialMarkovDecoding, MarkovSorting, MarkovDecoding, MarkovIHOP.definition(pfree=.25, niters=10000)],
                                                                 dataset=q_log_adv, runs=10, scenario=scen),
                                                  qsp,
                                                  query_counts=[10**3, 10**4, 5*10**4, 10**5, 5*10**5],
                                                  sinks=RangeMatPlotLibSink(f"markovFINAL_AOL__{i}_{sample_string}_{scen}.png", use_mean=False),
                                                  parallelism=30)

                eva.run()
