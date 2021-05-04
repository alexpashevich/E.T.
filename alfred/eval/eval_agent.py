import os

from sacred import Experiment

from alfred.config import exp_ingredient, eval_ingredient
from alfred.eval.eval_master import EvalMaster
from alfred.eval.eval_task import evaluate_task, process_eval_task
from alfred.eval.eval_subgoals import evaluate_subgoals, process_eval_subgoals
from alfred.gen import constants
from alfred.utils import eval_util, helper_util

ex = Experiment('eval_agent', ingredients=[eval_ingredient, exp_ingredient])


@ex.automain
def main(eval, exp):
    # arguments
    args = helper_util.AttrDict(**eval, **exp)
    args.dout = os.path.join(constants.ET_LOGS, args.exp)
    os.makedirs(args.dout, exist_ok=True)

    # create a queue of trials to be performed and a logging queue
    model_paths = eval_util.get_model_paths(args)
    master = EvalMaster(args, model_paths[0])
    trial_queue, log_queue = master.create_queues(model_paths)
    if trial_queue.qsize() > 0:
        evaluate_function = evaluate_subgoals if args.subgoals else evaluate_task
        # start the evaluation
        if args.num_workers > 0:
            # start threads
            workers = master.launch_workers(evaluate_function, trial_queue, log_queue)
        else:
            # debug mode
            eval_util.worker_loop(evaluate_function, master.dataset,
                        trial_queue, log_queue, master.args)
            workers = []
        # wait for workers results and log them
        master.gather_results(workers, log_queue)

    # parse what the threads have computed and save the results
    process_function = process_eval_subgoals if args.subgoals else process_eval_task
    process_function(master.results_path, model_paths, args)
