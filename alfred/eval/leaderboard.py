import os
import json

from datetime import datetime
from sacred import Experiment

from alfred.config import exp_ingredient, eval_ingredient
from alfred.eval.eval_master import EvalMaster
from alfred.gen import constants
from alfred.utils import eval_util, helper_util


ex = Experiment('eval_agent', ingredients=[eval_ingredient, exp_ingredient])


def evaluate_test(
    env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor):
    # load trajectory data from the dataset
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx = int(trial_uid.split(':')[1])
    assert traj_data['repeat_idx'] == r_idx
    # reset model and setup scene
    model.reset()
    eval_util.setup_scene(env, traj_data, reward_type='dense', test_split=True)
    vocab = {'word': dataset.vocab_in, 'action_low': model.vocab_out}
    # load language features and task info
    input_dict = eval_util.load_language(
        dataset, traj_data, traj_key, model.args, extractor, test_split=True)
    anns = traj_data['turk_annotations']['anns'][r_idx]
    print('Solving "{}"'.format(anns['task_desc']))

    prev_action = None
    t, num_fails = 0, 0
    actions = []

    while t < args.max_steps:
        input_dict['frames'] = eval_util.get_observation(env.last_event, extractor)
        episode_end, prev_action, num_fails, _, api_action = eval_util.agent_step(
            model, input_dict, vocab, prev_action, env, args, num_fails, obj_predictor)
        t += 1
        # save action
        if api_action is not None:
            actions.append(api_action)
        # break if stop is predicted or args.max_fails is reached
        if episode_end:
            break
    log_entry = {'actseq': {traj_data['task_id']: actions}, 'anns': anns}
    return log_entry


def process_results(results_paths, model_paths):
    '''
    save actseqs as JSONs
    '''
    assert len(results_paths) == 2
    print('Processing tests results')
    results, trials = {}, {}
    for results_path in results_paths:
        with open(results_path, 'r') as results_file:
            trial_results = json.load(results_file)
        split = results_path.split('/')[-1].split(':')[0]
        results[split] = []
        trials[split] = trial_results
    splits = {'tests_seen': 1533, 'tests_unseen': 1529}
    assert all(split in trials for split in splits)

    time_string = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for model_path in model_paths:
        for split, split_len in splits.items():
            model_name = os.path.basename(model_path)
            log_entries = trials[split][model_name]['task']
            assert len(log_entries) == split_len
            for log_name, log_entry in log_entries.items():
                assert split == log_name.split('/')[0]
                results[split].append(log_entry['actseq'])

        save_path = os.path.join(
            os.path.dirname(model_path),
            'tests_{}_{}.json'.format(
                model_name.split('.')[0],
                time_string))
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)
        print('Results are saved to {}'.format(save_path))


@ex.automain
def main(eval, exp):
    args = helper_util.AttrDict(**eval, **exp)
    args.dout = os.path.join(constants.ET_LOGS, args.exp)
    os.makedirs(args.dout, exist_ok=True)
    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    results_paths = []
    for split in ('tests_seen', 'tests_unseen'):
        args.split = split
        # create a queue of trials to be performed and a logging queue
        model_paths = eval_util.get_model_paths(args)
        master = EvalMaster(args, model_paths[0])
        trial_queue, log_queue = master.create_queues(model_paths)
        if trial_queue.qsize() > 0:
            # start the evaluation
            if args.num_workers > 0:
                # start threads
                workers = master.launch_workers(evaluate_test, trial_queue, log_queue)
            else:
                # debug mode
                eval_util.worker_loop(evaluate_test, master.dataset,
                            trial_queue, log_queue, master.args)
                workers = []
            # wait for workers results and log them
            master.gather_results(workers, log_queue, test_split=True)
        results_paths.append(master.results_path)

    # parse what the threads have computed and save the results
    process_results(results_paths, model_paths)
