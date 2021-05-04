import os
import sys
import json
import collections

from datetime import datetime

from alfred.utils import eval_util


def compute_metrics(subgoal_success, subgoal_idx, reward, task, t_agent):
    '''
    compute metrics for subgoal evaluation
    '''
    pl = float(t_agent) + 1 # +1 for last action
    expert_pl = len([ll for ll in task['plan']['low_actions']
                     if ll['high_idx'] == subgoal_idx])
    s_spl = (1 if subgoal_success else 0) * min(
        1., expert_pl / (pl + sys.float_info.epsilon))
    plw_s_spl = s_spl * expert_pl
    metrics = {'success_spl': float(s_spl),
               'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
               'subgoal_path_len_weight': float(expert_pl),
               'reward': float(reward),
               'success': subgoal_success}
    return metrics


def evaluate_subgoals(
        env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor):
    # load trajectory data from the dataset
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx, subgoal_idx = int(trial_uid.split(':')[1]), int(trial_uid.split(':')[2])
    assert traj_data['repeat_idx'] == r_idx
    # reset model and setup scene
    model.reset()
    eval_util.setup_scene(env, traj_data, reward_type='dense')
    vocab = {'word': dataset.vocab_in, 'action_low': model.vocab_out}
    # load language features, expert initialization and task info
    input_dict = eval_util.load_language(
        dataset, traj_data, traj_key, model.args, extractor, subgoal_idx)
    expert_dict = eval_util.load_expert_actions(
        dataset, traj_data, traj_key, subgoal_idx)
    task_info = eval_util.read_task_data(traj_data, subgoal_idx)
    if args.debug:
        print(task_info)

    prev_action, subgoal_success, init_failed = None, False, False

    # expert teacher-forcing upto subgoal, get expert action
    for a_expert in expert_dict['actions']:
        input_dict['frames'] = eval_util.get_observation(env.last_event, extractor)
        init_failed, prev_action = eval_util.expert_step(
            a_expert['action'], expert_dict['masks'], model,
            input_dict, vocab, prev_action, env, args)
        if init_failed:
            break

    t_agent, t_expert, num_fails, reward = 0, len(expert_dict['actions']), 0, 0
    if not init_failed:
        # this should be set during the teacher-forcing but sometimes it fails
        env.task.goal_idx = task_info['subgoal_idx']
        env.task.finished = task_info['subgoal_idx'] - 1
        while t_agent < args.max_steps:
            # get an observation and do an agent step
            input_dict['frames'] = eval_util.get_observation(env.last_event, extractor)
            episode_end, prev_action, num_fails, _, _ = eval_util.agent_step(
                model, input_dict, vocab, prev_action, env, args,
                num_fails, obj_predictor)
            # get rewards and subgoal success
            reward += env.get_transition_reward()[0]
            subgoal_success = (env.get_subgoal_idx() == subgoal_idx)
            t_agent += 1
            # break if stop is predicted, args.max_fails is reached or success
            if episode_end or subgoal_success:
                break

    # compute metrics and dump a video
    metrics = compute_metrics(subgoal_success, subgoal_idx, reward, traj_data, t_agent)
    return dict(**metrics, **task_info)


def process_eval_subgoals(results_path, model_paths, args):
    print('Processing evaluation results')
    with open(results_path, 'r') as results_file:
        trial_results = json.load(results_file)

    for model_path in model_paths:
        successes, failures, results = collections.defaultdict(list), collections.defaultdict(list), {}

        # collect all entries from the trial_results
        log_entries = trial_results[os.path.basename(model_path)]['subgoal'].values()
        for log_entry in log_entries:
            subgoal_action = log_entry['subgoal_action']
            if log_entry['success']:
                successes[subgoal_action].append(log_entry)
            else:
                failures[subgoal_action].append(log_entry)

        # save results
        print('Model = {}'.format(os.path.basename(model_path)))
        print("-------------")
        subgoals_to_evaluate = list(successes.keys())
        subgoals_to_evaluate.sort()
        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(successes[sg]), len(failures[sg])
            num_evals = len(successes[sg]) + len(failures[sg])
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum(
                    [entry['subgoal_path_len_weight'] for entry in successes[sg]]) + \
                    sum([entry['subgoal_path_len_weight'] for entry in failures[sg]])
                sr_plw = float(
                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in successes[sg]]) +
                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in failures[sg]]))
                sr_plw /= total_path_len_weight
                results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw
                }
                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        print("------------")

        # record everything
        results = {'successes': {k: v for k, v in successes.items() if len(v) > 0},
                   'failures': {k: v for k, v in failures.items() if len(v) > 0},
                   'results': results,
                   'model': model_path,
                   'timestamp': datetime.now().strftime("%d.%m.%Y_%H:%M:%S_%f")}
        dataset_id = os.path.basename(results_path).split('.')[0]
        eval_util.overwrite_eval_json(results, model_path, 'subgoal', args, dataset_id)
