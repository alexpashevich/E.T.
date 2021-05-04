import os
import json
import torch
import numpy as np

from datetime import datetime

from alfred.utils import eval_util


def compute_metrics(success, reward, task, t, pcs):
    '''
    compute metrics for task evaluation
    '''
    # goal_conditions
    goal_condition_success_rate = pcs[0] / float(pcs[1])
    # SPL
    path_len_weight = len(task['plan']['low_actions'])
    s_spl = (1 if success else 0) * min(1., path_len_weight / float(t))
    pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))
    # path length weighted SPL
    plw_s_spl = s_spl * path_len_weight
    plw_pc_spl = pc_spl * path_len_weight
    metrics = {'completed_goal_conditions': int(pcs[0]),
               'total_goal_conditions': int(pcs[1]),
               'goal_condition_success': float(goal_condition_success_rate),
               'success_spl': float(s_spl),
               'path_len_weighted_success_spl': float(plw_s_spl),
               'goal_condition_spl': float(pc_spl),
               'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
               'path_len_weight': int(path_len_weight),
               'reward': float(reward),
               'success': success}
    return metrics


def evaluate_task(
        env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor):
    # load trajectory data from the dataset
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx = int(trial_uid.split(':')[1])
    assert traj_data['repeat_idx'] == r_idx
    # reset model and setup scene
    model.reset()
    eval_util.setup_scene(env, traj_data, reward_type='dense')
    vocab = {'word': dataset.vocab_in, 'action_low': model.vocab_out}
    # load language features and task info
    input_dict = eval_util.load_language(
        dataset, traj_data, traj_key, model.args, extractor)
    task_info = eval_util.read_task_data(traj_data)

    prev_action = None
    t, num_fails, reward = 0, 0, 0
    while t < args.max_steps:
        # get an observation and do an agent step
        input_dict['frames'] = eval_util.get_observation(env.last_event, extractor)
        episode_end, prev_action, num_fails, _, _ = eval_util.agent_step(
            model, input_dict, vocab, prev_action, env, args, num_fails, obj_predictor)
        # get rewards
        reward += env.get_transition_reward()[0]
        t += 1
        # break if stop is predicted or args.max_fails is reached
        if episode_end:
            break

    # compute metrics and dump a video
    success = env.get_goal_satisfied()
    metrics = compute_metrics(success, reward, traj_data, t, env.get_goal_conditions_met())
    return dict(**metrics, **task_info)


def get_metrics(successes, failures):
    '''
    compute overall succcess and goal_condition success rates along with path-weighted metrics
    '''
    # stats
    num_successes, num_failures = len(successes), len(failures)
    num_evals = len(successes) + len(failures)
    total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                            sum([entry['path_len_weight'] for entry in failures])
    completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                sum([entry['completed_goal_conditions'] for entry in failures])
    total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                            sum([entry['total_goal_conditions'] for entry in failures])

    # metrics
    sr = float(num_successes) / num_evals
    pc = completed_goal_conditions / float(total_goal_conditions)
    plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                total_path_len_weight)
    plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                total_path_len_weight)

    # result table
    res = dict()
    res['success'] = {'num_successes': num_successes,
                        'num_evals': num_evals,
                        'success_rate': sr}
    res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                    'total_goal_conditions': total_goal_conditions,
                                    'goal_condition_success_rate': pc}
    res['path_length_weighted_success_rate'] = plw_sr
    res['path_length_weighted_goal_condition_success_rate'] = plw_pc

    return res


def process_eval_task(results_path, model_paths, args):
    print('Processing evaluation results')
    with open(results_path, 'r') as results_file:
        trial_results = json.load(results_file)

    for model_path in model_paths:
        successes, failures, results = [], [], {}
        # collect all entries from the log_file
        log_entries = trial_results[os.path.basename(model_path)]['task'].values()
        for log_entry in log_entries:
            if log_entry['success']:
                successes.append(log_entry)
            else:
                failures.append(log_entry)

        # overall results
        results['all'] = get_metrics(successes, failures)
        print('Model = {}'.format(os.path.basename(model_path)))
        print("-------------")
        print("SR: %d/%d = %.3f" % (
            results['all']['success']['num_successes'],
            results['all']['success']['num_evals'],
            results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (
            results['all']['goal_condition_success']['completed_goal_conditions'],
            results['all']['goal_condition_success']['total_goal_conditions'],
            results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")
        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep',
                      'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep',
                      'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        # record everything
        results = {'successes': successes,
                   'failures': failures,
                   'results': results,
                   'model': model_path,
                   'timestamp': datetime.now().strftime("%d.%m.%Y_%H:%M:%S_%f")}
        dataset_id = os.path.basename(results_path).split('.')[0]
        eval_util.overwrite_eval_json(results, model_path, 'task', args, dataset_id)
