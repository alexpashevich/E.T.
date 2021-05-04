import os
import json
import queue
import torch
import shutil
import filelock
import numpy as np

from PIL import Image
from termcolor import colored

from alfred.gen import constants
from alfred.env.thor_env import ThorEnv
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, model_util


def setup_scene(env, traj_data, reward_type='dense', test_split=False):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name, silent=True)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))
    # setup task for reward
    if not test_split:
        env.set_task(traj_data, reward_type=reward_type)


def load_agent(model_path, dataset_info, device):
    '''
    load a pretrained agent and its feature extractor
    '''
    learned_model, _ = model_util.load_model(model_path, device)
    model = learned_model.model
    model.eval()
    model.args.device = device
    extractor = FeatureExtractor(
        archi=dataset_info['visual_archi'],
        device=device,
        checkpoint=dataset_info['visual_checkpoint'],
        compress_type=dataset_info['compress_type'])
    return model, extractor


def load_object_predictor(args):
    if args.object_predictor is None:
        return None
    return FeatureExtractor(
        archi='maskrcnn', device=args.device,
        checkpoint=args.object_predictor, load_heads=True)


def worker_loop(
        evaluate_function,
        dataset,
        trial_queue,
        log_queue,
        args,
        cuda_device=None):
    '''
    evaluation loop
    '''
    if cuda_device:
        torch.cuda.set_device(cuda_device)
        args.device = 'cuda:{}'.format(cuda_device)
    # start THOR
    env = ThorEnv(x_display=args.x_display)
    # master may ask to evaluate different models
    model_path_loaded = None
    object_predictor = load_object_predictor(args)

    if args.num_workers == 0:
        num_success, num_trials_done, num_trials = 0, 0, trial_queue.qsize()
    try:
        while True:
            trial_uid, dataset_idx, model_path = trial_queue.get(timeout=3)
            if model_path != model_path_loaded:
                if model_path_loaded is not None:
                    del model, extractor
                    torch.cuda.empty_cache()
                model, extractor = load_agent(
                    model_path,
                    dataset.dataset_info,
                    args.device)
                dataset.vocab_translate = model.vocab_out
                model_path_loaded = model_path
            log_entry = evaluate_function(
                env, model, dataset, extractor, trial_uid, dataset_idx, args,
                object_predictor)
            if (args.debug or args.num_workers == 0) and 'success' in log_entry:
                if 'subgoal_action' in log_entry:
                    trial_type = log_entry['subgoal_action']
                else:
                    trial_type = 'full task'
                print(colored('Trial {}: {} ({})'.format(
                    trial_uid, 'success' if log_entry['success'] else 'fail',
                    trial_type), 'green' if log_entry['success'] else 'red'))
            if args.num_workers == 0 and 'success' in log_entry:
                num_trials_done += 1
                num_success += int(log_entry['success'])
                print('{:4d}/{} trials are done (current SR = {:.1f})'.format(
                    num_trials_done, num_trials, 100 * num_success / num_trials_done))
            log_queue.put((log_entry, trial_uid, model_path))
    except queue.Empty:
        pass
    # stop THOR
    env.stop()


def get_model_paths(args):
    '''
    check which models need to be evaluated
    '''
    model_paths = []
    if args.eval_range is None:
        # evaluate only the latest checkpoint
        model_paths.append(
            os.path.join(constants.ET_LOGS, args.exp, args.checkpoint))
    else:
        # evaluate a range of epochs
        for model_epoch in range(*args.eval_range):
            model_path = os.path.join(
                constants.ET_LOGS, args.exp,
                'model_{:02d}.pth'.format(model_epoch))
            if os.path.exists(model_path):
                model_paths.append(model_path)
    for idx, model_path in enumerate(model_paths):
        if os.path.islink(model_path):
            model_paths[idx] = os.readlink(model_path)
    if len(model_paths) == 0:
        raise ValueError('No models are found for evaluation')
    return model_paths


def get_sr(eval_json, eval_type):
    if eval_type == 'task':
        return eval_json['results']['all']['success']['success_rate']
    sr_sum, sr_count = 0, 0
    for _, sg_dict in sorted(eval_json['results'].items()):
        sr_sum += sg_dict['successes']
        sr_count += sg_dict['evals']
    return sr_sum / sr_count


def overwrite_eval_json(results, model_path, eval_type, args, dataset_id):
    '''
    append results to an existing eval.json or create a new one
    '''
    # eval_json: eval_epoch / eval_split / {'subgoal','task'} / {'normal','fast_epoch'}
    # see if eval.json file alredy existed
    eval_json = {}
    eval_json_path = os.path.join(os.path.dirname(model_path), 'eval.json')
    lock = filelock.FileLock(eval_json_path + '.lock')
    with lock:
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r') as eval_json_file:
                eval_json = json.load(eval_json_file)
    eval_epoch = os.path.basename(model_path)
    if eval_epoch not in eval_json:
        eval_json[eval_epoch] = {}
    if dataset_id not in eval_json[eval_epoch]:
        eval_json[eval_epoch][dataset_id] = {}
    if eval_type not in eval_json[eval_epoch][dataset_id]:
        eval_json[eval_epoch][dataset_id][eval_type] = {}
    eval_mode = 'normal' if not args.fast_epoch else 'fast_epoch'
    if eval_mode in eval_json[eval_epoch][dataset_id][eval_type]:
        print('WARNING: the evaluation was already done')
        sr_new = len(results['successes']) / (len(results['successes']) + len(results['failures']))
        prev_res = eval_json[eval_epoch][dataset_id][eval_type][eval_mode]
        sr_old = len(prev_res['successes']) / (len(prev_res['successes']) + len(prev_res['failures']))
        print('Previous success rate = {:.1f}, new success rate = {:.1f}'.format(
            100 * sr_old, 100 * sr_new))
    eval_json[eval_epoch][dataset_id][eval_type][eval_mode] = results
    # make a backup copy of eval.json file before writing
    save_with_backup(eval_json, eval_json_path, lock)
    print('Evaluation is saved to {}'.format(eval_json_path))


def save_with_backup(obj, file_path, lock):
    with lock:
        # make a backup copy of results file before writing and swipe it after
        file_path_back = file_path + '.back'
        file_path_back_back = file_path + '.back.back'
        # always remove the second backup
        if os.path.exists(file_path_back_back):
            os.remove(file_path_back_back)
        # rename the first backup to the second one
        if os.path.exists(file_path_back):
            os.rename(file_path_back, file_path_back_back)
        # put the original file as the first backup
        if os.path.exists(file_path):
            os.rename(file_path, file_path_back)
        # write the updated content to the file path
        with open(file_path, 'w') as file_opened:
            json.dump(obj, file_opened, indent=2, sort_keys=True)


def disable_lock_logs():
    lock_logger = filelock.logger()
    lock_logger.setLevel(30)


def extract_rcnn_pred(class_idx, obj_predictor, env, verbose=False):
    '''
    extract a pixel mask using a pre-trained MaskRCNN
    '''
    rcnn_pred = obj_predictor.predict_objects(Image.fromarray(env.last_event.frame))
    class_name = obj_predictor.vocab_obj.index2word(class_idx)
    candidates = list(filter(lambda p: p.label == class_name, rcnn_pred))
    if verbose:
        visible_objs = [
            obj for obj in env.last_event.metadata['objects']
            if obj['visible'] and obj['objectId'].startswith(class_name + '|')]
        print('Agent prediction = {}, detected {} objects (visible {})'.format(
            class_name, len(candidates), len(visible_objs)))
    if len(candidates) > 0:
        if env.last_interaction[0] == class_idx:
            # last_obj['id'] and class_name + '|' in env.last_obj['id']:
            # do the association based selection
            last_center = np.array(env.last_interaction[1].nonzero()).mean(axis=1)
            cur_centers = np.array(
                [np.array(c.mask[0].nonzero()).mean(axis=1) for c in candidates])
            distances = ((cur_centers - last_center)**2).sum(axis=1)
            index = np.argmin(distances)
            mask = candidates[index].mask[0]
        else:
            # do the confidence based selection
            index = np.argmax([p.score for p in candidates])
            mask = candidates[index].mask[0]
    else:
        mask = None
    return mask


def agent_step(
        model, input_dict, vocab, prev_action, env, args, num_fails, obj_predictor):
    '''
    environment step based on model prediction
    '''
    # forward model
    with torch.no_grad():
        m_out = model.step(input_dict, vocab, prev_action=prev_action)
    m_pred = model_util.extract_action_preds(
        m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
    action = m_pred['action']
    if args.debug:
        print("Predicted action: {}".format(action))

    mask = None
    obj = m_pred['object'][0][0] if model_util.has_interaction(action) else None
    if obj is not None:
        # get mask from a pre-trained RCNN
        assert obj_predictor is not None
        mask = extract_rcnn_pred(
            obj, obj_predictor, env, args.debug)
        m_pred['mask_rcnn'] = mask
    # remove blocking actions
    action = obstruction_detection(
        action, env, m_out, model.vocab_out, args.debug)
    m_pred['action'] = action

    # use the predicted action
    episode_end = (action == constants.STOP_TOKEN)
    api_action = None
    # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
    target_instance_id = ''
    if not episode_end:
        step_success, _, target_instance_id, err, api_action = env.va_interact(
            action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
        env.last_interaction = (obj, mask)
        if not step_success:
            num_fails += 1
            if num_fails >= args.max_fails:
                if args.debug:
                    print("Interact API failed {} times; latest error '{}'".format(
                        num_fails, err))
                episode_end = True
    return episode_end, str(action), num_fails, target_instance_id, api_action


def expert_step(action, masks, model, input_dict, vocab, prev_action, env, args):
    '''
    environment step based on expert action
    '''
    mask = masks.pop(0).float().numpy()[0] if model_util.has_interaction(
        action) else None
    # forward model
    if not args.no_model_unroll:
        with torch.no_grad():
            model.step(input_dict, vocab, prev_action=prev_action)
        prev_action = (action if not args.no_teacher_force else None)
    # execute expert action
    step_success, _, _, err, _ = env.va_interact(
        action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
    if not step_success:
        print("expert initialization failed")
        return True, prev_action
    # update transition reward
    _, _ = env.get_transition_reward()
    return False, prev_action


def get_observation(event, extractor):
    '''
    get environment observation
    '''
    frames = extractor.featurize([Image.fromarray(event.frame)], batch=1)
    return frames


def load_language(
        dataset, task, dataset_key, model_args, extractor, subgoal_idx=None,
        test_split=False):
    '''
    load language features from the dataset and unit-test the feature extractor
    '''
    # load language features
    if subgoal_idx is not None:
        feat_args = (task, subgoal_idx, subgoal_idx + 1)
    else:
        feat_args = (task,)
    feat_numpy = dataset.load_features(*feat_args)
    # test extractor with the frames
    if not test_split:
        frames_expert = dataset.load_frames(dataset_key)
        model_util.test_extractor(task['root'], extractor, frames_expert)
    if not test_split and 'frames' in dataset.ann_type:
        # frames will be used as annotations
        feat_numpy['frames'] = frames_expert
    _, input_dict, _ = data_util.tensorize_and_pad(
        [(task, feat_numpy)], model_args.device, dataset.pad)
    return input_dict


def load_expert_actions(dataset, task, dataset_key, subgoal_idx):
    '''
    load actions and masks for expert initialization
    '''
    expert_dict = dict()
    expert_dict['actions'] = [
        a['discrete_action'] for a in task['plan']['low_actions']
        if a['high_idx'] < subgoal_idx]
    expert_dict['masks'] = dataset.load_masks(dataset_key)
    return expert_dict


def read_task_data(task, subgoal_idx=None):
    '''
    read data from the traj_json
    '''
    # read general task info
    repeat_idx = task['repeat_idx']
    task_dict = {'repeat_idx': repeat_idx,
                 'type': task['task_type'],
                 'task': '/'.join(task['root'].split('/')[-3:-1])}
    # read subgoal info
    if subgoal_idx is not None:
        task_dict['subgoal_idx'] = subgoal_idx
        task_dict['subgoal_action'] = task['plan']['high_pddl'][
            subgoal_idx]['discrete_action']['action']
    return task_dict


def obstruction_detection(action, env, m_out, vocab_out, verbose):
    '''
    change 'MoveAhead' action to a turn in case if it has failed previously
    '''
    if action != 'MoveAhead_25':
        return action
    if env.last_event.metadata['lastActionSuccess']:
        return action
    dist_action = m_out['action'][0][0].detach().cpu()
    idx_rotateR = vocab_out.word2index('RotateRight_90')
    idx_rotateL = vocab_out.word2index('RotateLeft_90')
    action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'
    if verbose:
        print("Blocking action is changed to: {}".format(action))
    return action
