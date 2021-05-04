import os
import json
import random
import time
import torch
import filelock
import torch.multiprocessing as mp

from termcolor import colored
from etaprogress.eta import ETA

from alfred.data import AlfredDataset
from alfred.gen import constants
from alfred.utils import eval_util, model_util


class EvalMaster(object):
    def __init__(self, args, model_path):
        self.args = args
        # multiprocessing settings
        mp.set_start_method('spawn', force=True)
        self.manager = mp.Manager()
        # disable annoying locks of FileLocks
        eval_util.disable_lock_logs()
        # create the dataset (for future data loading)
        model_args = model_util.load_model_args(model_path)
        assert isinstance(model_args.data['train'], list)
        if len(args.data['valid']) > 0:
            data_name = args.data['valid']
        else:
            assert len(model_args.data['train']) == 1
            data_name = model_args.data['train'][0]
        # we will take care of it while filling the evaluation queue
        model_args.fast_epoch = False
        model_args.data['length'] = 0
        if ('ann_type' in model_args['data'] and
            args.data['ann_type'] not in model_args['data']['ann_type']):
            raise RuntimeError('Model {} was not trained on {}'.format(
                model_path, args.data['ann_type']))
        self.dataset = AlfredDataset(
            data_name, self.args.split, model_args, args.data['ann_type'])
        self.dataset.test_mode = True

        # set random seed for shuffling
        random.seed(int(time.time()))
        self.results_path = os.path.join(
            self.args.dout, '{}.json'.format(self.dataset.id.replace('/', ':')))

    def create_queues(self, model_paths):
        '''
        create a queue of trajectories to be evaluated and a queue to log into
        '''
        # if subgoal evalution is performed, parse which subgoals should be evaluated
        self._test = torch.zeros(100).to(self.args.device)
        if self.args.subgoals:
            use_subgoal_indices = all([c.isdecimal() for c in self.args.subgoals.split(',')])
            if use_subgoal_indices:
                # if subgoals_to_evaluate contains subgoals indices, e.g. '0,1' or '0'
                subgoals_to_evaluate = [int(c) for c in self.args.subgoals.split(',')]
            else:
                # if the subgoals are defined with their names
                if self.args.subgoals.lower() == 'all':
                    subgoals_to_evaluate = constants.ALL_SUBGOALS
                else:
                    subgoals_to_evaluate = self.args.subgoals.split(',')
                # assert that all subgoals are correctly specified
                assert all([sg in constants.ALL_SUBGOALS for sg in subgoals_to_evaluate])
            if self.args.subgoals.lower() != 'all':
                print(colored('Subgoals to evaluate: {}'.format(
                    str(subgoals_to_evaluate)), 'yellow'))

        # get all the trajectories in the form of 'task_id:repeat_id[:subgoal_id]'
        trials = []
        # in case if we evaluate on train, do not evaluate more trajectories than in eval
        max_trials_num = 820 if 'tests' not in self.dataset.partition else 9999999
        for dataset_idx, (task_json, unused_dataset_key) in enumerate(
                self.dataset.jsons_and_keys[:max_trials_num]):
            trial_uid = '{}:{}'.format(task_json['task'], task_json['repeat_idx'])
            if not self.args.subgoals:
                trials.append((trial_uid, dataset_idx))
            else:
                if use_subgoal_indices:
                    subgoal_idxs = subgoals_to_evaluate
                else:
                    subgoal_idxs = [
                        sg['high_idx'] for sg in task_json['plan']['high_pddl']
                        if sg['discrete_action']['action'] in subgoals_to_evaluate]
                trials.extend([('{}:{}'.format(trial_uid, idx), dataset_idx)
                               for idx in subgoal_idxs])

        if self.args.fast_epoch:
            # evaluate only 16 trials
            trials = trials[:16]
        print(colored('Number of trials in the evaluation: {}'.format(
            len(trials) * len(model_paths)), 'yellow'))

        # tell to workers which models have to evaluated
        self.num_trials, self.num_trials_done = len(trials) * len(model_paths), 0
        trials_and_models = []
        if os.path.exists(self.results_path):
            # check which trajs were already processed
            with open(self.results_path, 'r') as results_file:
                results = json.load(results_file)
            for model_path in model_paths:
                eval_epoch = os.path.basename(model_path)
                if eval_epoch in results \
                   and ('subgoal' if self.args.subgoals else 'task') in results[eval_epoch]:
                    trials_done = results[eval_epoch][
                        'subgoal' if self.args.subgoals else 'task']
                    print(colored('Found {} evaluated tials for {}'.format(
                        len(trials_done), eval_epoch), 'yellow'))
                else:
                    trials_done = []
                trials_and_models.extend([
                    (trial_uid, dataset_idx, model_path) for trial_uid, dataset_idx in trials
                    if trial_uid not in trials_done])
            self.num_trials_done = len(trials) * len(model_paths) - len(trials_and_models)
            print(colored('Number of trials to be executed: {}'.format(
                len(trials_and_models)), 'yellow'))
        else:
            # multiple trials by model_paths
            for model_path in model_paths:
                for trial_uid, dataset_idx in trials:
                    trials_and_models.append((trial_uid, dataset_idx, model_path))

        # put the trials into a distrubuted queue
        trial_queue = self.manager.Queue()
        for trial_uid, dataset_idx, model_path in trials_and_models:
            trial_queue.put((trial_uid, dataset_idx, model_path))
        # report the eval stage
        model_util.save_log(
            self.args.dout, progress=self.num_trials_done,
            total=self.num_trials, stage='eval')
        log_queue = self.manager.Queue()
        return trial_queue, log_queue

    def launch_workers(self, evaluate_function, trial_queue, log_queue):
        '''
        spawn multiple threads to run eval in parallel
        '''
        num_workers = min(self.args.num_workers, trial_queue.qsize())
        print(colored('Evaluating using {} workers'.format(num_workers), 'yellow'))
        # start threads
        workers = []
        cuda_device = None
        for worker_idx in range(num_workers):
            num_workers_per_gpu = constants.NUM_EVAL_WORKERS_PER_GPU
            if torch.cuda.device_count() > 1 and num_workers > num_workers_per_gpu:
                assert num_workers <= num_workers_per_gpu * torch.cuda.device_count()
                cuda_device = worker_idx // num_workers_per_gpu
            worker = mp.Process(target=eval_util.worker_loop, args=(
                evaluate_function,
                self.dataset, trial_queue, log_queue,
                self.args, cuda_device))
            worker.start()
            workers.append(worker)
        return workers

    def gather_results(self, workers, log_queue, test_split=False):
        '''
        check for logs while waiting for workers
        '''
        results = {}
        eval_type = 'subgoal' if self.args.subgoals else 'task'
        lock = filelock.FileLock(self.results_path + '.lock')
        eta = ETA(self.num_trials, scope=32)
        while True:
            if log_queue.qsize() > 0:
                # there is a new log entry available, process it
                log_entry, trial_uid, model_path = log_queue.get()
                # load old results (if available)
                with lock:
                    if os.path.exists(self.results_path):
                        with open(self.results_path, 'r') as results_file:
                            results = json.load(results_file)

                eval_epoch = os.path.basename(model_path)
                # update the old results with the new log entry
                if eval_epoch not in results:
                    results[eval_epoch] = {}
                if eval_type not in results[eval_epoch]:
                    results[eval_epoch][eval_type] = {}
                if trial_uid in results[eval_epoch][eval_type] and not test_split:
                    success_prev = results[eval_epoch][eval_type][trial_uid]['success']
                    success_curr = log_entry['success']
                    if success_prev != success_curr:
                        print(colored(
                            'WARNING: trial {} result has changed from {} to {}'.format(
                                trial_uid,
                                'success' if success_prev else 'fail',
                                'success' if success_curr else 'fail'), 'yellow'))
                results[eval_epoch][eval_type][trial_uid] = log_entry

                # print updated results
                self.num_trials_done += 1
                eta.numerator = self.num_trials_done
                if not test_split:
                    successes = [
                        log['success'] for log in results[eval_epoch][eval_type].values()]
                    print(colored(
                        '{:4d}/{} trials are done (current SR = {:.1f}), ETA = {}, elapsed = {}'.format(
                            self.num_trials_done,
                            self.num_trials,
                            100 * sum(successes) / len(successes),
                            time.strftime('%H:%M:%S', time.gmtime(eta.eta_seconds)),
                            time.strftime('%H:%M:%S', time.gmtime(eta.elapsed))),
                        'green'))
                # make a backup copy of results file before writing
                eval_util.save_with_backup(results, self.results_path, lock)
                # update info.json file
                model_util.update_log(
                    self.args.dout, stage='eval',
                    update='increase', progress=1)

            # check whether all workers have exited (exitcode == None means they are still running)
            all_terminated = all([worker.exitcode is not None for worker in workers])
            if all_terminated and log_queue.qsize() == 0:
                if self.num_trials_left > 0:
                    print(colored('WARNING: only {}/{} trials were evaluated'.format(
                        self.num_trials_done, self.num_trials), 'red'))
                # our mission is over
                break
            time.sleep(1)
        print(colored('Evaluation is complete', 'green'))

    @property
    def num_trials_left(self):
        return self.num_trials - self.num_trials_done
