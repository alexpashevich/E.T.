import os
import torch
import json
import shutil
import pickle
import threading
import copy

from progressbar import ProgressBar
from pathlib import Path
from sacred import Ingredient, Experiment
from vocab import Vocab

from alfred.gen import constants
from alfred.nn.enc_visual import FeatureExtractor
from alfred.data.preprocessor import Preprocessor
from alfred.utils import data_util, helper_util, model_util


args_ingredient = Ingredient('args')
ex = Experiment('create_data', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # name of the output dataset
    data_output = 'lmdb_2.1.0'
    # where to load the original ALFRED dataset images and jsons from
    data_input = 'generated_2.1.0'
    # whether to overwrite old data in case it exists
    overwrite = False
    # number of processes to run the data processing in (0 for main thread)
    num_workers = 4
    # debug run with only 16 entries
    fast_epoch = False

    # VISUAL FEATURES SETTINGS
    # visual archi (resnet18, fasterrcnn, maskrcnn)
    visual_archi = 'fasterrcnn'
    # where to load a pretrained model from
    visual_checkpoint = None
    # which images to use (by default: RGBs)
    image_folder = 'raw_images'
    # feature compression
    compress_type = '4x'
    # which device to use
    device = 'cuda'

    # LANGUAGE ANNOTATIONS SETTINGS
    # generate dataset with subgoal annotations instead of human annotations
    subgoal_ann = False
    # use an existing vocabulary if specified (None for starting from scratch)
    vocab_path = 'files/base.vocab'


def process_feats(traj_paths, extractor, lock, image_folder, save_path):
    (save_path / 'feats').mkdir(exist_ok=True)
    if str(save_path).endswith('/worker00'):
        with lock:
            progressbar = ProgressBar(max_value=traj_paths.qsize())
            progressbar.start()
    while True:
        with lock:
            if traj_paths.qsize() == 0:
                break
            traj_path = Path(traj_paths.get())
        filename_new = '{}:{}:{}.pt'.format(
            *[traj_path.parents[i].name for i in (2, 1, 0)])
        # extract features with th extractor
        images = data_util.read_traj_images(traj_path, image_folder)
        feat = data_util.extract_features(images, extractor)
        if feat is not None:
            torch.save(feat, save_path / 'feats' / filename_new)
        with lock:
            with open(save_path.parents[0] / 'processed_feats.txt', 'a') as f:
                f.write(str(traj_path) + '\n')
            model_util.update_log(
                save_path.parents[0], stage='feats', update='increase', progress=1)
            if str(save_path).endswith('/worker00'):
                progressbar.update(progressbar.max_value - traj_paths.qsize())
    if str(save_path).endswith('/worker00'):
        progressbar.finish()


def process_jsons(traj_paths, preprocessor, lock, save_path):
    save_path.mkdir(exist_ok=True)
    (save_path / 'masks').mkdir(exist_ok=True)
    (save_path / 'jsons').mkdir(exist_ok=True)
    if str(save_path).endswith('/worker00'):
        with lock:
            progressbar = ProgressBar(max_value=len(traj_paths))
            progressbar.start()
    while True:
        with lock:
            if len(traj_paths) == 0:
                break
            traj_path = Path(traj_paths.pop())
        with traj_path.open() as f:
            traj_orig = json.load(f)
        if 'turk_annotations' in traj_orig:
            # it's the original trajectory from ALFRED dataset
            num_annotations = len(traj_orig['turk_annotations']['anns'])
        else:
            # it's a generated trajectory (no human annotations are provided)
            num_annotations = 1
        if os.path.exists(str(traj_path).replace('traj_data.json', 'masks.pkl')):
            # maybe the masks were already extracted from jsons
            with open(str(traj_path).replace('traj_data.json', 'masks.pkl'), 'rb') as f:
                masks = pickle.load(f)
        else:
            masks = data_util.process_masks(traj_orig)
        trajs = []
        for r_idx in range(num_annotations):
            traj_proc = data_util.process_traj(
                copy.deepcopy(traj_orig), traj_path, r_idx, preprocessor)
            trajs.append(traj_proc)
        # save masks and traj jsons
        filename = '{}:{}:{}.pkl'.format(*[traj_path.parents[i].name for i in (2, 1, 0)])
        with (save_path / 'masks'/ filename).open('wb') as f:
            pickle.dump(masks, f)
        with (save_path / 'jsons'/ filename).open('wb') as f:
            pickle.dump(trajs, f)
        # report the progress
        with lock:
            model_util.update_log(save_path.parents[0], stage='jsons',
                       update='increase', progress=1)
            if str(save_path).endswith('/worker00'):
                progressbar.update(progressbar.max_value - len(traj_paths))
    if str(save_path).endswith('/worker00'):
        progressbar.finish()


def get_traj_paths(input_path, processed_files_path, fast_epoch):
    if (input_path / 'processed.txt').exists():
        # the dataset was generated locally
        with (input_path / 'processed.txt').open() as f:
            traj_paths = [line.strip() for line in f.readlines()]
            # traj_paths = [line.replace('traj_data.json', '') for line in traj_paths]
            traj_paths = [line.split(';')[0] for line in traj_paths
                          if line.split(';')[1] == '1']
            traj_paths = [str(input_path / line) for line in traj_paths]
    else:
        # the dataset was downloaded from ALFRED servers
        traj_paths_all = sorted([
            str(path) for path in input_path.glob('*/*/*/traj_data.json')])
        traj_paths = [path for path in traj_paths_all if 'tests_' not in path]
    if fast_epoch:
        traj_paths = traj_paths[::20]
    num_files = len(traj_paths)
    if processed_files_path is not None and processed_files_path.exists():
        if str(processed_files_path).endswith(constants.VOCAB_FILENAME):
            traj_paths = []
        else:
            with processed_files_path.open() as f:
                processed_files = set([line.strip() for line in f.readlines()])
            traj_paths = [traj for traj in traj_paths if traj not in processed_files]
    traj_paths = [Path(path) for path in traj_paths]
    return traj_paths, num_files


def run_in_parallel(func, num_workers, output_path, args, use_processes=False):
    if num_workers == 0:
        args.append(output_path / 'worker00')
        func(*args)
    else:
        threads = []
        for idx in range(num_workers):
            args_worker = copy.copy(args) + [output_path / 'worker{:02d}'.format(idx)]
            if not use_processes:
                ThreadClass = threading.Thread
            else:
                ThreadClass = torch.multiprocessing.Process
            thread = ThreadClass(target=func, args=args_worker)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()


def gather_data(output_path, num_workers):
    print('Start gathering data')
    for dirname in ('feats', 'masks', 'jsons'):
        if (output_path / dirname).is_dir():
            shutil.rmtree(output_path / dirname)
        (output_path / dirname).mkdir()
    for dirname in ('feats', 'masks', 'jsons'):
        for path_file in output_path.glob('worker*/{}/*'.format(dirname)):
            if path_file.stat().st_size == 0:
                continue
            path_symlink = output_path / dirname / path_file.name
            link_file = True
            if path_symlink.is_symlink():
                # this file was already linked
                if path_file.stat().st_size > path_symlink.stat().st_size:
                    # we should replace the previously linked file with a new one
                    link_file = True
                    path_symlink.unlink()
                else:
                    # we should keep the previously linked file
                    link_file = False
            if link_file:
                path_symlink.symlink_to(path_file)

    partitions = ('train', 'valid_seen', 'valid_unseen')
    if not (output_path / '.deleting_worker_dirs').exists():
        for partition in partitions:
            print('Processing {} trajectories'.format(partition))
            feats_files = output_path.glob('feats/{}:*.pt'.format(partition))
            feats_files = sorted([str(path) for path in feats_files])
            masks_files = [p.replace('/feats/', '/masks/').replace('.pt', '.pkl')
                           for p in feats_files]
            jsons_files = [p.replace('/feats/', '/jsons/').replace('.pt', '.pkl')
                           for p in feats_files]
            (output_path / partition).mkdir(exist_ok=True)
            data_util.gather_feats(feats_files, output_path / partition / 'feats')
            data_util.gather_masks(masks_files, output_path / partition / 'masks')
            data_util.gather_jsons(jsons_files, output_path / partition / 'jsons.pkl')

    print('Removing worker directories')
    (output_path / '.deleting_worker_dirs').touch()
    for worker_idx in range(max(num_workers, 1)):
        worker_dir = output_path / 'worker{:02d}'.format(worker_idx)
        shutil.rmtree(worker_dir)
    for dirname in ('feats', 'masks', 'jsons'):
        shutil.rmtree(output_path / dirname)
    os.remove(output_path / '.deleting_worker_dirs')
    os.remove(output_path / 'processed_feats.txt')


@ex.automain
def main(args):
    torch.multiprocessing.set_start_method('spawn')
    args = helper_util.AttrDict(**args)
    if args.data_output is None:
        raise RuntimeError('Please, specify the name of output dataset')

    # set up the paths
    output_path = Path(constants.ET_DATA) / args.data_output
    input_path = Path(constants.ET_DATA) / args.data_input
    print('Creating a dataset {} using data from {}'.format(
        args.data_output, input_path))
    if not input_path.is_dir():
        raise RuntimeError('The input dataset {} does not exist'.format(input_path))
    if output_path.is_dir() and args.overwrite:
        print('Erasing the old directory')
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)

    # read which files need to be processed
    trajs_list, num_files = get_traj_paths(
        input_path, output_path / constants.VOCAB_FILENAME, args.fast_epoch)
    model_util.save_log(output_path, progress=num_files - len(trajs_list),
             total=num_files, stage='jsons')
    print('Creating a dataset with {} trajectories using {} workers'.format(
        num_files, args.num_workers))
    print('Processing JSONs and masks ({} were already processed)'.format(
        num_files - len(trajs_list)))

    # first process jsons and masks
    if len(trajs_list) > 0:
        lock = threading.Lock()
        preprocessor = data_util.get_preprocessor(
            Preprocessor, args.subgoal_ann, lock, args.vocab_path)
        run_in_parallel(
            process_jsons, args.num_workers, output_path,
            args=[trajs_list, preprocessor, lock])
        vocab_copy = {}
        for key, vocab in preprocessor.vocab.items():
            vocab_copy[key] = Vocab.from_dict(vocab.to_dict())
        torch.save(vocab_copy, output_path / constants.VOCAB_FILENAME)

    # read which features need to be extracted
    trajs_list, num_files_again = get_traj_paths(
        input_path, output_path / 'processed_feats.txt', args.fast_epoch)
    assert num_files == num_files_again
    model_util.save_log(output_path, progress=num_files - len(trajs_list),
             total=num_files, stage='feats')
    print('Extracting features ({} were already processed)'.format(
        num_files - len(trajs_list)))

    # then extract features
    extractor = FeatureExtractor(
        args.visual_archi, args.device, args.visual_checkpoint,
        share_memory=True, compress_type=args.compress_type)
    if len(trajs_list) > 0:
        manager = torch.multiprocessing.Manager()
        lock = manager.Lock()
        trajs_queue = manager.Queue()
        for path in trajs_list:
            trajs_queue.put(path)
        args_process_feats = [trajs_queue, extractor, lock, args.image_folder]
        run_in_parallel(
            process_feats, args.num_workers, output_path,
            args=args_process_feats, use_processes=True)

    # finally, gather all the data
    gather_data(output_path, args.num_workers)
    # save dataset info to a file
    params = {
        'feat_shape': extractor.feat_shape,
        'visual_checkpoint': args.visual_checkpoint,
        'visual_archi': args.visual_archi,
        'compress_type': args.compress_type}
    with (output_path / 'params.json').open('w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    print('The dataset was saved to {}'.format(output_path))
