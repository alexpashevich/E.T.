import os
import torch
import types
import json
import copy
import pickle

from tqdm import tqdm
from pathlib import Path
from sacred import Ingredient, Experiment

from alfred.data.preprocessor import Preprocessor
from alfred.gen import constants
from alfred.utils import data_util

args_ingredient = Ingredient('args')
ex = Experiment('process_tests', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # where to load the original dataset from
    data_input = None
    # where to save the new dataset
    data_output = None

def get_traj_paths(input_path):
    traj_paths = sorted([
        str(path) for path in input_path.glob('*/traj_data.json')])
    traj_paths = [Path(path) for path in traj_paths]
    return traj_paths, len(traj_paths)


def process_jsons(traj_paths, preprocessor, save_path):
    # prepare
    save_path.mkdir(exist_ok=True)
    jsons_dict = {}
    # loop over trajectories
    for idx, traj_path in tqdm(enumerate(traj_paths), total=len(traj_paths)):
        traj_path = Path(traj_path)
        with traj_path.open() as f:
            traj_orig = json.load(f)
        key = '{:06}'.format(idx).encode('ascii')
        num_annotations = len(traj_orig['turk_annotations']['anns'])
        jsons_list = []
        for r_idx in range(num_annotations):
            traj_proc = data_util.process_traj(
                copy.deepcopy(traj_orig), traj_path, r_idx, preprocessor)
            jsons_list.append(traj_proc)
        jsons_dict[key] = jsons_list
    # save the result
    with (save_path / 'jsons.pkl').open('wb') as f:
        pickle.dump(jsons_dict, f)


@ex.automain
def main(args):
    torch.multiprocessing.set_start_method('spawn')
    args = types.SimpleNamespace(**args)

    # set up the paths
    output_path = Path(constants.ET_DATA) / args.data_output
    input_path = Path(constants.ET_DATA) / args.data_input
    print('Processing tests in {} using data from {}'.format(
        args.data_output, input_path))

    # load vocab
    vocab_path = output_path / constants.VOCAB_FILENAME
    assert output_path.is_dir() and input_path.is_dir() and vocab_path.exists()
    vocab = torch.load(vocab_path)
    print('Orig vocab: {}'.format(vocab))
    num_words_orig = len(vocab['word'])

    # create a preprocessor
    preprocessor = Preprocessor(vocab, is_test_split=True)

    for split in ('tests_seen', 'tests_unseen'):
        traj_paths, num_files = get_traj_paths(input_path / split)
        print('Found {} trajectories in {}'.format(num_files, split))
        save_path = output_path / split
        if len(traj_paths) > 0:
            process_jsons(traj_paths, preprocessor, save_path)
        print('Vocab after {}: {}'.format(split, preprocessor.vocab))
        if len(preprocessor.vocab['word']) != num_words_orig:
            raise ValueError('Unknown words were added')
