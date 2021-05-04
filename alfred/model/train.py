import os
import torch
import random
import shutil
import pprint
import numpy as np
from sacred import Experiment

from alfred.config import exp_ingredient, train_ingredient
from alfred.data import AlfredDataset, SpeakerDataset
from alfred.gen import constants
from alfred.model.learned import LearnedModel
from alfred.utils import data_util, helper_util, model_util

ex = Experiment('train', ingredients=[train_ingredient, exp_ingredient])


def prepare(train, exp):
    '''
    create logdirs, check dataset, seed pseudo-random generators
    '''
    # args and init
    args = helper_util.AttrDict(**train, **exp)
    args.dout = os.path.join(constants.ET_LOGS, args.name)
    args.data['train'] = args.data['train'].split(',')
    args.data['valid'] = args.data['valid'].split(',') if args.data['valid'] else []
    num_datas = len(args.data['train']) + len(args.data['valid'])
    for key in ('ann_type', ):
        args.data[key] = args.data[key].split(',')
        if len(args.data[key]) == 1:
            args.data[key] = args.data[key] * num_datas
        if len(args.data[key]) != num_datas:
            raise ValueError(
                'Provide either 1 {} or {} separated by commas'.format(key, num_datas))
    # set seeds
    torch.manual_seed(args.seed)
    random.seed(a=args.seed)
    np.random.seed(args.seed)
    # make output dir
    print(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args


def create_model(args, embs_ann, vocab_out):
    '''
    load a model and its optimizer
    '''
    prev_train_info = model_util.load_log(args.dout, stage='train')
    if args.resume and os.path.exists(os.path.join(args.dout, 'latest.pth')):
        # load a saved model
        loadpath = os.path.join(args.dout, 'latest.pth')
        model, optimizer = model_util.load_model(
            loadpath, args.device, prev_train_info['progress'] - 1)
        assert model.vocab_out.contains_same_content(vocab_out)
        model.args = args
    else:
        # create a new model
        if not args.resume and os.path.isdir(args.dout):
            shutil.rmtree(args.dout)
        model = LearnedModel(args, embs_ann, vocab_out)
        model = model.to(torch.device(args.device))
        optimizer = None
        if args.pretrained_path:
            if '/' not in args.pretrained_path:
                # a relative path at the logdir was specified
                args.pretrained_path = model_util.last_model_path(args.pretrained_path)
            print('Loading pretrained model from {}'.format(args.pretrained_path))
            pretrained_model = torch.load(
                args.pretrained_path, map_location=torch.device(args.device))
            model.load_state_dict(
                pretrained_model['model'], strict=False)
            loaded_keys = set(
                model.state_dict().keys()).intersection(
                    set(pretrained_model['model'].keys()))
            assert len(loaded_keys)
            print('Loaded keys:')
            pprint.pprint(loaded_keys)
    # put encoder on several GPUs if asked
    if torch.cuda.device_count() > 1:
        print('Parallelizing the model')
        model.model = helper_util.DataParallel(model.model)
    return model, optimizer, prev_train_info


def load_data(name, args, ann_type, valid_only=False):
    '''
    load dataset and wrap them into torch loaders
    '''
    partitions = ([] if valid_only else ['train']) + ['valid_seen', 'valid_unseen']
    datasets = []
    for partition in partitions:
        if args.model == 'speaker':
            dataset = SpeakerDataset(name, partition, args, ann_type)
        elif args.model == 'transformer':
            dataset = AlfredDataset(name, partition, args, ann_type)
        else:
            raise ValueError('Unknown model: {}'.format(args.model))
        datasets.append(dataset)
    return datasets


def wrap_datasets(datasets, args):
    '''
    wrap datasets with torch loaders
    '''
    batch_size = args.batch // len(args.data['train'])
    loader_args = {
        'num_workers': args.num_workers,
        'drop_last': (torch.cuda.device_count() > 1),
        'collate_fn': helper_util.identity}
    if args.num_workers > 0:
        # do not prefetch samples, this may speed up data loading
        loader_args['prefetch_factor'] = 1

    loaders = {}
    for dataset in datasets:
        if dataset.partition == 'train':
            weights = [1 / len(dataset)] * len(dataset)
            num_samples = 16 if args.fast_epoch else (
                args.data['length'] or len(dataset))
            num_samples = num_samples // len(args.data['train'])
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size, sampler=sampler, **loader_args)
        else:
            loader = torch.utils.data.DataLoader(
                dataset, args.batch, shuffle=(not args.fast_epoch), **loader_args)
        loaders[dataset.id] = loader
    return loaders


def process_vocabs(datasets, args):
    '''
    assign the largest output vocab to all datasets, compute embedding sizes
    '''
    # find the longest vocabulary for outputs among all datasets
    vocab_out = sorted(datasets, key=lambda x: len(x.vocab_out))[-1].vocab_out
    # make all datasets to use this vocabulary for outputs translation
    for dataset in datasets:
        dataset.vocab_translate = vocab_out
    # prepare a dictionary for embeddings initialization: vocab names and their sizes
    embs_ann = {}
    for dataset in datasets:
        embs_ann[dataset.name] = len(dataset.vocab_in)
    return embs_ann, vocab_out


@ex.automain
def main(train, exp):
    '''
    train a network using an lmdb dataset
    '''
    # parse args
    args = prepare(train, exp)
    # load dataset(s) and process vocabs
    datasets = []
    ann_types = iter(args.data['ann_type'])
    for name, ann_type in zip(args.data['train'], ann_types):
        datasets.extend(load_data(name, args, ann_type))
    for name, ann_type in zip(args.data['valid'], ann_types):
        datasets.extend(load_data(name, args, ann_type, valid_only=True))
    # assign vocabs to datasets and check their sizes for nn.Embeding inits
    embs_ann, vocab_out = process_vocabs(datasets, args)
    # wrap datasets with loaders
    loaders = wrap_datasets(datasets, args)
    # create the model
    model, optimizer, prev_train_info = create_model(args, embs_ann, vocab_out)
    # start train loop
    model.run_train(loaders, prev_train_info, optimizer=optimizer)
