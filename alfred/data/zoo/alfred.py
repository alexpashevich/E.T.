import os
import torch
import pickle
import numpy as np
from io import BytesIO

from alfred.gen import constants
from alfred.data.zoo.base import BaseDataset
from alfred.utils import data_util, model_util


class AlfredDataset(BaseDataset):
    def __init__(self, name, partition, args, ann_type):
        super().__init__(name, partition, args, ann_type)
        # preset values
        self.max_subgoals = constants.MAX_SUBGOALS
        self._load_features = True
        self._load_frames = True
        # load the vocabulary for object classes
        self.vocab_obj = torch.load(os.path.join(
            constants.ET_ROOT, constants.OBJ_CLS_VOCAB))

    def load_data(self, path):
        return super().load_data(
            path, feats=True, masks=True, jsons=True)

    def __getitem__(self, idx):
        task_json, key = self.jsons_and_keys[idx]
        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(key)
        return task_json, feat_dict

    def load_masks(self, key):
        '''
        load interaction masks from the disk
        '''
        if not hasattr(self, 'masks_lmdb'):
            self.masks_lmdb, self.masks = self.load_lmdb(
                self.masks_lmdb_path)
        masks_bytes = self.masks.get(key)
        masks_list = pickle.load(BytesIO(masks_bytes))
        masks = [data_util.decompress_mask_bytes(m) for m in masks_list]
        return masks

    def load_features(self, task_json):
        '''
        load features from task_json
        '''
        feat = dict()
        # language inputs
        feat['lang'] = AlfredDataset.load_lang(task_json)

        # action outputs
        if not self.test_mode:
            # low-level action
            feat['action'] = AlfredDataset.load_action(
                task_json, self.vocab_out, self.vocab_translate)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]
            feat['object'] = AlfredDataset.load_object_classes(
                task_json, self.vocab_obj)
            assert len(feat['object']) == sum(feat['action_valid_interact'])

        # auxillary outputs
        if not self.test_mode:
            # subgoal completion supervision
            if self.args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'] = np.array(
                    task_json['num']['low_to_high_idx']) / self.max_subgoals
            # progress monitor supervision
            if self.args.progress_aux_loss_wt > 0:
                num_actions = len(task_json['num']['low_to_high_idx'])
                goal_progress = [(i + 1) / float(num_actions)
                                 for i in range(num_actions)]
                feat['goal_progress'] = goal_progress
        return feat

    @staticmethod
    def load_lang(task_json):
        '''
        load numericalized language from task_json
        '''
        lang_num_goal = task_json['num']['lang_goal']
        lang_num_instr = sum(task_json['num']['lang_instr'], [])
        lang_num = lang_num_goal + lang_num_instr
        return lang_num

    @staticmethod
    def load_action(task_json, vocab_orig, vocab_translate, action_type='action_low'):
        '''
        load action as a list of tokens from task_json
        '''
        if action_type == 'action_low':
            # load low actions
            lang_action = [
                [a['action'] for a in a_list]
                for a_list in task_json['num']['action_low']]
        elif action_type == 'action_high':
            # load high actions
            lang_action = [
                [a['action']] + a['action_high_args']
                for a in task_json['num']['action_high']]
        else:
            raise NotImplementedError('Unknown action_type {}'.format(action_type))
        lang_action = sum(lang_action, [])
        # translate actions to the provided vocab if needed
        if vocab_translate and not vocab_orig.contains_same_content(
                vocab_translate):
            lang_action = model_util.translate_to_vocab(
                lang_action, vocab_orig, vocab_translate)
        return lang_action

    @staticmethod
    def load_object_classes(task_json, vocab=None):
        '''
        load object classes for interactive actions
        '''
        object_classes = []
        for action in task_json['plan']['low_actions']:
            if model_util.has_interaction(action['api_action']['action']):
                obj_key = ('receptacleObjectId'
                           if 'receptacleObjectId' in action['api_action']
                           else 'objectId')
                object_class = action['api_action'][obj_key].split('|')[0]
                object_classes.append(
                    object_class if vocab is None
                    else vocab.word2index(object_class))
        return object_classes
