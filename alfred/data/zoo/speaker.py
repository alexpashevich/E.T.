from alfred.data.zoo.base import BaseDataset
from alfred.data.zoo.alfred import AlfredDataset
from alfred.utils import model_util


class SpeakerDataset(BaseDataset):
    def load_data(self, path):
        return super(SpeakerDataset, self).load_data(
            path, feats=True, masks=False, jsons=True)

    def __getitem__(self, idx):
        task_json, key = self.jsons_and_keys[idx]
        # load language and frames if asked first
        feat_dict = {}
        feat_dict['lang'] = AlfredDataset.load_lang(task_json)
        if 'frames' in self.ann_type:
            feat_dict['frames'] = self.load_frames(key)

        # load output actions
        feat_dict['action'] = AlfredDataset.load_action(
            task_json, self.vocab_out, self.vocab_translate, 'action_high')

        # remove all the lang key/value pairs if only frames are used as input
        if self.ann_type == 'frames':
            keys_lang = [key for key in feat_dict if key.startswith('lang')]
            for key in keys_lang:
                feat_dict.pop(key)
        return task_json, feat_dict
