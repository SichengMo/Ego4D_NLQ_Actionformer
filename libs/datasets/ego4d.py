import os
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchtext
import numpy as np
# from nltk.tokenize import word_tokenize

from .data_utils import *
from .datasets import register_dataset

from transformers import CLIPTokenizer, CLIPTextModel
import torch


@register_dataset('ego4d')
class Ego4DDataset(Dataset):

    def __init__(
            self,
            split,  # data split, a tuple/list allowing concat of subsets
            is_training,  # whether in training mode
            vid_feat_dir,  # video feature directory
            anno_file,  # annotation json file
            max_text_len,  # max text feature length in training
            max_vid_len,  # max video feature length in training
            clip_size,  # number of frames per clip / feature
            clip_stride,  # temporal stride of clips (in frame)
            downsample_rate=None,  # down-sampling rate for video features
            normalize=False,  # if True, normalize video features to unit length
            to_fixed_len=False,  # if True, resize video features to max length
            resize_scale=None,  # lower and upper bounds for resizing scale
            jitter=None,  # random jittering of segment bounds
            trunc_thresh=0.5,  # threshold for event truncation
            crop_ratio=(0.9, 1.0),  # random cropping of video features
            name='',  # name of the dataset
            word_processor=''  # model for text embedding: one of (glove,clip,bert)
    ):
        super(Ego4DDataset, self).__init__()

        assert os.path.exists(vid_feat_dir)
        assert os.path.exists(anno_file)
        if isinstance(split, str):
            split = (split,)
        assert isinstance(split, (list, tuple))

        if downsample_rate is not None:
            assert isinstance(downsample_rate, int)
            assert downsample_rate >= 1

        if resize_scale is not None:
            assert isinstance(resize_scale, (list, tuple))
            assert len(resize_scale) == 2
            assert resize_scale[0] < resize_scale[1]

        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))
            assert len(crop_ratio) == 2
            assert crop_ratio[0] < crop_ratio[1]

        self.split = split
        self.is_training = is_training
        self.vid_feat_dir = vid_feat_dir

        self.anno_file = anno_file

        self.max_text_len = max_text_len
        self.max_vid_len = max_vid_len

        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.downsample_rate = downsample_rate
        self.normalize = normalize
        self.to_fixed_len = to_fixed_len

        self.resize_scale = resize_scale
        self.jitter = jitter
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.word_processor = word_processor  # glove, bert, clip

        self._load_text_model()
        self.data_list = self._load_annotation()

    def _load_text_model(self):
        # clip word embeddings
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

    def _load_annotation(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        # combine data from all splits
        anno_db = dict()
        for s in self.split:
            assert s in anno, 'split does not exist'
            anno_db.update(anno[s])

        data_list = tuple()
        for key, value in anno_db.items():
            fps, num_frames = 30, value['num_frames']
            duration = value['duration']
            # get sentence-event pairs
            if 'annotations' in value:
                for pair in value['annotations']:
                    start = max(pair['segment'][0], 0)
                    end = min(pair['segment'][1], 9999)

                    # filter out zero length windows in the training
                    if self.is_training and (start >= end):
                        # print(pair)
                        continue

                    sentence = pair['sentence'].strip().lower()
                    data_list += (
                        {'id': key,
                         'fps': fps,
                         'num_frames': num_frames,
                         'duration': duration,
                         'sentence': sentence,
                         'segment': (start, end),
                         'annotation_uid': pair['annotation_uid'],
                         'query_idx': pair['query_idx'],
                         },
                    )

        return data_list

    def get_text_embedding1(self, sentence):
        inputs = self.tokenizer(sentence, padding=True, return_tensors="pt", max_length=48)
        outputs = self.model(**inputs)
        text_feats = outputs.last_hidden_state.squeeze(0).detach()
        text_feats = text_feats.transpose(0, 1)
        return text_feats

    def get_vid_feature(self, data):
        vid_feat_file = os.path.join(self.vid_feat_dir, data['id'] + '.pt')
        try:

            vid_feats = torch.load(vid_feat_file).numpy().astype(np.float32)
        except:
            raise ValueError(
                'failed to load features at {:s} for video {:s}'.format(
                    vid_feat_file, data['id']
                )
            )
        return vid_feats

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # load text features
        ## NOTE: unknown tokens are assigned zero vector

        sentence = data['sentence']

        text_feats = self.get_text_embedding1(sentence)
        vid_feats = self.get_vid_feature(data)

        clip_size, clip_stride = self.clip_size, self.clip_stride

        if self.downsample_rate is not None:
            # temporally down-sample features
            ## NOTE: equivalent to using larger feature stride
            vid_feats = vid_feats[::self.downsample_rate]
            clip_stride *= self.downsample_rate

        start_frame = max(data['segment'][0] * data['fps'], 0)
        end_frame = min(data['segment'][1] * data['fps'], vid_feats.shape[0] * data['fps'])
        start = (start_frame - 0.5 * clip_size) / clip_stride
        end = (end_frame - 0.5 * clip_size) / clip_stride

        vid_len = vid_feats.shape[0]
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats.transpose()))
        if self.normalize:
            vid_feats = F.normalize(vid_feats, dim=0)

        tgt_len = None
        if self.to_fixed_len:
            tgt_len = self.max_vid_len
        elif self.is_training and self.resize_scale is not None:
            low, high = self.resize_scale
            scale = random.random() * (high - low) + low
            tgt_len = round(scale * vid_len)

        if tgt_len is not None:
            # temporally interpolate features to target length
            ## NOTE: set align_corners=True to make sure the two ends do not drift
            vid_feats = F.interpolate(
                vid_feats[None],
                size=tgt_len, mode='linear', align_corners=True
            )[0]
            # update effective clip stride
            clip_stride *= (vid_len - 1) / (tgt_len - 1)

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip

        data_dict = {'id': data['id'],  # video ID
                     'fps': data['fps'],  # frames per second
                     'num_frames': data['num_frames'],  # total number of frames
                     'duration': data['duration'],  # video duration in seconds
                     'sentence': data['sentence'],  # sentence query
                     'segment': data['segment'],  # event segment in seconds
                     'annotation_uid': data['annotation_uid'],
                     'query_idx': data['query_idx'],

                     # used in training
                     'text_feats': text_feats,  # text features (c1, t1)
                     'vid_feats': vid_feats,  # video features (c2, t2)
                     'clip_size': clip_size,  # number of frames per clip
                     'clip_stride': clip_stride,  # effective clip stride
                     'target': (start, end),  # event segment in grid unit
                     }

        if self.is_training:
            # randomly truncate video features during training
            ## NOTE: this serves as data augmentation
            jitter_seg_bounds(data_dict, self.jitter)
            truncate_vid_feats(
                data_dict, self.max_vid_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
