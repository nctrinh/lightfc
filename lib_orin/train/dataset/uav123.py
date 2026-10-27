import os
import random
from collections import OrderedDict

import numpy as np
import pandas
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from lib.test.evaluation.uavdataset import UAVDataset  # reuse the sequence table
from .base_video_dataset import BaseVideoDataset


class UAV123(BaseVideoDataset):
    """ UAV123 dataset (used here for training).

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016

    Expected folder structure under `root`:
        root/data_seq/UAV123/<seq_folder>/000001.jpg ...
        root/anno/UAV123/<seq_name>.txt
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, seq_names=None, data_fraction=None,
                 env_num=None):
        """
        args:
            root - path to the UAV123 dataset (the folder containing data_seq/ and anno/).
            image_loader - function used to read the images.
            seq_names - optional list of sequence names (e.g. ['uav_bike1', 'uav_car1_1']) to use.
                        If None, all 123 sequences are used.
            data_fraction - fraction of the sequences to use. The complete dataset is used by default.
        """
        root = env_settings(env_num).uav123_dir if root is None else root
        super().__init__('UAV123', root, image_loader)

        # Each element of sequence_list is a sequence-info dict (name, path, startFrame, ...)
        self.sequence_list = self._build_sequence_list(seq_names)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list,
                                               int(len(self.sequence_list) * data_fraction))

        self.class_list = sorted({s['object_class'] for s in self.sequence_list})
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, seq_names=None):
        # _get_sequence_info_list does not use `self`, so we can call it without building the test dataset
        info_list = UAVDataset._get_sequence_info_list(None)

        if seq_names is not None:
            seq_names = set(seq_names)
            info_list = [s for s in info_list if s['name'] in seq_names]
            if len(info_list) == 0:
                raise ValueError('None of the given seq_names were found in UAV123.')
        return info_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_info in enumerate(self.sequence_list):
            class_name = seq_info['object_class']
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]
        return seq_per_class

    def get_name(self):
        return 'uav123'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_info):
        bb_anno_file = os.path.join(self.root, seq_info['anno_path'])
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32,
                             low_memory=False).values
        # UAV123 uses NaN boxes when the target is not present -> turn into zeros (=> invalid)
        gt = np.nan_to_num(gt, nan=0.0)
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_info = self.sequence_list[seq_id]
        bbox = self._read_bb_anno(seq_info)

        num_frames = seq_info['endFrame'] - seq_info['startFrame'] + 1
        assert bbox.shape[0] >= num_frames, \
            'Annotation of {} has {} rows but {} frames are expected.'.format(
                seq_info['name'], bbox.shape[0], num_frames)
        bbox = bbox[:num_frames]

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()   # no occlusion / out-of-view labels in UAV123

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_info, frame_id):
        frame_num = seq_info['startFrame'] + frame_id   # frame_id is 0-based within the (sub)sequence
        return os.path.join(self.root, seq_info['path'],
                            '{:0{nz}}.{ext}'.format(frame_num, nz=seq_info['nz'], ext=seq_info['ext']))

    def _get_frame(self, seq_info, frame_id):
        return self.image_loader(self._get_frame_path(seq_info, frame_id))

    def get_class_name(self, seq_id):
        return self.sequence_list[seq_id]['object_class']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_info = self.sequence_list[seq_id]
        obj_class = seq_info['object_class']

        frame_list = [self._get_frame(seq_info, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta