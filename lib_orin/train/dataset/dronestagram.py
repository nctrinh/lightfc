import os
import re
import pickle
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
 
import torch
 
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from .base_video_dataset import BaseVideoDataset

class Dronestagram(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, min_visible_frames=1, use_cache=False):
        root = env_settings().dronestagram_dir if root is None else root
        super().__init__('Dronestagram', root, image_loader)

        self.img_dir = os.path.join(root, 'img')
        self.anno_dir = os.path.join(root, 'annotation')

        self.sequence_list = self._build_sequence_list(use_cache)

        if min_visible_frames > 1:
            self.sequence_list = [s for s in self.sequence_list
                                  if int(s['visible'].sum()) >= min_visible_frames]

        if data_fraction is not None:
            n = int(len(self.sequence_list) * data_fraction)
            idx = torch.randperm(len(self.sequence_list))[:n].tolist()
            self.sequence_list = [self.sequence_list[i] for i in sorted(idx)]

    def _build_sequence_list(self, use_cache):
        cache_path = os.path.join(self.root, 'cache_dronestagram.pkl')
        if use_cache and os.path.isfile(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        groups = defaultdict(list)
        for fname in os.listdir(self.img_dir):
            stem, ext = os.path.splitext(fname)

            xml_file = os.path.join(self.anno_dir, stem + '.xml')

            seq_name, idx = stem.split('_')
            idx = int(idx)

            groups[seq_name].append((idx, os.path.join(self.img_dir, fname), xml_file))

        sequences = []
        for seq_name, items in groups.items():
            items.sort(key=lambda x: x[0])
            frame_paths = [it[1] for it in items]
            n = len(items)

            tracks = defaultdict(dict)
            for pos, (_, _, xml_file) in enumerate(items):
                for obj_id, box in self._parse_xml(xml_file):
                    if pos not in tracks[obj_id]:
                        tracks[obj_id][pos] = box

            for obj_id, boxes in tracks.items():
                bbox = torch.zeros((n, 4))
                for pos, box in boxes.items():
                    bbox[pos] = torch.tensor(box, dtype=torch.float32)
                valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
                sequences.append({
                    'name': '{}_{}'.format(seq_name, obj_id),
                    'seq_name': seq_name,
                    'obj_id': obj_id,
                    'frames': frame_paths,
                    'bbox': bbox,
                    'valid': valid,
                    'visible': valid.clone().byte()
                })

        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(sequences, f)

        return sequences

    @staticmethod
    def _parse_xml(xml_file):
        out = []
        root = ET.parse(xml_file).getroot()
        for obj in root.findall('object'):
            obj_id = obj.findtext('name').strip()
            bb = obj.find('bndbox')
            xmin, ymin = float(bb.findtext('xmin')), float(bb.findtext('ymin'))
            xmax, ymax = float(bb.findtext('xmax')), float(bb.findtext('ymax'))

            out.append((obj_id, [xmin, ymin, xmax - xmin, ymax - ymin]))

        return out

    def get_name(self):
        return 'dronestagram'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        seq = self.sequence_list[seq_id]
        return {'bbox': seq['bbox'], 'valid': seq['valid'], 'visible': seq['visible']}

    def is_video_sequence(self):
        return True

    def get_class_name(self, seq_id):
        return 'building'

    def _get_frame(self, seq, frame_id):
        return self.image_loader(seq['frames'][frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq = self.sequence_list[seq_id]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
 
        frame_list = [self._get_frame(seq, f_id) for f_id in frame_ids]
        anno_frames = {key: [value[f_id, ...].clone() for f_id in frame_ids]
                       for key, value in anno.items()}
 
        object_meta = OrderedDict({'object_class_name': 'building',
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta