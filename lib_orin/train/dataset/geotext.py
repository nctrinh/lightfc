import os
import re
import json
import pickle
from collections import OrderedDict, defaultdict
 
import torch
from PIL import Image
 
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from .base_video_dataset import BaseVideoDataset

class GeoText(BaseVideoDataset):

    _IDX_RE = re.compile(r'(\d+)')

    def __init__(self, root=None, split='train', image_loader=jpeg4py_loader, anno_file='train_box.json',
                 data_fraction=None, min_visible_frames=2, use_cache=False):
        root = env_settings().geotext_dir if root is None else root
        super().__init__('Geotext', root, image_loader)

        self.img_dir = os.path.join(root, 'images', 'train')
        self.anno_path = os.path.join(root, anno_file)
        if split == 'train':
            seq_list_file = 'geotext_train.txt'
        elif split == 'test':
            seq_list_file = 'geotext_val.txt'
        elif split == 'all':
            seq_list_file = 'geotext_all.txt'
        else:
            raise ValueError("Invalid split: {}".format(split))
        self.seq_ids = self._read_seq_list(seq_list_file)

        self.cache_path = os.path.join(
            root, 'cache_geotext_{}.pkl'.format(split))

        self.sequence_list = self._build_sequence_list(use_cache)

        if min_visible_frames > 1:
            self.sequence_list = [s for s in self.sequence_list
                                  if int(s['visible'].sum()) >= min_visible_frames]
 
        if data_fraction is not None:
            n = int(len(self.sequence_list) * data_fraction)
            idx = torch.randperm(len(self.sequence_list))[:n].tolist()
            self.sequence_list = [self.sequence_list[i] for i in sorted(idx)]

    @staticmethod
    def _read_seq_list(seq_list_file):
        specs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_specs')
        with open(os.path.join(specs_dir, seq_list_file), 'r') as f:
            return set(line.strip() for line in f if line.strip())

    def _frame_index(self, rel_path):
        nums = self._IDX_RE.findall(os.path.basename(rel_path))
        return int(nums[-1]) if nums else 0

    def _build_sequence_list(self, use_cache):
        if use_cache and os.path.isfile(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        with open(self.anno_path, 'r') as f:
            entries = json.load(f)

        groups = defaultdict(list)
        for e in entries:
            rel_path = e['image']
            if not e.get('bboxes'):
                continue

            seq_id = os.path.basename(os.path.dirname(rel_path))
            if self.seq_ids is not None and seq_id not in self.seq_ids:
                continue

            groups[seq_id].append((self._frame_index(rel_path), rel_path, e['bboxes']))

        sequences = []
        for seq_id, items in groups.items():
            items.sort(key=lambda t: t[0])
            frame_paths = [os.path.join(self.img_dir, seq_id, os.path.basename(it[1])) for it in items]
            n = len(items)

            tracks = defaultdict(dict)
            for pos, (_, _, bboxes) in enumerate(items):
                W, H = self._image_size(frame_paths[pos])
                for tid, cx, cy, w, h in bboxes:
                    tid = str(int(tid))
                    if tid in tracks and pos in tracks[tid]:
                        continue  # duplicate id within a frame

                    x1 = max(0.0, (cx - w / 2) * W)
                    y1 = max(0.0, (cy - h / 2) * H)
                    x2 = min(float(W), (cx + w / 2) * W)
                    y2 = min(float(H), (cy + h / 2) * H)
                    tracks[tid][pos] = [x1, y1, x2 - x1, y2 - y1]

            for tid, boxes in tracks.items():
                bbox = torch.zeros((n, 4))
                for pos, box in boxes.items():
                    bbox[pos] = torch.tensor(box, dtype=torch.float32)
                valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
                sequences.append({
                    'name': '{}_{}'.format(seq_id, tid),
                    'seq_name': seq_id,
                    'obj_id': tid,
                    'frames': frame_paths,
                    'bbox': bbox,
                    'valid': valid,
                    'visible': valid.clone().byte(),
                })

        if use_cache:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(sequences, f)

        return sequences

    @staticmethod
    def _image_size(path):
        with Image.open(path) as img:
            return img.size

    def get_name(self):
        return 'geotext'

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