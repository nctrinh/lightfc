import os
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lib.utils import TensorDict
from .base_video_dataset import BaseVideoDataset


def _pil_to_tensor(img):
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255


def _random_scale_translation(img, boxes, jitter=0.3):
    w, h = img.size
    dw = int(w * jitter)
    dh = int(h * jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    sw = w - pl - pr
    sh = h - pt - pb

    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))
    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh - 1)

    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw - 1) - boxes[:, 2::-2]

    return cropped, boxes


def _augment_img(img, boxes, jitter=0.3):
    boxes = np.copy(boxes).astype(np.float32)
    for _ in range(5):
        img_t, boxes_t = _random_scale_translation(img.copy(), boxes.copy(), jitter=jitter)
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (boxes_t[:, 1] != boxes_t[:, 3])
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            return img_t, boxes_t
    return img, boxes


def openlogo_pair_collate(batch):
    max_boxes = max(sample["positive_boxes"].shape[0] for sample in batch)
    positive_boxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    positive_boxes_mask = torch.zeros((len(batch), max_boxes), dtype=torch.bool)

    for i, sample in enumerate(batch):
        num_boxes = sample["positive_boxes"].shape[0]
        positive_boxes[i, :num_boxes] = sample["positive_boxes"]
        positive_boxes_mask[i, :num_boxes] = True

    return TensorDict({
        "template_images": torch.stack([sample["template_images"] for sample in batch], dim=0),
        "search_images": torch.stack([sample["search_images"] for sample in batch], dim=0),
        "template_anno": torch.stack([sample["template_anno"] for sample in batch], dim=0),
        "search_anno": torch.stack([sample["search_anno"] for sample in batch], dim=0),
        "positive_boxes": positive_boxes,
        "positive_boxes_mask": positive_boxes_mask,
        "dataset": [sample["dataset"] for sample in batch],
        "test_class": [sample["test_class"] for sample in batch],
        "label_cls": torch.stack([sample["label_cls"] for sample in batch], dim=0),
        "valid": torch.tensor([sample["valid"] for sample in batch], dtype=torch.bool),
    })


class OpenLogoPairDataset(Dataset):
    """OpenLogo pair dataset that mirrors JNN_detection DatasetJNN sampling."""

    def __init__(self, root, annotations_dir, template_size, search_size, samples_per_epoch=None, is_training=True):
        self.root = root
        self.annotations_dir = annotations_dir
        self.template_size = int(template_size)
        if isinstance(search_size, (list, tuple)):
            self.search_h, self.search_w = int(search_size[0]), int(search_size[1])
        else:
            self.search_h = self.search_w = int(search_size)
        self.samples_per_epoch = samples_per_epoch
        self.is_training = is_training
        self.samples = []
        self.class_to_samples = OrderedDict()
        self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError("No OpenLogo samples found under {}".format(root))

    def _build_samples(self):
        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_key = class_name.lower().strip()
            for image_name in sorted(os.listdir(class_dir)):
                if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                image_path = os.path.join(class_dir, image_name)
                if not os.path.isfile(image_path):
                    continue
                boxes = self._read_class_boxes(image_name, class_key)
                if len(boxes) == 0:
                    continue
                sample = {"path": image_path, "class_name": class_key, "boxes": boxes}
                self.samples.append(sample)
                self.class_to_samples.setdefault(class_key, []).append(sample)

    def _read_class_boxes(self, image_name, class_name):
        stem = os.path.splitext(os.path.basename(image_name))[0]
        anno_path = os.path.join(self.annotations_dir, stem + ".xml")
        if not os.path.isfile(anno_path):
            return []
        try:
            root = ET.parse(anno_path).getroot()
        except ET.ParseError:
            return []

        boxes = []
        for obj in root.findall("object"):
            label = obj.findtext("name", default="").lower().strip()
            if label != class_name:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin")) - 1
            ymin = float(box.findtext("ymin")) - 1
            xmax = float(box.findtext("xmax")) - 1
            ymax = float(box.findtext("ymax")) - 1
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    def __len__(self):
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.samples)

    def __getitem__(self, index):
        query_sample = self.samples[index % len(self.samples)]
        target_sample = random.choice(self.class_to_samples[query_sample["class_name"]])

        query_img = Image.open(query_sample["path"]).convert("L")
        target_img = Image.open(target_sample["path"]).convert("L")

        query_boxes = query_sample["boxes"]
        qbox = random.choice(query_boxes)
        query_img = query_img.crop((qbox[0], qbox[1], qbox[2], qbox[3]))

        target_boxes = np.asarray(target_sample["boxes"], dtype=np.float32)
        if self.is_training:
            target_img, target_boxes = _augment_img(target_img, target_boxes)

        w, h = target_img.size
        target_boxes[:, 0::2] = np.clip(target_boxes[:, 0::2] / w, 0.001, 0.999)
        target_boxes[:, 1::2] = np.clip(target_boxes[:, 1::2] / h, 0.001, 0.999)
        positive_boxes = np.stack([
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2] - target_boxes[:, 0],
            target_boxes[:, 3] - target_boxes[:, 1],
        ], axis=1).astype(np.float32)

        query_img = query_img.resize((self.template_size, self.template_size), Image.BILINEAR)
        target_img = target_img.resize((self.search_w, self.search_h), Image.BILINEAR)

        template = _pil_to_tensor(query_img).unsqueeze(0)
        search = _pil_to_tensor(target_img).unsqueeze(0)
        positive_boxes = torch.from_numpy(positive_boxes)
        search_anno = positive_boxes[0].view(1, 4)
        template_anno = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

        return TensorDict({
            "template_images": template,
            "search_images": search,
            "template_anno": template_anno,
            "search_anno": search_anno,
            "positive_boxes": positive_boxes,
            "dataset": "openlogo",
            "test_class": query_sample["class_name"],
            "label_cls": torch.zeros((16, 16)),
            "valid": True,
        })


class OpenLogo(BaseVideoDataset):
    """OpenLogo split arranged like JNN_detection preproc/3/{training,testing}.

    Each class folder is exposed as a synthetic sequence. The LightFC sampler can
    then choose template/search frames from the same logo class while keeping its
    normal tracking-style data flow.
    """

    def __init__(self, root=None, annotations_dir=None, image_loader=None, split="training"):
        if root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
            root = os.path.join(repo_root, "data/openlogo")

        openlogo_root = root
        split_root = os.path.join(openlogo_root, "preproc", "3", split)
        if annotations_dir is None:
            annotations_dir = os.path.join(openlogo_root, "Annotations")

        super().__init__("OpenLogo", split_root, image_loader)
        self.annotations_dir = annotations_dir
        self.sequence_list = self._build_sequence_list()
        self.class_list = sorted(self.sequence_list)

        if len(self.sequence_list) == 0:
            raise RuntimeError("No OpenLogo samples found under {}".format(split_root))

    def get_name(self):
        return "openlogo"

    def has_class_info(self):
        return True

    def _build_sequence_list(self):
        sequences = OrderedDict()
        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            samples = []
            for image_name in sorted(os.listdir(class_dir)):
                if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                image_path = os.path.join(class_dir, image_name)
                if not os.path.isfile(image_path):
                    continue
                bbox = self._read_first_class_box(image_name, class_name)
                if bbox is not None:
                    samples.append((image_path, bbox))

            if samples:
                sequences[class_name.lower().strip()] = samples
        return sequences

    def _read_first_class_box(self, image_name, class_name):
        stem = os.path.splitext(os.path.basename(image_name))[0]
        anno_path = os.path.join(self.annotations_dir, stem + ".xml")
        if not os.path.isfile(anno_path):
            return None

        try:
            root = ET.parse(anno_path).getroot()
        except ET.ParseError:
            return None

        target_class = class_name.lower().strip()
        for obj in root.findall("object"):
            label = obj.findtext("name", default="").lower().strip()
            if label != target_class:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin")) - 1
            ymin = float(box.findtext("ymin")) - 1
            xmax = float(box.findtext("xmax")) - 1
            ymax = float(box.findtext("ymax")) - 1
            w = xmax - xmin
            h = ymax - ymin
            if w > 0 and h > 0:
                return torch.tensor([xmin, ymin, w, h], dtype=torch.float32)
        return None

    def get_num_sequences(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return [self.class_list.index(class_name)]

    def _get_samples(self, seq_id):
        return self.sequence_list[self.class_list[seq_id]]

    def get_sequence_info(self, seq_id):
        samples = self._get_samples(seq_id)
        min_len_for_sampler = 20
        num_frames = max(len(samples), min_len_for_sampler)
        bbox = torch.stack([samples[i % len(samples)][1] for i in range(num_frames)])
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()
        visible_ratio = torch.ones(num_frames, dtype=torch.float32)
        return {"bbox": bbox, "valid": valid, "visible": visible, "visible_ratio": visible_ratio}

    def get_class_name(self, seq_id):
        return self.class_list[seq_id]

    def get_frames(self, seq_id, frame_ids, anno=None):
        samples = self._get_samples(seq_id)
        frame_list = [self.image_loader(samples[f_id % len(samples)][0]) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_meta = OrderedDict({"object_class_name": self.get_class_name(seq_id)})
        return frame_list, anno_frames, obj_meta
