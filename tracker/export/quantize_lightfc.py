"""
PTQ (Post-Training Quantization) for LightFC, split into the 3 DPU-mappable parts.
Ported from the original quantize_lightfc.py and updated to match lib_orin (the source of truth).

Why the split: forward() contains two groups of ops that can never run on an
accelerator - pixel_wise_corr_fpga (a conv whose weights are runtime tensors)
and the head decode (sigmoid/clamp/argmax/gather/box math, cal_bbox). Only the
static-weight parts are quantized:
  1. backbone_template - MobileNetV2, input (1,3,T,T),  T = cfg.TEST.TEMPLATE_SIZE
  2. backbone_search   - MobileNetV2, input (1,3,480,640)
     (same weights as #1, compiled separately because the DPU needs fixed shapes)
  3. fusion_head       - fusion WITHOUT pw_corr + head WITHOUT sigmoid.
     Inputs: corr (1, (T/16)^2, 30, 40) computed on the host, x_feat (1,96,30,40).
     Outputs: raw_ctr (1,1,30,40), raw_offset (1,2,30,40), raw_size (1,2,30,40).
pixel_wise_corr_fpga and the decode run on the host (C++) between the DPU calls.

Changes vs. the previous script (see TRACKER_UPDATE_PLAN.md section 0.1, S1-S10):
  S1  lib_orin removed LightFC.forward_fusion_head_dpu -> FusionHeadWrapper calls the
      lib_orin submodules itself, op by op (same math as fusion.forward / head.get_score_map).
  S2  search: gray -> 3ch -> prutils.resize LETTERBOX to 640x480 (lib_orin track()),
      not a cv2.resize stretch.
  S3  template: sample_target(GT box, TEMPLATE_FACTOR, TEMPLATE_SIZE) on the gray x3
      frame (training uses ToGrayscale(p=1) jointly). --template_mode rgb is available
      to reproduce lib_orin's test initialize(), which crops from the colour frame.
  S4  template/search pairs come from the SAME sequence (template = GT crop, search = another
      visible frame, sampled like lib_orin TUSampler) so corr calibration sees real correlation
      peaks. Sources: GOT-10k (--data_dir) and/or GeoText-1652 (--geotext_dir).
  S8  --weights / --cfg_file are required (no hard-coded checkpoint).
  S9  calibration runs over --calib_samples images in mini-batches.
  S10 lib_orin is imported as package `lib` at runtime (read-only, nothing copied).
New:
  --quant_mode float : float-only checks (+ optional --export_onnx), no Vitis AI needed.
  A float equivalence check runs before every quantization, and io_contract.json
  records the graph I/O that tracker/src must follow.

Usage (each part twice, calib then test; Vitis AI docker, pytorch_nndct installed):
    python quantize_lightfc.py --part backbone_template --quant_mode calib --weights W --cfg_file C --data_dir D
    python quantize_lightfc.py --part backbone_template --quant_mode test  --weights W --cfg_file C --data_dir D
    (calibration data: --data_dir GOT-10k and/or --geotext_dir GeoText-1652 [--geotext_split train|val|all])
    ... same for --part backbone_search and --part fusion_head
Hardware-agnostic float export (any toolchain that takes ONNX):
    python quantize_lightfc.py --part fusion_head --quant_mode float --export_onnx ...
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Search input (H, W): lib_orin/train/data/processing_utils.py:resize letterboxes to 640x480.
SEARCH_SIZE = (480, 640)
BACKBONE_STRIDE = 16

# lib_orin/test/tracker/lightfc.py:lightFC.__init__ and test/tracker/data_utils.py:Preprocessor
# hard-code these at inference time; the deployed pipeline uses them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FUSION_HEAD_OUTPUTS = ["raw_ctr", "raw_offset", "raw_size"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "lib_orin"))


# ----------------------------------------------------------------------------
# lib_orin import (read-only)
# ----------------------------------------------------------------------------
def import_lib_orin(lib_root):
    """lib_orin's own imports say `from lib.xxx import ...`, so register lib_orin as `lib`."""
    init_file = os.path.join(lib_root, "__init__.py")
    if not os.path.isfile(init_file):
        raise FileNotFoundError(f"{init_file} not found - pass --lib_root pointing at lib_orin/")
    if "lib" in sys.modules:
        existing = os.path.normpath(os.path.dirname(sys.modules["lib"].__file__ or ""))
        if existing != os.path.normpath(lib_root):
            raise RuntimeError(f"A different package `lib` is already imported from {existing}")
        return
    spec = importlib.util.spec_from_file_location("lib", init_file, submodule_search_locations=[lib_root])
    module = importlib.util.module_from_spec(spec)
    sys.modules["lib"] = module
    spec.loader.exec_module(module)


def load_processing_utils(lib_root):
    """
    Load lib_orin/train/data/processing_utils.py by file path. Importing it as
    lib.train.data.processing_utils would run lib/train/data/__init__.py, which pulls
    in jpeg4py and torch._six (training-only dependencies). The file itself only needs
    torch/cv2/numpy, and we use sample_target() and resize() from it unchanged.
    """
    path = os.path.join(lib_root, "train", "data", "processing_utils.py")
    spec = importlib.util.spec_from_file_location("lib_orin_processing_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Wrappers (what gets quantized / compiled)
# ----------------------------------------------------------------------------
class BackboneWrapper(nn.Module):
    """
    Backbone only (MobileNetV2). Used for both --part backbone_template and
    --part backbone_search: same class and weights, only the input shape differs,
    so the toolchain produces 2 separately compiled graphs.
    """

    def __init__(self, lightfc_model):
        super().__init__()
        self.backbone = lightfc_model.backbone

    def forward(self, img):
        return self.backbone(img)


class FusionHeadWrapper(nn.Module):
    """
    Fusion after the correlation + head before the decode. Returns a tuple
    because the quantizer can only trace tensor/tuple/list outputs.

    lib_orin dropped forward_fusion_head_dpu, so this mirrors, op by op:
      lib_orin/models/lightfc/fusion/ecm.py : pwcorr_scf_sc_iab_sc_concat.forward  (minus self.pw_corr)
      lib_orin/models/lightfc/head/erh.py   : repn33_center_concat_geom.get_score_map (minus _sigmoid)
    `corr` must be pixel_wise_corr_fpga(z_feat, x_feat), computed outside (host side).
    """

    def __init__(self, lightfc_model):
        super().__init__()
        fusion_name = type(lightfc_model.fusion).__name__
        head_name = type(lightfc_model.head).__name__
        # Fail loudly if lib_orin's LightFC ever switches fusion/head (e.g. pwcorr_2conv_sc_concat,
        # which adds residuals and would need a different wrapper).
        if fusion_name != "pwcorr_scf_sc_iab_sc_concat" or head_name != "repn33_center_concat_geom":
            raise NotImplementedError(f"FusionHeadWrapper mirrors pwcorr_scf_sc_iab_sc_concat + "
                                      f"repn33_center_concat_geom, got {fusion_name} + {head_name}")
        self.fusion = lightfc_model.fusion
        self.head = lightfc_model.head

    def forward(self, corr, x_feat):
        f = self.fusion
        corr = f.bn_down1(f.conv_down1(corr))
        corr = f.conv1(corr)
        corr = f.bn_down2(f.conv_down2(corr))
        corr = f.conv2(corr)
        corr = f.adjust(corr)
        x = torch.cat((corr, x_feat), dim=1)

        h = self.head
        # ctr branch
        x_ctr = h.conv4_ctr(h.conv3_ctr(h.conv2_ctr(h.conv1_ctr(x))))
        raw_ctr = h.conv5_ctr(x_ctr)
        # geometry branch (shared trunk, separate 1x1 outputs)
        x_geom = h.conv4_geom(h.conv3_geom(h.conv2_geom(h.conv1_geom(x))))
        raw_offset = h.conv5_offset(x_geom)
        raw_size = h.conv5_size(x_geom)
        return raw_ctr, raw_offset, raw_size


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(weights_path, cfg):
    """
    training=False: the checkpoint holds the full trained weights, no load_pretrain().

    switch_to_deploy(): re-parameterize repN33 blocks into a single conv, exactly as
    lib_orin/test/tracker/lightfc.py does before track(). It is forward-equivalent, but
    it MUST happen before quantization so that the quantized graph is the deployed one.
    """
    from lib.models.tracker_model import LightFC

    base_model = LightFC(cfg, env_num=0, training=False)

    ckpt = torch.load(weights_path, map_location="cpu")
    # lib_orin/test/tracker/lightfc.py accepts both {'net': state_dict} and a bare state_dict.
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    base_model.load_state_dict(state_dict, strict=True)

    for part in (base_model.backbone, base_model.fusion, base_model.head):
        for module in part.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()

    base_model.eval()
    return base_model


def read_tracking_geometry(cfg):
    template_size = int(cfg.TEST.TEMPLATE_SIZE)
    template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    if template_size % BACKBONE_STRIDE != 0:
        raise ValueError(f"TEMPLATE_SIZE={template_size} is not a multiple of stride {BACKBONE_STRIDE}")
    z_feat_sz = template_size // BACKBONE_STRIDE
    x_feat_sz = (SEARCH_SIZE[0] // BACKBONE_STRIDE, SEARCH_SIZE[1] // BACKBONE_STRIDE)

    head_feat_sz = list(cfg.MODEL.HEAD.PARAMS.feat_sz)
    if head_feat_sz != list(x_feat_sz):
        raise ValueError(f"cfg.MODEL.HEAD.PARAMS.feat_sz={head_feat_sz} but a {SEARCH_SIZE} search image "
                         f"gives {list(x_feat_sz)} - cal_bbox would decode with the wrong grid")
    num_kernel = int(cfg.MODEL.FUSION.PARAMS.num_kernel)
    if num_kernel != z_feat_sz * z_feat_sz:
        raise ValueError(f"FUSION num_kernel={num_kernel} != (TEMPLATE_SIZE/16)^2={z_feat_sz ** 2}")

    mean = [float(v) for v in cfg.DATA.MEAN]
    std = [float(v) for v in cfg.DATA.STD]
    if not (np.allclose(mean, IMAGENET_MEAN) and np.allclose(std, IMAGENET_STD)):
        print(f"[warn] cfg.DATA.MEAN/STD {mean}/{std} differ from the ImageNet values hard-coded in "
              f"lib_orin/test/tracker/lightfc.py; using the test-time (ImageNet) values.")

    return dict(template_size=template_size, template_factor=template_factor, z_feat_sz=z_feat_sz,
                x_feat_sz=x_feat_sz, num_kernel=num_kernel)


# ----------------------------------------------------------------------------
# Calibration data: GOT-10k train and/or GeoText-1652 (same-sequence template/search pairs)
# ----------------------------------------------------------------------------
class _PairsBase:
    """
    Random (template, search) pairs preprocessed exactly like lib_orin. Subclasses fill
    self.sequences with (name, frame_paths, boxes (N,4) xywh px, visible (N,) bool).

    Pair sampling mirrors lib_orin/train/data/sampler.py:TUSampler: the template is drawn
    uniformly over all visible (sequence, frame) entries, the search frame is any OTHER
    visible frame of the same sequence/object. max_gap (None = no limit, as in TUSampler)
    optionally restricts |search - template|.
    """

    name = "pairs"

    def __init__(self, prutils, template_size, template_factor, template_mode="gray", max_gap=None):
        self.prutils = prutils
        self.template_size = template_size
        self.template_factor = template_factor
        self.template_mode = template_mode
        self.max_gap = max_gap
        self.sequences = []
        self.index = []

    def _finalize(self, where):
        # TUSampler._build_index: sequences with < 2 visible frames cannot form a pair.
        self.index = []
        for seq_idx, (_, _, _, visible) in enumerate(self.sequences):
            vis = np.flatnonzero(visible)
            if len(vis) >= 2:
                self.index.extend((seq_idx, int(f)) for f in vis)
        if len(self.index) == 0:
            raise ValueError(f"[{self.name}] no sequence with >= 2 visible frames found in {where}")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _read_rgb(path):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"cannot read {path}")
        # lib_orin trackers receive RGB frames.
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _gray3(img_rgb):
        # lib_orin/train/data/transforms.py:ToGrayscale and test/tracker/lightfc.py:track
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.stack([img_gray, img_gray, img_gray], axis=2)

    def template(self, img_rgb, box):
        # lib_orin/test/tracker/lightfc.py:initialize -> sample_target(image, init_bbox, factor, size).
        img = self._gray3(img_rgb) if self.template_mode == "gray" else img_rgb
        crop, _, _ = self.prutils.sample_target(img, list(map(float, box)), self.template_factor,
                                                output_sz=self.template_size)
        return crop

    def search(self, img_rgb):
        # lib_orin/test/tracker/lightfc.py:track -> gray x3 -> prutils.resize (letterbox, zero pad).
        resized_images, _ = self.prutils.resize(self._gray3(img_rgb))
        return resized_images[0]

    def sample(self, rng):
        for _ in range(50):
            seq_idx, t = self.index[rng.randrange(len(self.index))]
            _, frames, boxes, visible = self.sequences[seq_idx]
            candidates = [i for i in np.flatnonzero(visible) if i != t and
                          (self.max_gap is None or abs(int(i) - t) <= self.max_gap)]
            if not candidates:
                continue
            s = int(candidates[rng.randrange(len(candidates))])
            try:
                z = self.template(self._read_rgb(frames[t]), boxes[t])
            except Exception:  # sample_target raises on degenerate boxes
                continue
            x = self.search(self._read_rgb(frames[s]))
            return z, x
        raise RuntimeError(f"[{self.name}] could not draw a valid calibration pair in 50 tries")


class Got10kPairs(_PairsBase):
    """GOT-10k train layout: <data_dir>/<seq>/*.jpg + groundtruth.txt (+ absence.label)."""

    name = "got10k"

    def __init__(self, data_dir, prutils, template_size, template_factor, template_mode="gray", max_gap=None):
        super().__init__(prutils, template_size, template_factor, template_mode, max_gap)
        for seq_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
            gt_file = os.path.join(seq_dir, "groundtruth.txt")
            frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
            if not os.path.isfile(gt_file) or len(frames) == 0:
                continue
            boxes = np.loadtxt(gt_file, delimiter=",", dtype=np.float64).reshape(-1, 4)
            n = min(len(frames), len(boxes))
            visible = (boxes[:n, 2] > 0) & (boxes[:n, 3] > 0)
            absence_file = os.path.join(seq_dir, "absence.label")
            if os.path.isfile(absence_file):
                absent = np.loadtxt(absence_file, dtype=np.int64).reshape(-1)[:n]
                visible[:len(absent)] &= absent == 0
            self.sequences.append((os.path.basename(seq_dir), frames[:n], boxes[:n], visible))
        self._finalize(data_dir)


class GeoText1652Pairs(_PairsBase):
    """
    GeoText-1652 building tracks. Mirrors lib_orin/train/dataset/geotext.py:GeoText
    (_read_seq_list, _frame_index, _build_sequence_list, min_visible_frames) without importing
    it - that module imports lib.train.data (jpeg4py, torch._six) and lib.train.admin.

    Layout: <root>/<anno_file> (default train_box.json, a list of
    {"image": "<...>/<seq_id>/<file>", "bboxes": [[track_id, cx, cy, w, h], ...]} with
    normalized centre boxes) and images in <root>/images/train/<seq_id>/<file>.
    Split ids come from lib_orin/train/data_specs/geotext_{train,val,all}.txt.
    Every (seq_id, track_id) is one sequence; frames where the track is missing are invisible.
    """

    name = "geotext"
    _IDX_RE = re.compile(r"(\d+)")
    # GeoText(split=...) -> data_specs file; 'test' is the name base_functions.py uses for the val split.
    _SPLIT_FILES = {"train": "geotext_train.txt", "val": "geotext_val.txt", "test": "geotext_val.txt",
                    "all": "geotext_all.txt"}

    def __init__(self, root, lib_root, prutils, template_size, template_factor, split="train",
                 anno_file="train_box.json", min_visible_frames=2, template_mode="gray", max_gap=None):
        super().__init__(prutils, template_size, template_factor, template_mode, max_gap)
        if split not in self._SPLIT_FILES:
            raise ValueError(f"Invalid GeoText split: {split}")
        img_dir = os.path.join(root, "images", "train")
        seq_ids = self._read_seq_list(os.path.join(lib_root, "train", "data_specs", self._SPLIT_FILES[split]))

        with open(os.path.join(root, anno_file), "r") as fp:
            entries = json.load(fp)

        groups = defaultdict(list)
        for e in entries:
            rel_path = e["image"]
            if not e.get("bboxes"):
                continue
            seq_id = os.path.basename(os.path.dirname(rel_path))
            if seq_id not in seq_ids:
                continue
            groups[seq_id].append((self._frame_index(rel_path), rel_path, e["bboxes"]))

        for seq_id in sorted(groups):
            items = sorted(groups[seq_id], key=lambda it: it[0])
            frame_paths = [os.path.join(img_dir, seq_id, os.path.basename(it[1])) for it in items]
            n = len(items)

            tracks = defaultdict(dict)
            for pos, (_, _, bboxes) in enumerate(items):
                W, H = self._image_size(frame_paths[pos])
                for tid, cx, cy, w, h in bboxes:
                    tid = str(int(tid))
                    if pos in tracks[tid]:
                        continue  # duplicate id within a frame: keep the first, as GeoText does
                    x1 = max(0.0, (cx - w / 2) * W)
                    y1 = max(0.0, (cy - h / 2) * H)
                    x2 = min(float(W), (cx + w / 2) * W)
                    y2 = min(float(H), (cy + h / 2) * H)
                    tracks[tid][pos] = [x1, y1, x2 - x1, y2 - y1]

            for tid in sorted(tracks, key=int):
                boxes = np.zeros((n, 4), dtype=np.float64)
                for pos, box in tracks[tid].items():
                    boxes[pos] = box
                visible = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
                if int(visible.sum()) >= min_visible_frames:
                    self.sequences.append((f"{seq_id}_{tid}", frame_paths, boxes, visible))
        self._finalize(root)

    @staticmethod
    def _read_seq_list(path):
        with open(path, "r") as fp:
            return set(line.strip() for line in fp if line.strip())

    def _frame_index(self, rel_path):
        nums = self._IDX_RE.findall(os.path.basename(rel_path))
        return int(nums[-1]) if nums else 0

    @staticmethod
    def _image_size(path):
        # geotext.py uses PIL (header read only); fall back to cv2 if PIL is unavailable.
        try:
            from PIL import Image
            with Image.open(path) as img:
                return img.size
        except ImportError:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cannot read {path}")
            return img.shape[1], img.shape[0]


class MixedPairs:
    """
    Several pair sources. Mirrors TUSampler with p_datasets=None: a source is chosen with
    probability proportional to its number of sequences, then sampled as above.
    """

    def __init__(self, sources):
        self.sources = sources
        self.weights = [len(s.sequences) for s in sources]

    def sample(self, rng):
        return rng.choices(self.sources, weights=self.weights)[0].sample(rng)

    def describe(self):
        return ", ".join(f"{s.name}: {len(s.sequences)} sequences / {len(s)} template frames" for s in self.sources)


def build_pairs(args, prutils, geometry):
    common = dict(prutils=prutils, template_size=geometry["template_size"],
                  template_factor=geometry["template_factor"], template_mode=args.template_mode,
                  max_gap=args.max_gap)
    sources = []
    if args.data_dir:
        sources.append(Got10kPairs(args.data_dir, **common))
    if args.geotext_dir:
        sources.append(GeoText1652Pairs(args.geotext_dir, os.path.abspath(args.lib_root), split=args.geotext_split,
                                        anno_file=args.geotext_anno, **common))
    if not sources:
        raise SystemExit("give at least one calibration dataset: --data_dir (GOT-10k) and/or --geotext_dir")
    return MixedPairs(sources)


def to_normalized_tensor(images_hwc_uint8):
    """(N,H,W,3) uint8 -> (N,3,H,W) float, (img/255 - mean)/std, as Preprocessor.process / track()."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    t = torch.from_numpy(np.stack(images_hwc_uint8)).permute(0, 3, 1, 2).contiguous().float()
    return (t / 255.0 - mean) / std


def make_part_inputs(part, base_model, pairs, rng, batch_size, device):
    """Return the tuple of inputs for one batch of `part`."""
    zs, xs = zip(*[pairs.sample(rng) for _ in range(batch_size)])
    if part == "backbone_template":
        return (to_normalized_tensor(zs).to(device),)
    if part == "backbone_search":
        return (to_normalized_tensor(xs).to(device),)
    z = to_normalized_tensor(zs).to(device)
    x = to_normalized_tensor(xs).to(device)
    with torch.no_grad():
        z_feat = base_model.backbone(z)
        x_feat = base_model.backbone(x)
        # lib_orin pw_corr == pixel_wise_corr_fpga (permute-fixed kernel layout).
        corr = base_model.fusion.pw_corr(z_feat, x_feat)
    return corr, x_feat


def iterate_batches(part, base_model, pairs, rng, total, batch_size, device):
    done = 0
    while done < total:
        n = min(batch_size, total - done)
        yield make_part_inputs(part, base_model, pairs, rng, n, device)
        done += n


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def calculate_similarity(float_tensor, quant_tensor):
    """Cosine similarity and MSE between the float and the quantized output."""
    f = float_tensor.reshape(-1).float().cpu()
    q = quant_tensor.reshape(-1).float().cpu()
    cos_sim = F.cosine_similarity(f, q, dim=0).item()
    mse = F.mse_loss(f, q).item()
    return cos_sim, mse


def check_float_equivalence(base_model, pairs, rng, device, n=4, atol=1e-4):
    """
    The split pipeline must equal lib_orin's monolithic forward in float:
      (a) pw_corr (pixel_wise_corr_fpga) == pixel_wise_corr (matmul)  -> corr channel layout
      (b) sigmoid/clamp(FusionHeadWrapper(corr, x_feat)) == head.get_score_map(fusion(z_feat, x_feat))
    """
    from lib.models.lightfc.fusion.ecm import pixel_wise_corr

    wrapper = FusionHeadWrapper(base_model).to(device).eval()
    zs, xs = zip(*[pairs.sample(rng) for _ in range(n)])
    with torch.no_grad():
        z_feat = base_model.backbone(to_normalized_tensor(zs).to(device))
        x_feat = base_model.backbone(to_normalized_tensor(xs).to(device))
        corr = base_model.fusion.pw_corr(z_feat, x_feat)
        err_corr = (corr - pixel_wise_corr(z_feat, x_feat)).abs().max().item()

        ref_ctr, ref_size, ref_offset = base_model.head.get_score_map(base_model.fusion(z_feat, x_feat))
        raw_ctr, raw_offset, raw_size = wrapper(corr, x_feat)
        # lib_orin/models/lightfc/head/erh.py:get_score_map._sigmoid
        ctr = torch.clamp(raw_ctr.sigmoid(), min=1e-4, max=1 - 1e-4)
        size = torch.clamp(raw_size.sigmoid(), min=1e-4, max=1 - 1e-4)
        errs = {"corr_vs_matmul": err_corr,
                "ctr": (ctr - ref_ctr).abs().max().item(),
                "size": (size - ref_size).abs().max().item(),
                "offset": (raw_offset - ref_offset).abs().max().item()}

    print("[float check] max |split - lib_orin forward|: " +
          ", ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    # corr magnitudes are large (sums of 96 products), compare relative to its scale.
    corr_tol = atol * max(1.0, corr.abs().max().item())
    bad = [k for k, v in errs.items() if v > (corr_tol if k == "corr_vs_matmul" else atol)]
    if bad:
        raise RuntimeError(f"Split pipeline does not reproduce lib_orin forward for: {bad}")
    return errs


def write_io_contract(path, args, geometry, part, inputs, outputs):
    contract = {
        "part": part,
        "source": "tracker/export/quantize_lightfc.py (mirrors lib_orin)",
        "weights": os.path.abspath(args.weights),
        "cfg_file": os.path.abspath(args.cfg_file),
        "calibration_data": {"got10k": args.data_dir, "geotext": args.geotext_dir,
                             "geotext_split": args.geotext_split if args.geotext_dir else None},
        "layout_note": "shapes below are PyTorch NCHW; compiled DPU graphs use NHWC",
        "inputs": {name: list(t.shape) for name, t in inputs.items()},
        "outputs": {name: list(t.shape) for name, t in outputs.items()},
        "preprocess": {
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "formula": "(px/255 - mean_c)/std_c",
            "channels": "gray (cv2 RGB2GRAY) stacked x3" if (part != "backbone_template"
                                                              or args.template_mode == "gray") else "RGB",
            "template": f"sample_target(box, factor={geometry['template_factor']}, "
                        f"output_sz={geometry['template_size']})",
            "search": f"prutils.resize letterbox to {SEARCH_SIZE[1]}x{SEARCH_SIZE[0]} (WxH), zero pad",
        },
        "corr": f"pixel_wise_corr_fpga: out[k=i*{geometry['z_feat_sz']}+j][h][w] = "
                f"sum_c z[c][i][j] * x[c][h][w], {geometry['num_kernel']} channels",
        "fusion_head_outputs": "pre-sigmoid; host applies clamp(sigmoid, 1e-4, 1-1e-4) to ctr and size, "
                               "offset stays raw",
        "template_size": geometry["template_size"],
        "template_factor": geometry["template_factor"],
        "search_size_hw": list(SEARCH_SIZE),
        "feat_sz_hw": list(geometry["x_feat_sz"]),
        "num_kernel": geometry["num_kernel"],
    }
    with open(path, "w") as fp:
        json.dump(contract, fp, indent=2)
    print(f"[io] wrote {path}")


def input_names(part):
    return ["corr", "x_feat"] if part == "fusion_head" else ["img"]


def output_names(part):
    return FUSION_HEAD_OUTPUTS if part == "fusion_head" else ["feat"]


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, required=True,
                        choices=["backbone_template", "backbone_search", "fusion_head"],
                        help="Which part of the model to quantize/export.")
    parser.add_argument("--quant_mode", type=str, default="calib", choices=["float", "calib", "test"],
                        help="float: float checks (+ --export_onnx) only, no Vitis AI. "
                             "calib: collect activation statistics. test: evaluate and export the xmodel.")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint .pth.tar (e.g. LightFC_best.pth.tar).")
    parser.add_argument("--cfg_file", type=str, required=True, help="Experiment yaml used to train the checkpoint.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="GOT-10k train folder: <data_dir>/<seq>/*.jpg + groundtruth.txt.")
    parser.add_argument("--geotext_dir", type=str, default=None,
                        help="GeoText-1652 root: <root>/train_box.json + <root>/images/train/<seq_id>/.")
    parser.add_argument("--geotext_split", type=str, default="train", choices=["train", "val", "all"],
                        help="lib_orin/train/data_specs/geotext_{split}.txt")
    parser.add_argument("--geotext_anno", type=str, default="train_box.json", help="Annotation json under --geotext_dir.")
    parser.add_argument("--output_dir", type=str, default="./quantize_lightfc_result",
                        help="Root output folder (one sub-folder per --part).")
    parser.add_argument("--lib_root", type=str, default=DEFAULT_LIB_ROOT, help="Path to lib_orin/.")
    parser.add_argument("--template_mode", type=str, default="gray", choices=["gray", "rgb"],
                        help="gray: matches training (ToGrayscale p=1). rgb: matches lib_orin test initialize().")
    parser.add_argument("--calib_samples", type=int, default=200, help="Number of calibration samples.")
    parser.add_argument("--eval_samples", type=int, default=16, help="Samples for float-vs-quant similarity.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size during calibration.")
    parser.add_argument("--max_gap", type=int, default=None,
                        help="Max |search - template| frame gap. Default: none, like lib_orin TUSampler.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export_onnx", action="store_true", help="Also export the float wrapper to ONNX.")
    parser.add_argument("--skip_float_check", action="store_true")
    args = parser.parse_args()

    import_lib_orin(os.path.abspath(args.lib_root))
    prutils = load_processing_utils(os.path.abspath(args.lib_root))
    from lib.utils.load import load_yaml

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    part_output_dir = os.path.join(args.output_dir, args.part)
    os.makedirs(part_output_dir, exist_ok=True)

    cfg = load_yaml(args.cfg_file)
    geometry = read_tracking_geometry(cfg)
    base_model = load_model(args.weights, cfg).to(device)

    pairs = build_pairs(args, prutils, geometry)
    print(f"[data] {pairs.describe()}")

    if not args.skip_float_check:
        check_float_equivalence(base_model, pairs, rng, device)

    wrapped = (FusionHeadWrapper(base_model) if args.part == "fusion_head" else BackboneWrapper(base_model))
    wrapped = wrapped.to(device).eval()

    # Batch-1 example: traced by the quantizer and used for export (the DPU graph has batch 1).
    example = make_part_inputs(args.part, base_model, pairs, rng, 1, device)
    with torch.no_grad():
        example_out = wrapped(*example)
    example_out = example_out if isinstance(example_out, tuple) else (example_out,)
    write_io_contract(os.path.join(part_output_dir, "io_contract.json"), args, geometry, args.part,
                      dict(zip(input_names(args.part), example)), dict(zip(output_names(args.part), example_out)))

    if args.export_onnx:
        onnx_path = os.path.join(part_output_dir, f"{args.part}.onnx")
        onnx_kwargs = dict(opset_version=13, input_names=input_names(args.part), output_names=output_names(args.part))
        try:  # torch >= 2.5 has a `dynamo` switch (default True from 2.9); keep the TorchScript exporter
            torch.onnx.export(wrapped, example, onnx_path, dynamo=False, **onnx_kwargs)
        except TypeError:  # older torch (e.g. Vitis AI docker): no `dynamo` argument
            torch.onnx.export(wrapped, example, onnx_path, **onnx_kwargs)
        print(f"[float:{args.part}] exported {onnx_path}")

    if args.quant_mode == "float":
        return

    from pytorch_nndct.apis import torch_quantizer  # Vitis AI only

    quantizer = torch_quantizer(quant_mode=args.quant_mode, module=wrapped, input_args=example,
                                output_dir=part_output_dir)
    quant_model = quantizer.quant_model

    if args.quant_mode == "calib":
        with torch.no_grad():
            for batch in iterate_batches(args.part, base_model, pairs, rng, args.calib_samples,
                                         args.batch_size, device):
                _ = quant_model(*batch)
        quantizer.export_quant_config()
        print(f"[calib:{args.part}] saved quant config to {part_output_dir}")

    else:  # test
        print("\n" + "=" * 50)
        print(f" QUANTIZATION ERROR FOR: {args.part}")
        print("=" * 50)
        names = output_names(args.part)
        sums = {name: [0.0, 0.0] for name in names}
        with torch.no_grad():
            for batch in iterate_batches(args.part, base_model, pairs, rng, args.eval_samples, 1, device):
                float_outputs = wrapped(*batch)
                quant_outputs = quant_model(*batch)
                if not isinstance(float_outputs, tuple):
                    float_outputs, quant_outputs = (float_outputs,), (quant_outputs,)
                for name, f_out, q_out in zip(names, float_outputs, quant_outputs):
                    cos_sim, mse = calculate_similarity(f_out, q_out)
                    sums[name][0] += cos_sim
                    sums[name][1] += mse
            # The export below needs the last forward to be batch 1 with the example input.
            _ = quant_model(*example)
        for name in names:
            print(f" -> Output [{name}]: mean Cosine Similarity = {sums[name][0] / args.eval_samples:.5f} | "
                  f"mean MSE = {sums[name][1] / args.eval_samples:.6f}")
        print("=" * 50 + "\n")

        quantizer.export_torch_script()
        quantizer.export_xmodel(output_dir=part_output_dir, deploy_check=False)
        print(f"[test:{args.part}] exported xmodel to {part_output_dir}")


if __name__ == "__main__":
    main()
