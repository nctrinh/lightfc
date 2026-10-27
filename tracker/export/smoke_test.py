"""
Smoke test for quantize_lightfc.py. Needs torch/cv2/timm/easydict/pyyaml/tensorboard (your training env),
but no checkpoint, dataset or Vitis AI. Covers GOT-10k and GeoText-1652 calibration sources.

It builds a RANDOM-weight LightFC from a placeholder config (BN statistics randomized so that
switch_to_deploy is non-trivial), writes a tiny fake GOT-10k folder, then runs:
  1. --quant_mode float --export_onnx for all 3 parts (includes the float equivalence check
     against lib_orin's monolithic forward, which raises on mismatch);
  2. --quant_mode calib / test for fusion_head with a STUB torch_quantizer (control flow only)
     when the real pytorch_nndct is not installed.
Usage: python smoke_test.py [--lib_root ../../lib_orin] [--keep]
The placeholder config only fixes shapes; it is NOT your experiment yaml.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quantize_lightfc as q  # noqa: E402

PLACEHOLDER_YAML = """\
DATA: {MEAN: [0.485, 0.456, 0.406], STD: [0.229, 0.224, 0.225]}
MODEL:
  BACKBONE: {TYPE: MobileNetV2, STRIDE: 16}
  FUSION: {PARAMS: {num_kernel: 64, adj_channel: 96}}
  HEAD: {PARAMS: {inplanes: 192, channel: 64, feat_sz: [30, 40], stride: 16, freeze_bn: false}}
TEST: {TEMPLATE_SIZE: 128, TEMPLATE_FACTOR: 2.0}
"""


def make_fixture(root, lib_root):
    q.import_lib_orin(lib_root)
    from lib.utils.load import load_yaml
    from lib.models.tracker_model import LightFC

    cfg_file = os.path.join(root, "placeholder.yaml")
    with open(cfg_file, "w") as fp:
        fp.write(PLACEHOLDER_YAML)

    torch.manual_seed(0)
    model = LightFC(load_yaml(cfg_file), env_num=0, training=False)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    weights = os.path.join(root, "random.pth.tar")
    torch.save({"net": model.state_dict()}, weights)

    rng = np.random.default_rng(0)
    data_dir = os.path.join(root, "got10k")
    # different aspect ratios exercise the letterbox padding on both axes
    for s, (h, w) in enumerate([(720, 1280), (480, 640), (1080, 1440)]):
        seq = os.path.join(data_dir, f"GOT-10k_Train_{s:06d}")
        os.makedirs(seq)
        with open(os.path.join(seq, "groundtruth.txt"), "w") as fp:
            for i in range(6):
                cv2.imwrite(os.path.join(seq, f"{i + 1:08d}.jpg"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
                fp.write(f"{100 + 10 * i},{80 + 5 * i},{60 + 3 * i},{40 + 2 * i}\n")
    # fake GeoText-1652: seq 0839 is listed in lib_orin/train/data_specs/geotext_val.txt
    geotext_dir = os.path.join(root, "geotext")
    seq_dir = os.path.join(geotext_dir, "images", "train", "0839")
    os.makedirs(seq_dir)
    entries = []
    for i in range(4):
        name = f"image-{i + 1:02d}.jpeg"
        cv2.imwrite(os.path.join(seq_dir, name), rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
        entries.append({"image": f"images/train/0839/{name}", "bboxes": [[1, 0.5 + 0.02 * i, 0.5, 0.3, 0.25]]})
    with open(os.path.join(geotext_dir, "train_box.json"), "w") as fp:
        json.dump(entries, fp)
    return cfg_file, weights, data_dir, geotext_dir


def install_stub_quantizer(calls):
    class _Recorder:
        def __init__(self, module):
            self.module = module

        def __call__(self, *inputs):
            calls.append(("forward", tuple(inputs[0].shape)))
            return self.module(*inputs)

    class torch_quantizer:  # noqa: N801 - mirrors the Vitis AI API name
        def __init__(self, quant_mode, module, input_args, output_dir):
            calls.append(("init", quant_mode, [tuple(t.shape) for t in input_args]))
            self.quant_model = _Recorder(module)

        def export_quant_config(self):
            calls.append(("export_quant_config",))

        def export_torch_script(self):
            calls.append(("export_torch_script",))

        def export_xmodel(self, output_dir, deploy_check):
            calls.append(("export_xmodel",))

    pkg = types.ModuleType("pytorch_nndct")
    apis = types.ModuleType("pytorch_nndct.apis")
    apis.torch_quantizer = torch_quantizer
    pkg.apis = apis
    sys.modules["pytorch_nndct"] = pkg
    sys.modules["pytorch_nndct.apis"] = apis


def run(argv):
    sys.argv = ["quantize_lightfc.py"] + argv
    q.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_root", default=q.DEFAULT_LIB_ROOT)
    parser.add_argument("--keep", action="store_true", help="keep the temp folder for inspection")
    args = parser.parse_args()
    lib_root = os.path.abspath(args.lib_root)

    root = tempfile.mkdtemp(prefix="lightfc_smoke_")
    try:
        cfg_file, weights, data_dir, geotext_dir = make_fixture(root, lib_root)
        base = ["--weights", weights, "--cfg_file", cfg_file, "--lib_root", lib_root,
                "--output_dir", os.path.join(root, "out"), "--calib_samples", "5", "--eval_samples", "2",
                "--batch_size", "2"]
        common = base + ["--data_dir", data_dir]

        for part in ["backbone_template", "backbone_search", "fusion_head"]:
            run(["--part", part, "--quant_mode", "float", "--export_onnx"] + common)
            assert os.path.isfile(os.path.join(root, "out", part, f"{part}.onnx"))
            assert os.path.isfile(os.path.join(root, "out", part, "io_contract.json"))
        # GeoText-1652 alone, then mixed with GOT-10k
        geo = ["--geotext_dir", geotext_dir, "--geotext_split", "val"]
        run(["--part", "fusion_head", "--quant_mode", "float"] + base + geo)
        run(["--part", "fusion_head", "--quant_mode", "float"] + common + geo)

        try:
            import pytorch_nndct  # noqa: F401
            print("[smoke] real pytorch_nndct found - skipping stub calib/test (run them for real)")
        except ImportError:
            calls = []
            install_stub_quantizer(calls)
            run(["--part", "fusion_head", "--quant_mode", "calib", "--skip_float_check"] + common)
            run(["--part", "fusion_head", "--quant_mode", "test", "--skip_float_check"] + common)
            kinds = [c[0] for c in calls]
            assert kinds.count("init") == 2 and "export_quant_config" in kinds and "export_xmodel" in kinds, kinds
            # calib: 5 samples in batches of 2 -> 3 forwards; the export needs a final batch-1 forward
            assert calls[-3] == ("forward", (1, 64, 30, 40)), calls[-3]
            print("[smoke] stub quantizer call sequence OK")
        print(f"[smoke] PASS ({root})")
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
