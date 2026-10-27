"""
Task-level evaluation of the quantized LightFC parts (run AFTER quantize_lightfc.py --quant_mode test).

Per-tensor cosine similarity says little about tracking quality: only the argmax cell of the
ctr map and the size/offset values at that cell are used. This script runs the full lib_orin
inference pipeline on (template, search) pairs with GT and compares 4 variants:
    F   float backbones + float fusion_head                    (reference = lib_orin)
    QB  int8 backbones  + float fusion_head                    (isolates backbone error)
    QH  float backbones + int8 fusion_head                     (isolates fusion_head error)
    Q   int8 backbones  + int8 fusion_head                     (what the board runs)
The correlation is always computed on the host in float (pixel_wise_corr_fpga), as on the board.
Int8 parts are the TorchScript models written by --quant_mode test (<quant_dir>/<part>/*_int.pt).

Decode mirrors lib_orin/test/tracker/lightfc.py:track: head._sigmoid on ctr/size, head.cal_bbox
(incl. its idx_y quirk), then un-letterbox with (scale, left, top).

Usage (Vitis AI docker):
    python eval_e2e.py --weights W --cfg_file C --geotext_dir G --geotext_split val --samples 300
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch

import quantize_lightfc as q

VARIANTS = ["F", "QB", "QH", "Q"]


def load_int_model(quant_dir, part, device):
    paths = sorted(glob.glob(os.path.join(quant_dir, part, "*_int.pt")))
    if not paths:
        raise FileNotFoundError(f"no *_int.pt in {os.path.join(quant_dir, part)} - rerun "
                                f"quantize_lightfc.py --part {part} --quant_mode test")
    model = torch.jit.load(paths[0], map_location=device)
    model.eval()
    return model


def decode(base_model, raw_ctr, raw_offset, raw_size, resize_info, search_wh):
    """lib_orin head.get_score_map._sigmoid + head.cal_bbox + track() un-letterbox. Returns (xywh, score, idx)."""
    ctr = torch.clamp(raw_ctr.float().sigmoid(), min=1e-4, max=1 - 1e-4)
    size = torch.clamp(raw_size.float().sigmoid(), min=1e-4, max=1 - 1e-4)
    bbox, max_score = base_model.head.cal_bbox(ctr, size, raw_offset.float(), return_score=True)
    idx = int(torch.max(ctr.flatten(1), dim=1)[1].item())
    cx, cy, w, h = bbox.tolist()[0]
    scale, left, top = resize_info
    sw, sh = search_wh
    w_full, h_full = w * sw / scale, h * sh / scale
    cx_orig, cy_orig = (cx * sw - left) / scale, (cy * sh - top) / scale
    return np.array([cx_orig - w_full / 2, cy_orig - h_full / 2, w_full, h_full]), float(max_score.item()), idx


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = max(a[2], 0) * max(a[3], 0) + max(b[2], 0) * max(b[3], 0) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--quant_dir", default="./quantize_lightfc_result")
    parser.add_argument("--lib_root", default=q.DEFAULT_LIB_ROOT)
    parser.add_argument("--data_dir", default=None, help="GOT-10k folder (see quantize_lightfc.py)")
    parser.add_argument("--geotext_dir", default=None)
    parser.add_argument("--geotext_split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--geotext_anno", default="train_box.json")
    parser.add_argument("--template_mode", default="gray", choices=["gray", "rgb"])
    parser.add_argument("--max_gap", type=int, default=None)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    q.import_lib_orin(os.path.abspath(args.lib_root))
    prutils = q.load_processing_utils(os.path.abspath(args.lib_root))
    from lib.utils.load import load_yaml
    import pytorch_nndct  # noqa: F401  registers the fix ops used by the *_int.pt TorchScript models

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    cfg = load_yaml(args.cfg_file)
    geometry = q.read_tracking_geometry(cfg)
    base_model = q.load_model(args.weights, cfg).to(device)
    fusion_head_f = q.FusionHeadWrapper(base_model).to(device).eval()
    bb_t_q = load_int_model(args.quant_dir, "backbone_template", device)
    bb_s_q = load_int_model(args.quant_dir, "backbone_search", device)
    fusion_head_q = load_int_model(args.quant_dir, "fusion_head", device)
    pairs = q.build_pairs(args, prutils, geometry)
    print(f"[data] {pairs.describe()}")
    search_wh = (q.SEARCH_SIZE[1], q.SEARCH_SIZE[0])

    stats = {v: {"iou_gt": [], "iou_f": [], "argmax_eq": [], "score_err": [], "score": []} for v in VARIANTS}
    with torch.no_grad():
        for n in range(args.samples):
            z, x, gt, info = pairs.sample_with_gt(rng)
            z_t = q.to_normalized_tensor([z]).to(device)
            x_t = q.to_normalized_tensor([x]).to(device)
            feats = {"f": (base_model.backbone(z_t), base_model.backbone(x_t)),
                     "q": (bb_t_q(z_t), bb_s_q(x_t))}
            corr = {k: base_model.fusion.pw_corr(zf.float(), xf.float()) for k, (zf, xf) in feats.items()}
            outs = {"F": fusion_head_f(corr["f"], feats["f"][1]),
                    "QB": fusion_head_f(corr["q"], feats["q"][1].float()),
                    "QH": fusion_head_q(corr["f"], feats["f"][1]),
                    "Q": fusion_head_q(corr["q"], feats["q"][1])}
            res = {v: decode(base_model, *outs[v], info, search_wh) for v in VARIANTS}
            box_f, score_f, idx_f = res["F"]
            for v in VARIANTS:
                box, score, idx = res[v]
                stats[v]["iou_gt"].append(iou_xywh(box, gt))
                stats[v]["iou_f"].append(iou_xywh(box, box_f))
                stats[v]["argmax_eq"].append(float(idx == idx_f))
                stats[v]["score_err"].append(abs(score - score_f))
                stats[v]["score"].append(score)
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{args.samples}")

    report = {}
    print("\n variant | mIoU vs GT | SR@0.5 vs GT | mIoU vs float | argmax == float | mean |dscore| | mean score")
    for v in VARIANTS:
        s = {k: np.array(val) for k, val in stats[v].items()}
        report[v] = {"miou_gt": s["iou_gt"].mean(), "sr50_gt": (s["iou_gt"] > 0.5).mean(),
                     "miou_vs_float": s["iou_f"].mean(), "argmax_match": s["argmax_eq"].mean(),
                     "score_abs_err": s["score_err"].mean(), "score_mean": s["score"].mean()}
        r = report[v]
        print(f" {v:7s} | {r['miou_gt']:.4f}     | {r['sr50_gt']:.4f}       | {r['miou_vs_float']:.4f}        | "
              f"{r['argmax_match']:.4f}          | {r['score_abs_err']:.4f}         | {r['score_mean']:.4f}")
    out = os.path.join(args.quant_dir, "eval_e2e.json")
    with open(out, "w") as fp:
        json.dump({"samples": args.samples, "data": pairs.describe(),
                   "report": {v: {k: float(x) for k, x in r.items()} for v, r in report.items()}}, fp, indent=2)
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
