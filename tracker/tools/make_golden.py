"""
Dump golden tensors from lib_orin (float model) for tracker/test/test_host_ops.cpp.

For one (template frame, init box, search frame) it writes, into --out:
  frame_t.png, frame_s.png         the input frames (PNG: no JPEG-decoder differences between Python and C++)
  manifest.txt                     one line per array: <name> <dtype> <dim0> <dim1> ...
  <name>.bin                       raw little-endian data
Arrays (NHWC, batch dropped):
  init_box f64[4] (x,y,w,h)        template_crop u8[T,T,3]     search_lb u8[480,640,3]   lb_info f64[3] (scale,left,top)
  template_norm f32[T,T,3]         search_norm f32[480,640,3]
  z_feat f32[T/16,T/16,96]         x_feat f32[30,40,96]        corr f32[30,40,K]
  raw_ctr f32[30,40,1]             raw_offset f32[30,40,2]     raw_size f32[30,40,2]
  box f64[4] (x,y,w,h original px) score f64[1]                index f64[1]
  template_factor f64[1]           template_gray f64[1]

Usage (training env, GPU optional):
  python make_golden.py --weights W --cfg_file C --image_t t.jpg --image_s s.jpg --box x,y,w,h --out golden/
  ./build.sh test && ./test_host_ops golden/
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "export"))

DTYPES = {np.dtype(np.uint8): "u8", np.dtype(np.float32): "f32", np.dtype(np.float64): "f64"}


class GoldenWriter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.lines = []

    def add(self, name, array):
        array = np.ascontiguousarray(array)
        if array.dtype not in DTYPES:
            raise TypeError(f"{name}: unsupported dtype {array.dtype}")
        array.astype(array.dtype.newbyteorder("<"), copy=False).tofile(os.path.join(self.out_dir, name + ".bin"))
        self.lines.append(" ".join([name, DTYPES[array.dtype]] + [str(d) for d in array.shape]))

    def close(self):
        with open(os.path.join(self.out_dir, "manifest.txt"), "w") as fp:
            fp.write("\n".join(self.lines) + "\n")


def nchw_to_nhwc(t):
    return t.permute(0, 2, 3, 1).contiguous().float().cpu().numpy()[0]


def main():
    import cv2
    import torch
    import quantize_lightfc as q
    from eval_e2e import decode

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--lib_root", default=q.DEFAULT_LIB_ROOT)
    parser.add_argument("--image_t", required=True, help="frame holding the target at --box")
    parser.add_argument("--image_s", required=True, help="search frame")
    parser.add_argument("--box", required=True, help="x,y,w,h of the target in --image_t")
    parser.add_argument("--template_mode", default="gray", choices=["gray", "rgb"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    q.import_lib_orin(os.path.abspath(args.lib_root))
    prutils = q.load_processing_utils(os.path.abspath(args.lib_root))
    from lib.utils.load import load_yaml

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.cfg_file)
    geometry = q.read_tracking_geometry(cfg)
    base_model = q.load_model(args.weights, cfg).to(device)
    fusion_head = q.FusionHeadWrapper(base_model).to(device).eval()
    pairs = q._PairsBase(prutils, geometry["template_size"], geometry["template_factor"], args.template_mode)

    box = [float(v) for v in args.box.split(",")]
    img_t_bgr = cv2.imread(args.image_t, cv2.IMREAD_COLOR)
    img_s_bgr = cv2.imread(args.image_s, cv2.IMREAD_COLOR)
    if img_t_bgr is None or img_s_bgr is None:
        raise IOError("cannot read --image_t / --image_s")
    w = GoldenWriter(args.out)
    cv2.imwrite(os.path.join(args.out, "frame_t.png"), img_t_bgr)
    cv2.imwrite(os.path.join(args.out, "frame_s.png"), img_s_bgr)

    # same calls as quantize_lightfc / eval_e2e (= lib_orin initialize() / track())
    z = pairs.template(cv2.cvtColor(img_t_bgr, cv2.COLOR_BGR2RGB), box)
    x, info = pairs.search_with_info(cv2.cvtColor(img_s_bgr, cv2.COLOR_BGR2RGB))
    z_t = q.to_normalized_tensor([z]).to(device)
    x_t = q.to_normalized_tensor([x]).to(device)
    with torch.no_grad():
        z_feat = base_model.backbone(z_t)
        x_feat = base_model.backbone(x_t)
        corr = base_model.fusion.pw_corr(z_feat, x_feat)
        raw_ctr, raw_offset, raw_size = fusion_head(corr, x_feat)
        out_box, score, index = decode(base_model, raw_ctr, raw_offset, raw_size, info,
                                       (q.SEARCH_SIZE[1], q.SEARCH_SIZE[0]))

    w.add("init_box", np.array(box, np.float64))
    w.add("template_factor", np.array([geometry["template_factor"]], np.float64))
    w.add("template_gray", np.array([1.0 if args.template_mode == "gray" else 0.0], np.float64))
    w.add("template_crop", z.astype(np.uint8))
    w.add("search_lb", x.astype(np.uint8))
    w.add("lb_info", np.array(info, np.float64))
    w.add("template_norm", nchw_to_nhwc(z_t))
    w.add("search_norm", nchw_to_nhwc(x_t))
    w.add("z_feat", nchw_to_nhwc(z_feat))
    w.add("x_feat", nchw_to_nhwc(x_feat))
    w.add("corr", nchw_to_nhwc(corr))
    w.add("raw_ctr", nchw_to_nhwc(raw_ctr))
    w.add("raw_offset", nchw_to_nhwc(raw_offset))
    w.add("raw_size", nchw_to_nhwc(raw_size))
    w.add("box", np.asarray(out_box, np.float64))
    w.add("score", np.array([score], np.float64))
    w.add("index", np.array([index], np.float64))
    w.close()
    print(f"[golden] wrote {args.out}: box={np.round(out_box, 2).tolist()} score={score:.4f} index={index}")


if __name__ == "__main__":
    main()
