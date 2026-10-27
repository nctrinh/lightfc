# tracker/export: model export / PTQ for the LightFC tracker

`quantize_lightfc.py` splits `lib_orin`'s LightFC into the 3 graphs the C++ tracker runs. It imports
`lib_orin/` read-only, registered as the package `lib`, and copies nothing.

| part | input (NCHW) | output (NCHW) | runs on |
|---|---|---|---|
| `backbone_template` | `img` 1×3×T×T | `feat` 1×96×T/16×T/16 | accelerator |
| `backbone_search` | `img` 1×3×480×640 | `feat` 1×96×30×40 | accelerator |
| *(corr)* | z_feat, x_feat | 1×(T/16)²×30×40 | **host** (`pixel_wise_corr_fpga`) |
| `fusion_head` | `corr`, `x_feat` | `raw_ctr` 1×1×30×40, `raw_offset` 1×2×30×40, `raw_size` 1×2×30×40 (pre-sigmoid) | accelerator |
| *(decode)* | raw outputs | box + score | **host** (sigmoid/clamp, argmax, `cal_bbox`, un-letterbox) |

`T = cfg.TEST.TEMPLATE_SIZE`. Compiled Vitis AI graphs are NHWC. Each run writes `io_contract.json` next to
its outputs, recording the exact shapes and the preprocessing the C++ must apply.

## Run

```bash
W=/path/LightFC_best.pth.tar; C=/path/experiments/lightfc/<name>.yaml; D=/path/got10k/train
# 1) float only: checks the split equals lib_orin forward; no Vitis AI needed; optional ONNX
python quantize_lightfc.py --part fusion_head --quant_mode float --export_onnx --weights $W --cfg_file $C --data_dir $D
# 2) Vitis AI PTQ (inside the Vitis AI PyTorch docker): calib then test, for each part
for p in backbone_template backbone_search fusion_head; do
  python quantize_lightfc.py --part $p --quant_mode calib --weights $W --cfg_file $C --data_dir $D
  python quantize_lightfc.py --part $p --quant_mode test  --weights $W --cfg_file $C --data_dir $D
done
# 3) compile each quantized xmodel for YOUR board (board-specific, arch.json comes with the DPU IP)
vai_c_xir -x quantize_lightfc_result/<part>/<Wrapper>_int.xmodel -a <arch.json> -o <out_dir> -n <part>
```

The script uses the exact `lib_orin` preprocessing: gray ×3, `sample_target` template from the GT box,
letterboxed search, and ImageNet mean/std. Template/search calibration pairs come from the same sequence and are
sampled like `lib_orin`'s `TUSampler`: the template is any visible frame, the search frame is any *other*
visible frame of the same object, with no gap limit unless `--max_gap` is set. Sources:
- `--data_dir`: GOT-10k train (`<seq>/*.jpg`, `groundtruth.txt`, `absence.label`)
- `--geotext_dir [--geotext_split train|val|all] [--geotext_anno train_box.json]`: GeoText-1652. Parsing
  mirrors `lib_orin/train/dataset/geotext.py`, and split ids come from `lib_orin/train/data_specs/geotext_*.txt`.
  Each (sequence, building id) is one track.

When both are given, a source is picked with probability proportional to its number of sequences
(`TUSampler` with `p_datasets=None`). `--template_mode rgb` switches the template to colour, as in `lib_orin`'s test `initialize()`.

## Smoke test (run this first, in your training env)
`python smoke_test.py` builds random weights, a placeholder config and fake GOT-10k / GeoText-1652 folders in a temp dir.
It then runs the float path for all parts (including the equivalence check against `lib_orin`'s forward) and,
if Vitis AI is absent, the calib/test control flow with a stub quantizer. It needs no checkpoint or dataset.

## Target-dependent (not hard-coded here)
- Step 3 and `pytorch_nndct` are Xilinx Vitis AI only. For another accelerator, start from the float ONNX
  (`--export_onnx`) and use that vendor's quantizer, calibrating with the same preprocessing.
- The Vitis AI version must match the DPU IP on the board (fingerprint). The old models were built with
  xcompiler 3.5.0 for `DPUCZDX8G_ISA1_B4096`.
- Bind fusion_head outputs **by name** (`conv5_ctr`, `conv5_offset`, `conv5_size`) on the C++ side, never by
  index: compilers may reorder outputs.
