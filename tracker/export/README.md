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
# 2b) task-level check of the int8 parts (needs the *_int.pt written by the test runs)
python eval_e2e.py --weights $W --cfg_file $C --geotext_dir $G --geotext_split val --samples 300
# 2c) if backbone accuracy is too low: AdaQuant fast finetune, pass the flag to BOTH runs
python quantize_lightfc.py --part backbone_search --quant_mode calib --fast_finetune ...
python quantize_lightfc.py --part backbone_search --quant_mode test  --fast_finetune ...
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

## Reading the quantization numbers
The per-part cosine similarity printed by `--quant_mode test` uses **float inputs** for each part, so errors
don't accumulate there. `eval_e2e.py` runs the whole pipeline (F / QB / QH / Q variants) and reports what
matters for tracking: IoU against GT and against float, whether the argmax matches, and the score error.

## Smoke test (run this first, in your training env)
`python smoke_test.py` builds random weights, a placeholder config and fake GOT-10k / GeoText-1652 folders in a temp dir.
It then runs the float path for all parts (including the equivalence check against `lib_orin`'s forward) and,
if Vitis AI is absent, the calib/test control flow with a stub quantizer. It needs no checkpoint or dataset.

## Target-dependent (not hard-coded here)
- **Match the board.** Read `xdputil query` on the board. Then (1) compile with an arch.json whose fingerprint equals
  the board's: `arch/DPUCZDX8G_ISA1_B3136_kit.json` is the current kit (B3136, fingerprint `0x101000056010406`,
  Vivado-flow DPU, not the KV260/ZCU102 B4096 presets), and (2) quantize and compile with the **same Vitis AI version
  as the board runtime** (current kit: VART 3.0 on PetaLinux 2022.2, so use the Vitis AI 3.0 docker).
- Step 3 and `pytorch_nndct` are Xilinx Vitis AI only. For another accelerator, start from the float ONNX
  (`--export_onnx`) and use that vendor's quantizer, calibrating with the same preprocessing.
- The Vitis AI version must match the DPU IP on the board (fingerprint). The old models were built with
  xcompiler 3.5.0 for `DPUCZDX8G_ISA1_B4096`.
- Bind fusion_head outputs **by name** (`conv5_ctr`, `conv5_offset`, `conv5_size`) on the C++ side, never by
  index: compilers may reorder outputs.
