# TRACKER_UPDATE_PLAN — bring `tracker/` in line with `lib_orin`

Status: **Phase 2, read-only. Nothing under `tracker/` has been edited.**
Backup: `tracker/` is tracked in git (branch `claude/lib-orin-tracker-port-0jhcyw`, commit `58fea4c`, clean tree),
so every Phase 3 step will be its own commit and can be reverted.

---

## 0. Headline: `tracker/` does not implement `lib_new` either

Reading every C++/bash file shows that the C++ is a **different tracker family**: a NanoTrack/SiamBAN-style
Siamese tracker. It does not port LightFC. Specifically:

| Aspect | `tracker/` C++ today | `lib_new` / `lib_orin` LightFC |
|---|---|---|
| Models loaded | 4: `Init_model`, `Track_model1..3.xmodel` | 3 graphs: backbone(z), backbone(x), fusion+head |
| Template | 127×127, SiamFC context crop (`context_amount=0.5`), avg-colour pad, `warpAffine` | 128×128 `sample_target(factor)`, zero pad, `cv.resize` |
| Search | 287×287 crop around the previous state | **full frame**, letterboxed to 640×480, no search region |
| Correlation | depthwise xcorr, 128 ch, 4×4 kernel, 24×24 → 21×21 | pixel-wise (1×1 kernels, 64 = 8×8 template positions) on 30×40 |
| Head outputs | cls + ctr scores, 4 LTRB distances `exp(si·x+bi)·stride` | ctr (1), offset (2), size (2); sigmoid on ctr/size |
| Post-process | 15×15 Hanning window, scale/ratio penalty, LR smoothing, `restrict_box`, lost counter | argmax → decode → un-letterbox. Nothing else |
| Constants | `score_size=15` but `XCORR_OUT=21` (internally inconsistent) | feat 30×40, stride 16 |

The shipped `model/models_KV260/*.xmodel` match **neither** of these, nor either Python lib. I decoded the XIR
protobuf (scratch script; nothing modified):

| xmodel | input (NHWC) | output (NHWC) | built from |
|---|---|---|---|
| `backbone_template` | 1×128×128×**1** | 1×8×8×96 | `BackboneWrapper(MobileNetV2)` |
| `backbone_search` | 1×480×640×**1** | 1×30×40×96 | same |
| `fusion_head` | in0 1×30×40×64 (corr), in1 1×30×40×96 (x_feat) | ctr 30×40×1, offset 30×40×2, size 30×40×2 | **`pwcorr_se_scf_sc_iab_sc_concat` + `ChannelResidualMLP` + `repn31_shared_crmlp_center_concat_head`**: these classes exist in neither lib |

All three target `DPUCZDX8G_ISA1_B4096` (xcompiler 3.5.0). `run.sh` points to `model/models_ZCU102`, which
doesn't exist, and `main.cpp` asks for filenames that aren't shipped.
⇒ "Updating" the C++ in practice means **re-implementing the host-side LightFC pipeline**, plus an export
script, plus **re-exporting and re-quantizing every model**. The plan below is scoped that way.

## 1. File → component map

| `tracker/` file | Function(s) | Intended `lib_*` counterpart | Verdict |
|---|---|---|---|
| `src/main.cpp` | `main`, `ListImages`, `LoadRect` | `test/evaluation/running.py` loop (`initialize` then `track` per frame) | keep; fix model names/paths |
| `src/tracker/AItracker.h` | `hyper_params_tracker`, `#define XCORR_*`, `window[225]` | `cfg.TEST.*`, `cfg.MODEL.*`, `tracker.feat_sz` | replace |
| `AItracker.cpp` `initPreprocess`, `get_crop`, `get_subwindow*` | template crop | `processing_utils.sample_target` + `Preprocessor.process` | replace |
| `AItracker.cpp` `preprocess` | search crop | `prutils.resize` (letterbox) + normalization in `lightFC.track` | replace |
| `AItracker.cpp` `runInitModel` | z features | `LightFC.forward_backbone` | adapt |
| `AItracker.cpp` `xcorr_depthwise`, `conv2d_channel_first`, `ParallelXcorr_*` | correlation | `ecm.pixel_wise_corr_fpga` | replace |
| `AItracker.cpp` `runTrackModel(1..3)` | backbone(x), fusion/head | `backbone(x)`; `fusion` minus `pw_corr`, then `head` minus sigmoid | adapt (3 → 2 track graphs) |
| `AItracker.cpp` `postprocess`, `restrict_box`, `cal_sz`, `change`, `get_ctr_plane` | decode + state update | `head.get_score_map` (sigmoid/clamp) + `head.cal_bbox` + box un-mapping in `track` | replace |
| `src/tracker/tracker.{h,cpp}` | `Tracker` interface, `TrackResult` | `BaseTracker` | keep; add `score` to `TrackResult` |
| `src/DPU/dpu.{h,cpp}`, `src/common/common.{h,cpp}` | VART runner wrapper | none (Vitis-AI specific) | keep, put behind an interface (§4, R8) |
| `build.sh` | g++ build against VART/XIR/OpenCV | — | add new sources and a host-test target |
| `run.sh`, `trace_cfg.json` | demo command and profiling list | — | fix model dir and traced function names |
| *(none)* | export / quantize | `lib_new` `forward_search_backbone` / `forward_fusion_head_dpu` (removed in `lib_orin`) | **new** `tracker/export/` |

## 2. Diff item → required tracker change

| DIFF_REPORT item | Tracker file / function | Why |
|---|---|---|
| §2.1 corr kernel layout | new `pixelwise_corr()` replacing `xcorr_depthwise` | the fusion_head graph's input channel `k` must be `Σ_c z[i,j,c]·x[h,w,c]`, `k=i*8+j` |
| §2.2 letterbox + un-mapping | new `letterbox()` replacing `preprocess`; `decode()` replacing `postprocess` | weights expect an aspect-preserving 640×480 input; boxes must be un-mapped with `(scale,left,top)` |
| §2.3 export split removed | new `tracker/export/export_lightfc.py` (wrappers) | the Python entry points that produced the graphs are gone, so the export needs its own wrappers that call `lib_orin` modules |
| §3.1 `max_score` | `decode()` returns the score; `TrackResult.score_` | only confidence signal; used for optional lost detection (Q5) |
| §3.2 joint grayscale | template **and** search are grayscale in C++ | matches training (see Q2) |
| §3.3 best checkpoint | export script `--checkpoint` argument (no hard-coded path) | decides which weights get exported |
| §3.4 SE/GELU swaps | none (modules not instantiated) | flagged under risks in case fusion is changed |
| §5 contract (unchanged ops) | all of the above | the C++ never implemented LightFC |

## 3. Ordered checklist (highest risk first; each item = one commit)

Dependencies are marked **⇄**. Items 1–3 fix the interfaces that everything else depends on.

1. **[R-HIGH] Export script** `tracker/export/export_lightfc.py` (new, Python, imports `lib_orin` read-only by aliasing it as `lib`).
   - Build `LightFC(cfg, training=False)`, load the checkpoint, call `switch_to_deploy()` on all repN33 modules (same as `lightFC.__init__`).
   - `BackboneWrapper(x) = model.backbone(x)`, exported twice: 1×3×128×128 and 1×3×480×640.
   - `FusionHeadWrapper(corr, x_feat)`: `lib_orin` `pwcorr_scf_sc_iab_sc_concat.forward` **minus `pw_corr`**, then `head.get_score_map` **minus `_sigmoid`**. It calls the `lib_orin` submodules op by op, so it returns `raw_ctr, raw_offset, raw_size`.
   - Self-check: `sigmoid/clamp(FusionHeadWrapper(pixel_wise_corr_fpga(z,x), x))` must equal `model.forward_tracking` to fp32 tolerance.
   - Emit framework-neutral float **ONNX** + TorchScript. The quantize/compile step goes in a separate, clearly marked board-specific script (`export/quantize_vitis.sh`, Vitis AI only, untested here).
   - ⇄ fixes corr layout (item 4), input channels (item 3), pre-sigmoid outputs (item 5).
2. **[R-HIGH] Config replaces `hyper_params_tracker`.** New `LightFCConfig` struct: `template_size=128`, `template_factor` (**must be set, no default**, see Q4), `search_w=640`, `search_h=480`, `stride=16`, `feat_h=30`, `feat_w=40`, `feat_c=96`, `num_kernel=64`, `mean/std`, model filenames. Loaded from a small `key=value` file next to the models, so nothing is board-specific in code. Delete `XCORR_*`, `FILTER_NUMBER`, `KERNEL_*`, `window[225]`, `bi/si`, `penalty_k`, `window_influence`, `test_lr`, `context_amount`.
3. **[R-HIGH] Preprocessing** (mirrors `lib_orin` op by op, each function commented with its source):
   - `sample_target()` ← `train/data/processing_utils.py:sample_target` (ceil, half-even round, the `+1` pad quirk, zero pad, bilinear resize).
   - `letterbox()` ← `processing_utils.py:resize` (min-scale, `int(round())`, `pad//2`, zero pad). Returns `(scale,left,top)`.
   - `normalize_to_nhwc()` ← `test/tracker/lightfc.py:track` / `data_utils.Preprocessor` (`(p/255-mean_c)/std_c`, 3 identical gray channels → NHWC float).
   - ⇄ backbone input channel count (Q3).
4. **[R-HIGH] Correlation** `pixelwise_corr()` ← `fusion/ecm.py:pixel_wise_corr_fpga` (`lib_orin` version): NHWC `out[h][w][i*8+j] = Σ_c z[i][j][c]·x[h][w][c]`, i.e. a (1200×96)·(96×64) matmul on the host. Replaces `xcorr_depthwise` and its parallel loop (can keep `cv::parallel_for_` over rows).
5. **[R-MED] Decode** `decode()` ← `head/erh.py:get_score_map._sigmoid` + `cal_bbox` + `test/tracker/lightfc.py:track` un-mapping: sigmoid+clamp on ctr/size; first-max argmax; `idx_y=idx/feat_sz[0]`, `idx_x=idx%feat_sz[1]` **(quirk mirrored, Q1)**; offsets raw; un-letterbox; return `max_score`.
6. **[R-MED] Tracker flow** `AITracker::initialize/update`: 3 graphs (`backbone_template`, `backbone_search`, `fusion_head`) instead of 4. Remove the Hanning window, penalties, LR smoothing, `restrict_box` and the lost counters (not in `lib_orin`). `TrackerState::Lost` only when an **opt-in** `score_threshold` is configured (default off, Q5). No template update (none in `lib_orin`).
7. **[R-MED] Backend isolation (hardware-agnostic).** Small `ModelRunner` interface (`run(inputs NHWC float) → outputs NHWC float`, plus shape queries). The `VART_DPU` class becomes one implementation; the float↔int8 fix-point conversion is done inside it (R8). Tracker code never includes VART/XIR headers. `build.sh` keeps today's VART build as the default.
8. **[R-LOW] Driver and scripts.** `main.cpp`: 3 model names from config; read frames in colour and convert with `cv::COLOR_BGR2GRAY` (matches `cvtColor(RGB2GRAY)` in training; avoids libjpeg's own gray decode, Q10); write `x,y,w,h` results to a txt file for comparison. `run.sh`: model dir as an argument (no `models_ZCU102`). `trace_cfg.json`: new function names. `build.sh`: new sources and a `host_test` target.
9. **[R-LOW] Verification.** `tracker/tools/make_golden.py` (`lib_orin` + checkpoint → dumps the template crop, letterboxed search, normalized tensors, z/x features, corr, raw head outputs, decoded box as `.npy`/`.bin`). Plus `tracker/test/test_host_ops.cpp`: runs the C++ preprocessing/corr/decode on the same inputs (no accelerator needed) and reports max-abs errors. End-to-end check on the board: compare `results.txt` vs `lib_orin` boxes (IoU per frame).
10. **`CHANGELOG_TRACKER.md`.**

Build note: this container has g++/cmake but **no OpenCV, VART/XIR or PyTorch**. I can syntax-check the
VART-free translation units only if OpenCV can be installed. I won't claim anything compiles that I couldn't build.

## 4. Open questions / risks

**Decisions I need from you (my default in bold):**
- **Q1 – decode quirk.** `cal_bbox` uses `idx_y = idx // feat_sz[0]` (÷30) on a 40-wide map, so `cy` is wrong except in row 0. It's in both libs and in training. **Default: mirror it exactly** (you said don't invent behavior). Fixing it would also need a retrain/fine-tune and a `lib_orin` change on your side.
- **Q2 – template colour.** Training: template gray with p=1. `lib_orin` `initialize()`: colour template. C++: gray frames. **Default: gray** (matches training and the 1-channel camera path). Tell me if you want colour, which would also need colour frames in `main.cpp`.
- **Q3 – backbone input channels.** The old xmodels take **1 channel** (normalization presumably folded into conv1, but the wrapper code isn't in the repo). Folding is **not exact** at zero-padded borders (per-channel mean/std differ), and the letterbox bars are large padded areas. **Default: 3-channel input, exact.** It costs 3× input bandwidth. The 1-channel fold is an option you can choose if bandwidth matters.
- **Q4 – missing config.** `experiments/lightfc/<name>.yaml` isn't in the repo. Unknown: `TEST.TEMPLATE_FACTOR`, `TEST.TEMPLATE_SIZE` (128 inferred from the 8×8 xmodel), `MODEL.HEAD.PARAMS.feat_sz` (assumed `[30,40]`), `FUSION.PARAMS.num_kernel/adj_channel` (assumed 64/96), `DATA.MEAN/STD` (assumed ImageNet = test code), `BACKBONE.STRIDE` (16). **I will not guess `template_factor`**: the C++ refuses to start until it's set. Please send the yaml.
- **Q5 – lost-target handling.** `lib_orin` has none (always "tracking"). **Default: always Tracking, report `max_score`, optional threshold off.**
- **Q6 – which fusion produced your checkpoint?** `pwcorr_2conv_sc_concat` (new, adds residuals) and `pwcorr_scf_sc_iab_sc_concat` (used by `LightFC`) have **identical state_dict keys**, so `strict=True` loading cannot tell them apart. **Default: `pwcorr_scf_sc_iab_sc_concat`** (what `tracker_model.py` builds). Please confirm the best checkpoint wasn't trained with the 2conv variant.

**Re-export / re-quantization (mandatory):**
- R1. All three xmodels must be regenerated: wrong fusion/head architecture, possibly 1-channel input, and quantized against old weights. Nothing in C++ can be validated against them.
- R2. Interfaces that change together: corr layout (64 ch, `k=i*8+j`, NHWC) ⇄ fusion_head input0; x_feat 30×40×96 ⇄ fusion_head input1; 3-ch vs 1-ch ⇄ backbone inputs and C++ normalization; pre-sigmoid outputs ⇄ C++ sigmoid; template 128 ⇄ `num_kernel` 64 ⇄ `conv_down1` in-channels. Changing any of these means re-exporting and re-quantizing.
- R3. The output order of the fusion_head graph isn't guaranteed by the compiler. The C++ should bind outputs **by tensor name** (`getTensorShape(..., output_names)` already exists in `common.cpp`), not by index.

**Accelerator / quantization concerns:**
- R4. The correlation is a **dynamic-weight** conv (weights = template features). Most fixed-function NPUs/DPUs can't do it, so it runs on the host: about 7.4 M MAC/frame (1200×96×64). That's fine on a Cortex-A53/A72 with NEON (low ms). It's an option if your target supports runtime-weight matmul.
- R5. `concat[adjust(corr), x_feat]` merges two tensors with different dynamic ranges. Int8 accelerators force one shared scale, so expect accuracy loss there. Check it during QAT/PTQ.
- R6. The input to fusion_head is a host-computed float correlation, so it gets **re-quantized at the graph boundary**. Calibrate with real corr tensors (the export script will dump them).
- R7. MobileNetV2 at 640×480 is about 1 GMAC/frame (my estimate). This dominates latency and is the key number for sizing the board. Ops: conv/dw-conv/BN/ReLU6/add; all mainstream int8 accelerators support them.
- R8. Existing `dpu.cpp` hands **float** buffers (`cloneTensorBuffer` forces FLOAT) to a runner on the **DPU subgraph**, whose tensors are normally XINT8. Unless your Vitis AI version converts automatically, that is wrong. The fix (scale by `2^fix_point`) belongs in the VART backend and is Vitis-version-specific. I'll implement the standard Vitis AI 2.x/3.x pattern but **can't test it here**.
- R9. If you ever switch fusion/head to the SE/IAB variants: `lib_orin` now uses **Sigmoid** and **GELU** there (was Hardsigmoid/Hardswish). GELU isn't supported by DPUCZDX8G and many NPUs; Sigmoid is often CPU-fallback. `pixel_wise_corr` (matmul) vs `_fpga` (grouped conv) is numerically equal now.
- R10. repN33 must be re-parameterized (`switch_to_deploy`) **before** quantization, otherwise the two-branch sum is quantized separately.

**Numerical-parity details (low risk, handled in code):**
- Q9. Python `round()` is half-to-even (`sample_target`, `resize`); C++ uses `std::nearbyint`. `scale` is kept in `double`.
- Q10. Bilinear resize can differ by ±1 LSB across OpenCV versions/platforms (NEON paths). `IMREAD_GRAYSCALE` on JPEG uses libjpeg's own conversion; I'll use `imread` colour + `BGR2GRAY` instead. Argmax ties resolve to the first index (PyTorch on CUDA doesn't guarantee this, but ties are vanishingly rare after sigmoid).

**Decide before choosing a board:**
- Int8-only (DPU, most NPUs) vs fp16/bf16 support. This decides whether you need PTQ/QAT at all.
- Can the accelerator run a dynamic-weight matmul? (R4)
- Input bandwidth: 640×480×3 int8 ≈ 0.9 MB/frame (or ×1 with Q3's fold).
- Host CPU for corr + preprocessing; OpenCV availability on the image.
- Toolchain coupling: for Xilinx, the Vitis AI version must match the DPU IP/arch fingerprint (old models: VAI 3.5, `DPUCZDX8G_ISA1_B4096`).
- Latency/FPS target. Today the pipeline is "full-frame detection per frame", so you can't cut cost by shrinking a search region without changing the algorithm.

---
**STOP: waiting for your approval of this plan (and answers to Q1–Q6, or "use defaults") before Phase 3.**
