# DIFF_REPORT — `lib_new` (old) → `lib_orin` (source of truth)

Method: `diff -rqwB` for the file map, then `diff -ruNwB` (whitespace/blank-line changes ignored) on every
differing file; every hunk was read. Comment-only hunks are left out below. `lib_orin` is treated as correct.

> **Key finding, read this first.** `lib_new` is not just an older `lib_orin`. It also has **FPGA-port edits**
> that never made it into `lib_orin`: Hardsigmoid/Hardswish swaps, `forward_dpu()` split methods,
> `forward_search_backbone()` and `forward_fusion_head_dpu()`. So for these files the diff goes both ways.
> `lib_orin` wins (per your rules), but the DPU export entry points now have to live outside the library.
> Section 3 covers this.

---

## 1. Summary table

| File | Status | Impact | Affects inference? |
|---|---|---|---|
| `models/lightfc/fusion/ecm.py` → `pixel_wise_corr_fpga` | modified | **HIGH** | **yes** (template-kernel channel layout fixed) |
| `test/tracker/lightfc.py` → `track()` | modified | **HIGH** | **yes** (letterbox resize + box un-mapping) |
| `models/tracker_model.py` | modified | **HIGH** (export) | yes, export path (DPU split methods removed) |
| `models/lightfc/fusion/ecm.py` → `pwcorr_scf_sc_iab_sc_concat.forward_dpu` | removed | MED | export path only, numerics unchanged |
| `models/lightfc/head/erh.py` → `repn33_center_concat_geom` | modified | MED | yes: new `max_score` output; `forward_dpu` removed (numerics unchanged) |
| `train/data/base_functions.py` (grayscale now joint, p=1) | modified | MED | **indirectly**: the template must now be grayscale at inference too |
| `train/data/processing_utils.py` → `resize()` | modified | LOW | yes (accepts a single image; same math) |
| `models/lightfc/fusion/ecm.py` → `SE` (Hardsigmoid→Sigmoid), IAB act (Hardswish→GELU), `pwcorr_se_scf…` corr (fpga→matmul) | modified | LOW today / HIGH if you switch fusion | no: these modules are **not instantiated** by `LightFC` |
| `models/lightfc/head/erh.py` → `SE` (Hardsigmoid→Sigmoid) | modified | LOW | no: used only by `repn33_se_center_concat`, which isn't instantiated |
| `models/lightfc/fusion/ecm.py` → `pwcorr_2conv_sc_concat` | added | LOW (unused) | no: defined and exported but **not instantiated** |
| `models/lightfc/backbone/__init__.py` (+`tiny_vit_5m_224` import) | modified | LOW | no (MobileNetV2 is still the backbone) |
| `models/lightfc/__init__.py`, `fusion/__init__.py` | modified | LOW | no (import lists only) |
| `models/lightfc/salad/` | removed | LOW | no (never referenced by LightFC) |
| `test/parameter/lightfc.py` | modified | MED | yes, *which weights*: `LightFC_best.pth.tar` instead of `LightFC_ep%04d` |
| `test/tracker/lightfc.py` → `__init__` | modified | LOW | checkpoint path hard-coded to `/mnt/tupm/.../building_new_model/LightFC_best.pth.tar` |
| `train/data/processing.py` (`TUProcessing`) | modified | LOW | no (training-only; see §4) |
| `train/data/sampler.py` (+`TUSampler`) | modified | — | no (training-only) |
| `train/actors/lightfc.py` | modified | — | no (val visualization) |
| `train/train_script.py` | modified | — | no (fine-tunes from a hard-coded checkpoint) |
| `train/trainers/base_trainer.py`, `ltr_trainer.py` | modified | — | no (best-IoU checkpointing) |
| `train/dataset/{uav123,dronestagram,geotext,openlogo}.py`, `dataset/__init__.py`, `dataset/lasot.py` | added / modified | — | no |
| `train/admin/local.py`, `test/evaluation/{environment,local,datasets}.py` | modified | — | no (paths, dataset registry) |
| `test/analysis/{extract_results,plot_results}.py` | modified | — | no (analysis/visualization) |
| `train/data_specs/*.txt` (+geotext_*, lasot_val, trackingnet_classmap, 2×"copy") | added / modified | — | no |

Unchanged but **critical for the port** (identical in both libs, so they aren't diffs): `backbone/mobilnetv2.py`,
`repN33`, `conv_center_head`, `repn33_center_concat_geom.cal_bbox` / `get_score_map`, `sample_target`,
`Preprocessor`. Their exact behavior is listed in §5, because the C++ never implemented them (see the plan).

---

## 2. HIGH-impact changes

### 2.1 `pixel_wise_corr_fpga`: template kernel layout fixed (`fusion/ecm.py`)
This is the op `LightFC.fusion` (`pwcorr_scf_sc_iab_sc_concat`) uses.
```python
# lib_new
z_weight = z.reshape(b * hz * wz, c, 1, 1)                       # z is (b,c,hz,wz): scrambles c and (hz,wz)
# lib_orin
z_weight = z.permute(0, 2, 3, 1).reshape(b * hz * wz, c, 1, 1)   # kernel k=(i*wz+j) = z[:, :, i, j]
```
Semantics: output channel `k = i*wz + j` is `Σ_c z[c,i,j] · x[c,h,w]`. This now equals `pixel_wise_corr`
(matmul) exactly. The `lib_new` version fed each 1×1 kernel 96 values that cut across channels and positions,
so its output is a different function. Weights trained in `lib_orin` expect the corrected layout. With a
128² template there are 8×8 = **64 output channels** at 30×40.

### 2.2 `track()`: aspect-preserving letterbox instead of a stretch (`test/tracker/lightfc.py`)
```python
# lib_new
resized_image = cv.resize(image, (640, 480))
x = cx * W - w*W/2 ; ...                               # un-stretch with frame W,H
# lib_orin
resized_images, resize_infos = prutils.resize(image)   # scale=min(640/w,480/h), INTER_LINEAR, zero pad centred
scale, left, top = resize_infos[0]
cx_orig = (cx*640 - left)/scale ; cy_orig = (cy*480 - top)/scale
w_full = w*640/scale ; h_full = h*480/scale
x = cx_orig - w_full/2 ; y = cy_orig - h_full/2
```
Matches training (`TUProcessing` already letterboxed the search image in both versions). The padding is raw 0
**before** normalization, so padded pixels become `(0-mean)/std`.

### 2.3 DPU export split removed (`tracker_model.py`, `ecm.py`, `erh.py`)
`lib_new` added `LightFC.forward_search_backbone`, `LightFC.forward_fusion_head_dpu(corr, x_feat)`,
`pwcorr_scf_sc_iab_sc_concat.forward_dpu(corr, x)` and `repn33_center_concat_geom.forward_dpu(x)`
(the pre-sigmoid `raw_ctr, raw_offset, raw_size`). `lib_orin` has none of them; `forward`/`get_score_map` are
monolithic again. **Numerics are unchanged**: `get_score_map = clamp(sigmoid(raw), 1e-4, 1-1e-4)` for ctr and size,
with the offset left raw. But any export script that called these methods no longer works against `lib_orin`,
so it needs its own wrappers.

---

## 3. MEDIUM-impact changes

### 3.1 Head returns `max_score` (`erh.py: repn33_center_concat_geom.forward`)
```python
bbox, max_score = self.cal_bbox(score_map_ctr, size_map, offset_map, return_score=True)
out = {'pred_boxes': bbox, 'max_score': max_score, ...}
```
`max_score` is the post-sigmoid, clamped peak of the ctr map. `lib_orin`'s `track()` doesn't use it yet. It is
the only confidence signal the model has, and the natural input for any lost-target logic.

### 3.2 Grayscale augmentation is now joint and always on (`train/data/base_functions.py`, `processing.py`)
```python
# lib_new: search gray p=1, template gray p=0.05 (inside TUProcessing)
# lib_orin: transform_joint = Transform(RandomHorizontalFlip(0.5), ToGrayscale(probability=1))  -> both always gray
```
`ToGrayscale` = `cv.cvtColor(RGB2GRAY)` stacked to 3 channels. Consequence: the model is trained on **gray
templates and gray search images**. `lib_orin`'s `initialize()` still crops the template from the colour frame,
which is inconsistent with training (open question Q2 in the plan). The C++ already reads gray frames.

### 3.3 Checkpoint selection (`test/parameter/lightfc.py`, `test/tracker/lightfc.py`)
Inference now uses `LightFC_best.pth.tar` (best val mean-IoU, from the new trainer logic). The tracker
hard-codes `/mnt/tupm/LightFC/checkpoints/train/lightfc/building_new_model/LightFC_best.pth.tar` and loads it
with `strict` defaulting to True. This decides which weights you export.

### 3.4 Activation swaps in the SE/IAB modules (`ecm.py`, `erh.py`)
`SE.sigmoid`: Hardsigmoid → **Sigmoid**. IAB `act`: Hardswish → **GELU**. `pwcorr_se_scf_sc_iab_sc_concat.pw_corr`:
`pixel_wise_corr_fpga` → `pixel_wise_corr`. These are LOW today because `LightFC` instantiates
`pwcorr_scf_sc_iab_sc_concat` + `repn33_center_concat_geom`, and neither contains SE/IAB/GELU. They become
HIGH if you ever switch to the SE variants: Sigmoid and GELU are not accelerator-friendly.

---

## 4. Training-only changes (ignore for the tracker)
- `TUSampler` (new sampler with retry and frame indexing), and `TrackingSampler` → `TUSampler` in the loaders.
- `TUProcessing`: `min_area` 4² → 16² px, `visualize_num` 10 → 100, the `box` kwarg, removal of the internal gray transforms (moved to the joint transform, see §3.2).
- New datasets (UAV123, Dronestagram, GeoText, OpenLogo, a LaSOT val split) and their data_specs / paths.
- Trainer: best-mean-IoU tracking, `_best` checkpoints, `best_iou` stored in the checkpoint.
- `train_script.py`: builds `loader_val` twice and fine-tunes from `/mnt/tupm/.../building_new/LightFC_best.pth.tar`.
- Actor: validation box visualization.
- `test/analysis/*`: `skip_missing_seq=True`, result visualizations; `test/evaluation/*`: got10k paths and a `got10k_train` entry.
- `salad/` removed and `tiny_vit` import added: neither is used by `LightFC`.

---

## 5. Inference contract of `lib_orin` (what the C++ must reproduce)
Derived from `lib_orin/test/tracker/lightfc.py`, `test/tracker/data_utils.py`,
`train/data/processing_utils.py`, `models/**`.

| Stage | Exact behavior |
|---|---|
| Template crop | `sample_target(img, init_bbox, TEMPLATE_FACTOR, 128)`: `crop_sz=ceil(sqrt(w*h)*f)`; `x1=round(x+0.5w-0.5crop_sz)` (Python banker's rounding); `x2_pad=max(x2-W+1,0)` (note the **+1**); zero `BORDER_CONSTANT`; `cv.resize` bilinear to 128×128 |
| Search image | full frame (no search-region crop, no motion model) → gray → 3-ch → letterbox to 640×480 (`scale=min(640/w,480/h)`, `new=int(round(·))`, `top=pad_h//2`, `left=pad_w//2`, zero pad) |
| Normalization | `(px/255 - mean_c)/std_c`, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, RGB order (all equal for gray) |
| Backbone | MobileNetV2 truncated after stage `[6,96,3,1]`, stride 16, 96 ch, BN+ReLU6. z: 8×8×96, x: 30×40×96 |
| Correlation | `pixel_wise_corr_fpga` §2.1 → 64×30×40 |
| Fusion | `conv_down1`(64→32,1×1,no bias)+BN → repN33(32)+ReLU → `conv_down2`(32→16)+BN → repN33(16)+ReLU → `adjust`(16→96,1×1) → concat[adjust, x_feat] = 192 ch. No residuals (those are in the unused `pwcorr_2conv_sc_concat`) |
| Head | `repn33_center_concat_geom`: ctr branch → 1 ch; shared geom branch → offset 2 ch (raw) and size 2 ch; `sigmoid` + clamp `[1e-4, 1-1e-4]` on ctr and size |
| Decode | `idx = argmax(ctr)` (flattened, row-major 30×40); `idx_y = idx // feat_sz[0]`, `idx_x = idx % feat_sz[1]`; `cx=(idx_x+off[0])/feat_sz[1]`, `cy=(idx_y+off[1])/feat_sz[0]`, `w=size[0]`, `h=size[1]` |
| Un-map | §2.2; `state=[x,y,w,h]` in original pixels. **No** clipping, smoothing, hanning window, scale/ratio penalty, template update or failure handling |

⚠ `idx_y = idx // feat_sz[0]`: with `feat_sz=(30,40)` this divides by **30**, not by the row width 40. It is
identical in both libs and the model was trained through this decode. It's listed as open question Q1 in the plan.
