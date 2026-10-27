# TrackerV2: LightFC (lib_orin) on an accelerator

Pipeline per frame: `letterbox(gray)` → **backbone_search** → host `pixel_wise_corr_fpga` → **fusion_head** → host decode.
The template goes through **backbone_template** once, at `initialize()`.

```
config/lightfc.cfg      key=value config (model_dir, template_factor [required], ...)
export/                 PTQ/export of the 3 graphs from lib_orin (+ eval_e2e.py, smoke_test.py)
src/tracker/            AITracker, LightFCConfig, lightfc_ops (host ops mirroring lib_orin)
src/DPU/                ModelRunner interface + Vitis AI VART backend (dpu.cpp)
tools/make_golden.py    golden tensors from lib_orin for the host tests
test/                   test_host_ops, test_tracker_flow (OpenCV only)
```

```bash
./build.sh                     # TrackerV2 (needs Vitis AI VART/XIR + OpenCV, e.g. on the board or its sysroot)
./build.sh test                # host tests, OpenCV only
./run.sh config/lightfc.cfg <img_dir> <groundtruth.txt> [out_dir]   # out_dir/results.txt: x,y,w,h,score
```
See `../CHANGELOG_TRACKER.md` for what is verified and what is still target-specific.
