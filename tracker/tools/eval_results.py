"""
Score a TrackerV2 run: mIoU and success rate (IoU > 0.5) of results.txt against the ground truth.
Frame 0 (the init box) is skipped; GT lines that are NaN or have w/h <= 0 are skipped.

Usage: python eval_results.py <out_dir>/results.txt <groundtruth.txt>
"""
import re
import sys

import numpy as np


def load_boxes(path, cols=4):
    boxes = []
    with open(path) as fp:
        for line in fp:
            vals = [v for v in re.split(r"[,\t ]+", line.strip()) if v]
            try:
                boxes.append([float(v) for v in vals[:cols]])
            except ValueError:
                boxes.append([np.nan] * cols)
    return np.array(boxes, dtype=np.float64)


def iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pred = load_boxes(sys.argv[1])
    scores = load_boxes(sys.argv[1], cols=5)[:, 4]
    gt = load_boxes(sys.argv[2])
    n = min(len(pred), len(gt))
    pred, gt, scores = pred[1:n], gt[1:n], scores[1:n]
    valid = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    ious = iou(pred[valid], gt[valid])
    print(f"frames: {valid.sum()} (of {n - 1})  mIoU: {ious.mean():.4f}  SR@0.5: {(ious > 0.5).mean():.4f}  "
          f"mean score: {np.nanmean(scores):.4f}")


if __name__ == "__main__":
    main()
