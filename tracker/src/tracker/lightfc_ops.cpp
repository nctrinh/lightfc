#include "lightfc_ops.h"

#include <algorithm>
#include <cmath>

namespace lightfc {

// Python's round() is round-half-to-even; std::nearbyint uses the current rounding mode, which is
// FE_TONEAREST (half-to-even) by default.
static inline long pyRound(double v) {
    return static_cast<long>(std::nearbyint(v));
}

// Python floor division for the non-negative operands used here.
static inline int floorDiv2(int v) {
    return v >= 0 ? v / 2 : -((-v + 1) / 2);
}

cv::Mat toGray(const cv::Mat& img, bool isBgr) {
    if (img.channels() == 1) return img;
    cv::Mat gray;
    // lib_orin: cv.cvtColor(image, cv.COLOR_RGB2GRAY) on an RGB frame == COLOR_BGR2GRAY on the same
    // frame stored as BGR (same weights applied to the same physical channels).
    cv::cvtColor(img, gray, isBgr ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    return gray;
}

cv::Mat sampleTarget(const cv::Mat& img, const cv::Rect2f& box, float searchAreaFactor, int outSz,
                     float* resizeFactor) {
    // lib_orin/train/data/processing_utils.py:sample_target, op by op. Python floats are doubles.
    const double x = box.x, y = box.y, w = box.width, h = box.height;

    // crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    const long cropSz = static_cast<long>(std::ceil(std::sqrt(w * h) * static_cast<double>(searchAreaFactor)));
    if (cropSz < 1) {
        throw std::runtime_error("sampleTarget: too small bounding box");  // Python: raise Exception
    }

    // x1 = round(x + 0.5 * w - crop_sz * 0.5); x2 = x1 + crop_sz (same for y)
    const long x1 = pyRound(x + 0.5 * w - cropSz * 0.5);
    const long x2 = x1 + cropSz;
    const long y1 = pyRound(y + 0.5 * h - cropSz * 0.5);
    const long y2 = y1 + cropSz;

    // x2_pad = max(x2 - im.shape[1] + 1, 0): note the +1 - the last image column/row is never
    // copied and is zero-padded instead when the crop reaches it. Mirrored on purpose.
    const long x1Pad = std::max(0L, -x1);
    const long x2Pad = std::max(x2 - img.cols + 1, 0L);
    const long y1Pad = std::max(0L, -y1);
    const long y2Pad = std::max(y2 - img.rows + 1, 0L);

    // im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    const long cx0 = x1 + x1Pad, cx1 = x2 - x2Pad;
    const long cy0 = y1 + y1Pad, cy1 = y2 - y2Pad;

    cv::Mat padded;
    if (cx1 > cx0 && cy1 > cy0) {
        cv::Mat crop = img(cv::Range(static_cast<int>(cy0), static_cast<int>(cy1)),
                           cv::Range(static_cast<int>(cx0), static_cast<int>(cx1)));
        // cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT) -> value 0.
        // BORDER_ISOLATED is required: `crop` is a ROI, and without it OpenCV C++ fills the border with the
        // real pixels outside the ROI. A numpy slice has no parent, so Python always pads with zeros.
        cv::copyMakeBorder(crop, padded, static_cast<int>(y1Pad), static_cast<int>(y2Pad),
                           static_cast<int>(x1Pad), static_cast<int>(x2Pad), cv::BORDER_CONSTANT | cv::BORDER_ISOLATED,
                           cv::Scalar::all(0));
    } else {
        // The crop lies completely outside the image. Python's empty slice makes copyMakeBorder
        // fail; return an all-zero crop instead of crashing the tracker.
        padded = cv::Mat::zeros(static_cast<int>(cropSz), static_cast<int>(cropSz), img.type());
    }

    // resize_factor = output_sz / crop_sz; cv.resize(im_crop_padded, (output_sz, output_sz)) (INTER_LINEAR)
    if (resizeFactor) *resizeFactor = static_cast<float>(static_cast<double>(outSz) / cropSz);
    cv::Mat out;
    cv::resize(padded, out, cv::Size(outSz, outSz), 0, 0, cv::INTER_LINEAR);
    return out;
}

cv::Mat letterbox(const cv::Mat& img, int outW, int outH, LetterboxInfo& info) {
    // lib_orin/train/data/processing_utils.py:resize
    const int h = img.rows, w = img.cols;
    // scale = min(640 / w, 480 / h)
    const double scale = std::min(static_cast<double>(outW) / w, static_cast<double>(outH) / h);
    // new_w, new_h = int(round(w * scale)), int(round(h * scale))
    const int newW = static_cast<int>(pyRound(w * scale));
    const int newH = static_cast<int>(pyRound(h * scale));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // top = pad_h // 2; bottom = pad_h - top; left = pad_w // 2; right = pad_w - left
    const int padW = outW - newW, padH = outH - newH;
    const int top = floorDiv2(padH), bottom = padH - top;
    const int left = floorDiv2(padW), right = padW - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    info.scale = scale;
    info.left = left;
    info.top = top;
    return padded;
}

void normalizeToNHWC(const cv::Mat& img, const float mean[3], const float stdv[3], float* out) {
    // ((img_tensor / 255.0) - mean) / std in float32 (torch float tensors), channel order kept.
    CV_Assert(img.depth() == CV_8U && (img.channels() == 1 || img.channels() == 3));
    const int ch = img.channels();
    for (int yy = 0; yy < img.rows; yy++) {
        const uint8_t* row = img.ptr<uint8_t>(yy);
        float* dst = out + static_cast<size_t>(yy) * img.cols * 3;
        for (int xx = 0; xx < img.cols; xx++) {
            for (int c = 0; c < 3; c++) {
                const float px = static_cast<float>(row[xx * ch + (ch == 1 ? 0 : c)]);
                dst[xx * 3 + c] = (px / 255.0f - mean[c]) / stdv[c];
            }
        }
    }
}

void pixelwiseCorr(const float* z, int zh, int zw, const float* x, int h, int w, int c, float* out) {
    // lib_orin/models/lightfc/fusion/ecm.py:pixel_wise_corr_fpga
    //   z_weight = z.permute(0, 2, 3, 1).reshape(hz * wz, c, 1, 1)   -> kernel k = z[:, i, j], k = i*wz + j
    //   out = conv2d(x, z_weight)                                    -> out[k, h, w] = sum_c z[c,i,j] x[c,h,w]
    // With NHWC z (zh, zw, c), kernel k is the contiguous vector z + k*c.
    const int K = zh * zw;
    cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& rows) {
        for (int yy = rows.start; yy < rows.end; yy++) {
            for (int xx = 0; xx < w; xx++) {
                const float* xv = x + (static_cast<size_t>(yy) * w + xx) * c;
                float* o = out + (static_cast<size_t>(yy) * w + xx) * K;
                for (int k = 0; k < K; k++) {
                    const float* zv = z + static_cast<size_t>(k) * c;
                    float acc = 0.f;
                    for (int ci = 0; ci < c; ci++) acc += zv[ci] * xv[ci];
                    o[k] = acc;
                }
            }
        }
    });
}

static inline float sigmoidClamp(float v) {
    // erh.py get_score_map._sigmoid: torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    const float s = 1.f / (1.f + std::exp(-v));
    return std::min(std::max(s, 1e-4f), 1.f - 1e-4f);
}

Decoded decode(const float* rawCtr, const float* rawOffset, const float* rawSize, int fh, int fw,
               int searchW, int searchH, const LetterboxInfo& info) {
    // erh.py:cal_bbox: max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1)
    // First maximum wins (strict >), like torch.max on CPU. Comparing the clamped sigmoid (not the
    // raw logit) keeps the ties created by the clamp identical to lib_orin.
    const int n = fh * fw;
    int idx = 0;
    float best = sigmoidClamp(rawCtr[0]);
    for (int i = 1; i < n; i++) {
        const float s = sigmoidClamp(rawCtr[i]);
        if (s > best) {
            best = s;
            idx = i;
        }
    }

    // idx_y = idx // self.feat_sz[0]; idx_x = idx % self.feat_sz[1]   with feat_sz = (fh, fw).
    // NOTE (TRACKER_UPDATE_PLAN.md Q1): lib_orin divides by feat_sz[0] (= fh = 30), not by the row
    // width fw (= 40). The model was trained through this decode, so it is mirrored exactly.
    const int idxY = idx / fh;
    const int idxX = idx % fw;

    // size = size_map.gather(idx) (sigmoid), offset = offset_map.gather(idx) (raw); NHWC channel 0/1
    const float offX = rawOffset[static_cast<size_t>(idx) * 2 + 0];
    const float offY = rawOffset[static_cast<size_t>(idx) * 2 + 1];
    const float sizeW = sigmoidClamp(rawSize[static_cast<size_t>(idx) * 2 + 0]);
    const float sizeH = sigmoidClamp(rawSize[static_cast<size_t>(idx) * 2 + 1]);

    // bbox = [(idx_x + offset_x) / feat_sz[1], (idx_y + offset_y) / feat_sz[0], w, h]  (float32)
    const float cx = (static_cast<float>(idxX) + offX) / static_cast<float>(fw);
    const float cy = (static_cast<float>(idxY) + offY) / static_cast<float>(fh);

    // lib_orin/test/tracker/lightfc.py:track (Python doubles from .tolist()):
    //   cx_pad, cy_pad = cx * 640, cy * 480; w_pad, h_pad = w * 640, h * 480
    //   cx_orig = (cx_pad - left) / scale; w_full = w_pad / scale; x = cx_orig - w_full / 2
    const double cxPad = static_cast<double>(cx) * searchW, cyPad = static_cast<double>(cy) * searchH;
    const double wPad = static_cast<double>(sizeW) * searchW, hPad = static_cast<double>(sizeH) * searchH;
    const double cxOrig = (cxPad - info.left) / info.scale;
    const double cyOrig = (cyPad - info.top) / info.scale;
    const double wFull = wPad / info.scale;
    const double hFull = hPad / info.scale;

    Decoded d;
    d.box = cv::Rect2f(static_cast<float>(cxOrig - wFull / 2), static_cast<float>(cyOrig - hFull / 2),
                       static_cast<float>(wFull), static_cast<float>(hFull));
    d.score = best;
    d.index = idx;
    return d;
}

} // namespace lightfc
