#ifndef LIGHTFC_OPS_H
#define LIGHTFC_OPS_H

#include <opencv2/opencv.hpp>

/*
 * Host-side LightFC operations. They run on the CPU between the accelerator calls and must
 * reproduce lib_orin numerically, so each function names the lib_orin function it mirrors.
 * No accelerator/VART dependency: these build and are tested on any machine with OpenCV.
 *
 * Tensor layout convention: NHWC (batch 1), float32, which is what DPU/most NPU runtimes expose.
 */
namespace lightfc {

struct LetterboxInfo
{
    double scale = 1.0;   // resized = original * scale
    int left = 0;         // horizontal zero padding before the image
    int top = 0;          // vertical zero padding before the image
};

struct Decoded
{
    cv::Rect2f box;       // x, y, w, h in ORIGINAL image pixels (not clipped, like lib_orin)
    float score = 0.f;    // max of clamp(sigmoid(ctr)) = lib_orin head 'max_score'
    int index = 0;        // argmax index in the flattened (featH x featW) map
};

// BGR/RGB/gray frame -> single-channel gray (cv::COLOR_BGR2GRAY == RGB2GRAY on an RGB frame).
cv::Mat toGray(const cv::Mat& img, bool isBgr = true);

// lib_orin/train/data/processing_utils.py:sample_target (output_sz given, no mask).
// Returns the (outSz x outSz) crop; resizeFactor = outSz / crop_sz.
cv::Mat sampleTarget(const cv::Mat& img, const cv::Rect2f& box, float searchAreaFactor, int outSz,
                     float* resizeFactor = nullptr);

// lib_orin/train/data/processing_utils.py:resize (aspect-preserving resize + centred zero padding).
cv::Mat letterbox(const cv::Mat& img, int outW, int outH, LetterboxInfo& info);

// lib_orin/test/tracker/data_utils.py:Preprocessor.process / lightfc.py:track normalization:
// out[(y*W + x)*3 + c] = (px/255 - mean[c]) / std[c]. A 1-channel image is replicated to 3 channels
// (== np.stack([gray, gray, gray], axis=2)).
void normalizeToNHWC(const cv::Mat& img, const float mean[3], const float stdv[3], float* out);

// lib_orin/models/lightfc/fusion/ecm.py:pixel_wise_corr_fpga (permute-fixed version), NHWC:
//   out[(h*W + w)*K + k] = sum_c z[k*C + c] * x[(h*W + w)*C + c],  k = i*zw + j,  K = zh*zw.
void pixelwiseCorr(const float* z, int zh, int zw, const float* x, int h, int w, int c, float* out);

// lib_orin/models/lightfc/head/erh.py:repn33_center_concat_geom.get_score_map._sigmoid + cal_bbox,
// then lib_orin/test/tracker/lightfc.py:track un-letterboxing. Raw (pre-sigmoid) NHWC maps:
// ctr (fh, fw, 1), offset (fh, fw, 2) [x, y], size (fh, fw, 2) [w, h].
Decoded decode(const float* rawCtr, const float* rawOffset, const float* rawSize, int fh, int fw,
               int searchW, int searchH, const LetterboxInfo& info);

} // namespace lightfc

#endif // LIGHTFC_OPS_H
