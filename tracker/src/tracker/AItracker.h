#ifndef AITRACKER_H
#define AITRACKER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include "model_runner.h"
#include "tracker.h"
#include "lightfc_config.h"
#include "lightfc_ops.h"

#define GET_MOMENT	    		std::chrono::high_resolution_clock::now()
#define GET_DURATION(stop,start)	std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()

/*
 * LightFC tracker (port of lib_orin/test/tracker/lightfc.py:lightFC).
 *
 * initialize(): template = sample_target(frame, box)   -> backbone_template            -> z_feat (kept)
 * update():     search   = letterbox(gray(frame))      -> backbone_search              -> x_feat
 *               corr     = pixel_wise_corr_fpga(z_feat, x_feat)                        [host]
 *               raw      = fusion_head(corr, x_feat)                                   [accelerator]
 *               box      = decode(raw) (sigmoid/clamp, argmax, cal_bbox, un-letterbox) [host]
 * Like lib_orin, every frame is searched in full (no search region, window, smoothing or template
 * update); the previous state is not used.
 */
class AITracker: public Tracker
{
private:
    LightFCConfig cfg;
    std::unique_ptr<ModelRunner> backboneTemplate;
    std::unique_ptr<ModelRunner> backboneSearch;
    std::unique_ptr<ModelRunner> fusionHead;

    // fusion_head I/O indices, bound by tensor name (see LightFCConfig)
    int corrIn = -1, xfeatIn = -1;
    int ctrOut = -1, offsetOut = -1, sizeOut = -1;

    // NHWC float buffers
    std::vector<float> zInput, xInput;
    std::vector<float> zFeat, xFeat, corr;
    std::vector<float> rawCtr, rawOffset, rawSize;

    bool hasTemplate = false;
    cv::Rect2f rect;
    std::string error;

    bool checkShape(const std::string& what, const std::vector<int>& shape, int h, int w, int c);

    void initPreprocess(cv::Mat img, cv::Rect2f box);

    void preprocess(cv::Mat img, lightfc::LetterboxInfo& info);

    void xcorr();

    lightfc::Decoded postprocess(const lightfc::LetterboxInfo& info);

public:
    explicit AITracker(const LightFCConfig& cfg);
    ~AITracker() override;
    InitState init() override;
    void initialize(cv::Mat img, cv::Rect2f box) override;

    TrackResult update(cv::Mat img) override;

    const std::string& lastError() const { return error; }
};

#endif // AITRACKER_H
