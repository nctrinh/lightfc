#include "AItracker.h"

#include <stdexcept>

AITracker::AITracker(const LightFCConfig& cfg) : cfg(cfg) {
}

bool AITracker::checkShape(const std::string& what, const std::vector<int>& shape, int h, int w, int c) {
    // NHWC with batch first; only the per-item dims are checked (the DPU batch may be > 1)
    if (shape.size() == 4 && shape[1] == h && shape[2] == w && shape[3] == c) return true;
    error = what + ": expected [N," + std::to_string(h) + "," + std::to_string(w) + "," + std::to_string(c) +
            "], model has " + shapeToString(shape) + " (config and exported model disagree)";
    return false;
}

InitState AITracker::init() {
    std::string err;
    if (!cfg.validate(err)) {
        error = err;
        return InitState::Failure;
    }
    try {
        backboneTemplate = createModelRunner(cfg.modelPath(cfg.backbone_template_model));
        backboneSearch = createModelRunner(cfg.modelPath(cfg.backbone_search_model));
        fusionHead = createModelRunner(cfg.modelPath(cfg.fusion_head_model));
    } catch (const std::exception& e) {
        error = std::string("cannot load models: ") + e.what();
        return InitState::Failure;
    }

    const int T = cfg.template_size, tf = cfg.templateFeat(), C = cfg.feat_channels;
    const int fh = cfg.featH(), fw = cfg.featW(), K = cfg.numKernel();

    if (backboneTemplate->inputCount() != 1 || backboneTemplate->outputCount() != 1 ||
        backboneSearch->inputCount() != 1 || backboneSearch->outputCount() != 1 ||
        fusionHead->inputCount() != 2 || fusionHead->outputCount() != 3) {
        error = "unexpected number of model inputs/outputs (expected 1/1, 1/1, 2/3)";
        return InitState::Failure;
    }

    // fusion_head inputs by name; fall back to the channel count when the names do not match
    corrIn = fusionHead->findInput(cfg.fusion_corr_input);
    xfeatIn = fusionHead->findInput(cfg.fusion_xfeat_input);
    if ((corrIn < 0 || xfeatIn < 0 || corrIn == xfeatIn) && K != C) {
        corrIn = fusionHead->inputShape(0).back() == K ? 0 : 1;
        xfeatIn = 1 - corrIn;
    }
    ctrOut = fusionHead->findOutput(cfg.out_ctr);
    offsetOut = fusionHead->findOutput(cfg.out_offset);
    sizeOut = fusionHead->findOutput(cfg.out_size);
    if (corrIn < 0 || xfeatIn < 0 || corrIn == xfeatIn || ctrOut < 0 || offsetOut < 0 || sizeOut < 0) {
        error = "cannot bind fusion_head tensors by name: check fusion_*_input / out_* in the config";
        return InitState::Failure;
    }

    if (!checkShape("backbone_template input", backboneTemplate->inputShape(0), T, T, 3) ||
        !checkShape("backbone_template output", backboneTemplate->outputShape(0), tf, tf, C) ||
        !checkShape("backbone_search input", backboneSearch->inputShape(0), cfg.search_height, cfg.search_width, 3) ||
        !checkShape("backbone_search output", backboneSearch->outputShape(0), fh, fw, C) ||
        !checkShape("fusion_head corr input", fusionHead->inputShape(corrIn), fh, fw, K) ||
        !checkShape("fusion_head x_feat input", fusionHead->inputShape(xfeatIn), fh, fw, C) ||
        !checkShape("fusion_head ctr output", fusionHead->outputShape(ctrOut), fh, fw, 1) ||
        !checkShape("fusion_head offset output", fusionHead->outputShape(offsetOut), fh, fw, 2) ||
        !checkShape("fusion_head size output", fusionHead->outputShape(sizeOut), fh, fw, 2)) {
        return InitState::Failure;
    }

    zInput.resize(static_cast<size_t>(T) * T * 3);
    xInput.resize(static_cast<size_t>(cfg.search_height) * cfg.search_width * 3);
    zFeat.resize(static_cast<size_t>(tf) * tf * C);
    xFeat.resize(static_cast<size_t>(fh) * fw * C);
    corr.resize(static_cast<size_t>(fh) * fw * K);
    rawCtr.resize(static_cast<size_t>(fh) * fw);
    rawOffset.resize(static_cast<size_t>(fh) * fw * 2);
    rawSize.resize(static_cast<size_t>(fh) * fw * 2);

    return InitState::Success;
}

void AITracker::initPreprocess(cv::Mat img, cv::Rect2f box) {
    // lib_orin/test/tracker/lightfc.py:initialize: sample_target(image, init_bbox, factor, size) +
    // Preprocessor.process. Frames arrive as BGR (cv::imread) or gray.
    cv::Mat src;
    if (cfg.template_gray || img.channels() == 1) {
        src = lightfc::toGray(img);
    } else {
        cv::cvtColor(img, src, cv::COLOR_BGR2RGB);   // lib_orin works on RGB frames
    }
    cv::Mat crop = lightfc::sampleTarget(src, box, cfg.template_factor, cfg.template_size);
    lightfc::normalizeToNHWC(crop, cfg.mean, cfg.stdv, zInput.data());
}

void AITracker::initialize(cv::Mat img, cv::Rect2f box) {
    if (!backboneTemplate) throw std::runtime_error("AITracker::initialize called before init()");
    this->initPreprocess(img, box);
    backboneTemplate->run({zInput.data()}, {zFeat.data()});
    this->rect = box;
    this->hasTemplate = true;
}

void AITracker::preprocess(cv::Mat img, lightfc::LetterboxInfo& info) {
    // lib_orin/test/tracker/lightfc.py:track: gray -> x3 -> prutils.resize (letterbox) -> normalize.
    // Letterboxing the single gray channel and replicating it in normalizeToNHWC is identical to
    // letterboxing the stacked 3-channel image (same per-channel resize and zero padding).
    cv::Mat lb = lightfc::letterbox(lightfc::toGray(img), cfg.search_width, cfg.search_height, info);
    lightfc::normalizeToNHWC(lb, cfg.mean, cfg.stdv, xInput.data());
}

void AITracker::xcorr() {
    const int tf = cfg.templateFeat();
    lightfc::pixelwiseCorr(zFeat.data(), tf, tf, xFeat.data(), cfg.featH(), cfg.featW(), cfg.feat_channels,
                           corr.data());
}

lightfc::Decoded AITracker::postprocess(const lightfc::LetterboxInfo& info) {
    return lightfc::decode(rawCtr.data(), rawOffset.data(), rawSize.data(), cfg.featH(), cfg.featW(),
                           cfg.search_width, cfg.search_height, info);
}

TrackResult AITracker::update(cv::Mat img) {
    TrackResult result;
    if (!hasTemplate) {
        result.trackState_ = TrackerState::Lost;
        result.rect_ = this->rect;
        return result;
    }

    lightfc::LetterboxInfo info;
    this->preprocess(img, info);
    backboneSearch->run({xInput.data()}, {xFeat.data()});

    this->xcorr();

    std::vector<const float*> fusionInputs(2);
    fusionInputs[corrIn] = corr.data();
    fusionInputs[xfeatIn] = xFeat.data();
    std::vector<float*> fusionOutputs(3);
    fusionOutputs[ctrOut] = rawCtr.data();
    fusionOutputs[offsetOut] = rawOffset.data();
    fusionOutputs[sizeOut] = rawSize.data();
    fusionHead->run(fusionInputs, fusionOutputs);

    lightfc::Decoded d = this->postprocess(info);
    this->rect = d.box;

    result.rect_ = d.box;
    result.score_ = d.score;
    // lib_orin has no lost-target logic; the optional threshold must be tuned on int8 scores.
    result.trackState_ = (cfg.score_threshold >= 0.f && d.score < cfg.score_threshold) ? TrackerState::Lost
                                                                                       : TrackerState::Tracking;
    return result;
}

AITracker::~AITracker() {
}
