// End-to-end test of AITracker on the host: the accelerator is replaced by fake ModelRunners that
// return the golden tensors from tools/make_golden.py and check what the tracker feeds them.
// Covers preprocessing -> model I/O binding (by tensor name, shuffled order) -> correlation -> decode.
// Usage: ./test_tracker_flow <golden_dir>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "AItracker.h"

namespace {

struct Array
{
    std::vector<int> shape;
    std::vector<float> f;     // f32 arrays (or f64 converted)
    std::vector<uint8_t> u8;
};

std::map<std::string, Array> golden;
int failures = 0;

void loadGolden(const std::string& dir) {
    std::ifstream manifest(dir + "/manifest.txt");
    if (!manifest.is_open()) throw std::runtime_error("cannot open " + dir + "/manifest.txt");
    std::string line;
    while (std::getline(manifest, line)) {
        std::istringstream ss(line);
        std::string name, dtype;
        if (!(ss >> name >> dtype)) continue;
        Array a;
        int d;
        size_t n = 1;
        while (ss >> d) {
            a.shape.push_back(d);
            n *= static_cast<size_t>(d);
        }
        std::ifstream bin(dir + "/" + name + ".bin", std::ios::binary);
        if (dtype == "u8") {
            a.u8.resize(n);
            bin.read(reinterpret_cast<char*>(a.u8.data()), static_cast<std::streamsize>(n));
        } else if (dtype == "f32") {
            a.f.resize(n);
            bin.read(reinterpret_cast<char*>(a.f.data()), static_cast<std::streamsize>(n * 4));
        } else {
            std::vector<double> tmp(n);
            bin.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(n * 8));
            a.f.assign(tmp.begin(), tmp.end());
        }
        golden[name] = a;
    }
}

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    if (!ok) failures++;
    std::printf("[%s] %-40s max err = %.3e (tol %.1e)\n", ok ? " OK " : "FAIL", what.c_str(), err, tol);
}

double maxDiff(const float* a, const std::vector<float>& b, double norm = 1.0) {
    double m = 0;
    for (size_t i = 0; i < b.size(); i++) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m / norm;
}

std::vector<int> nhwc(const std::string& name) {
    std::vector<int> s = golden.at(name).shape;
    s.insert(s.begin(), 1);
    return s;
}

struct FakeTensor
{
    std::string name;
    std::string golden;   // golden array checked (inputs) or returned (outputs)
};

class FakeRunner : public ModelRunner
{
public:
    std::string model;
    std::vector<FakeTensor> ins, outs;

    size_t inputCount() const override { return ins.size(); }
    size_t outputCount() const override { return outs.size(); }
    std::vector<int> inputShape(size_t i) const override { return nhwc(ins.at(i).golden); }
    std::vector<int> outputShape(size_t i) const override { return nhwc(outs.at(i).golden); }
    std::string inputName(size_t i) const override { return ins.at(i).name; }
    std::string outputName(size_t i) const override { return outs.at(i).name; }

    void run(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) override {
        for (size_t i = 0; i < ins.size(); i++) {
            const auto& g = golden.at(ins[i].golden).f;
            double norm = 1.0;
            if (ins[i].golden == "corr") {
                for (float v : g) norm = std::max(norm, static_cast<double>(std::fabs(v)));
            }
            report(model + " input '" + ins[i].golden + "'", maxDiff(inputs[i], g, norm),
                   ins[i].golden == "corr" ? 1e-5 : 1e-5);
        }
        for (size_t i = 0; i < outs.size(); i++) {
            const auto& g = golden.at(outs[i].golden).f;
            std::memcpy(outputs[i], g.data(), g.size() * sizeof(float));
        }
    }
};

} // namespace

// Link-time replacement of the accelerator backend (dpu.cpp is not linked into this test).
std::unique_ptr<ModelRunner> createModelRunner(const std::string& modelPath) {
    auto r = std::make_unique<FakeRunner>();
    r->model = modelPath.substr(modelPath.find_last_of('/') + 1);
    if (r->model == "backbone_template.xmodel") {
        r->ins = {{"BackboneWrapper__input_0_fix", "template_norm"}};
        r->outs = {{"BackboneWrapper__ret_fix", "z_feat"}};
    } else if (r->model == "backbone_search.xmodel") {
        r->ins = {{"BackboneWrapper__input_0_fix", "search_norm"}};
        r->outs = {{"BackboneWrapper__ret_fix", "x_feat"}};
    } else {
        // deliberately NOT in the order the tracker expects: binding must go by name
        r->ins = {{"FusionHeadWrapper__input_1_fix", "x_feat"}, {"FusionHeadWrapper__input_0_fix", "corr"}};
        r->outs = {{"FusionHeadWrapper__head__Conv2d_conv5_size__ret_fix", "raw_size"},
                   {"FusionHeadWrapper__head__Conv2d_conv5_ctr__ret_fix", "raw_ctr"},
                   {"FusionHeadWrapper__head__Conv2d_conv5_offset__ret_fix", "raw_offset"}};
    }
    return r;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::printf("usage: %s <golden_dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    loadGolden(dir);

    LightFCConfig cfg;
    cfg.model_dir = "fake";
    cfg.template_size = golden.at("template_crop").shape[0];
    cfg.template_factor = golden.at("template_factor").f[0];
    cfg.template_gray = golden.at("template_gray").f[0] > 0.5f;
    cfg.search_height = golden.at("search_lb").shape[0];
    cfg.search_width = golden.at("search_lb").shape[1];
    cfg.feat_channels = golden.at("x_feat").shape[2];

    AITracker tracker(cfg);
    if (tracker.init() != InitState::Success) {
        std::printf("init failed: %s\n", tracker.lastError().c_str());
        return 1;
    }
    const auto& b = golden.at("init_box").f;
    tracker.initialize(cv::imread(dir + "/frame_t.png", cv::IMREAD_COLOR), cv::Rect2f(b[0], b[1], b[2], b[3]));
    TrackResult r = tracker.update(cv::imread(dir + "/frame_s.png", cv::IMREAD_COLOR));

    const auto& gb = golden.at("box").f;
    report("tracker box (original px)",
           std::max({std::fabs(r.rect_.x - gb[0]), std::fabs(r.rect_.y - gb[1]), std::fabs(r.rect_.width - gb[2]),
                     std::fabs(r.rect_.height - gb[3])}), 1e-2);
    report("tracker score", std::fabs(r.score_ - golden.at("score").f[0]), 1e-6);
    report("tracker state == Tracking", r.trackState_ == TrackerState::Tracking ? 0 : 1, 0);

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
