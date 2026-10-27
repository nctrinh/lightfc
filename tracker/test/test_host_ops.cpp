// Compares the C++ host ops (src/tracker/lightfc_ops.*) against golden tensors dumped from lib_orin by
// tracker/tools/make_golden.py. No accelerator needed. Usage: ./test_host_ops <golden_dir>
// Exit code 0 = all checks within tolerance.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "lightfc_ops.h"

struct Array
{
    std::string dtype;
    std::vector<int> shape;
    std::vector<char> bytes;

    size_t count() const {
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
    template <typename T> const T* as() const { return reinterpret_cast<const T*>(bytes.data()); }
};

static std::map<std::string, Array> loadGolden(const std::string& dir) {
    std::map<std::string, Array> arrays;
    std::ifstream manifest(dir + "/manifest.txt");
    if (!manifest.is_open()) throw std::runtime_error("cannot open " + dir + "/manifest.txt");
    std::string line;
    while (std::getline(manifest, line)) {
        std::istringstream ss(line);
        std::string name;
        Array a;
        if (!(ss >> name >> a.dtype)) continue;
        int d;
        while (ss >> d) a.shape.push_back(d);
        const size_t elem = a.dtype == "u8" ? 1 : (a.dtype == "f32" ? 4 : 8);
        a.bytes.resize(a.count() * elem);
        std::ifstream bin(dir + "/" + name + ".bin", std::ios::binary);
        if (!bin.read(a.bytes.data(), static_cast<std::streamsize>(a.bytes.size())))
            throw std::runtime_error("cannot read " + name + ".bin");
        arrays[name] = a;
    }
    return arrays;
}

static int failures = 0;

static void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    if (!ok) failures++;
    std::printf("[%s] %-38s max err = %.3e (tol %.1e)\n", ok ? " OK " : "FAIL", what.c_str(), err, tol);
}

// Max |a - b| between an 8-bit image (1 or 3 channels) and a golden u8 HWC3 array; for a 1-channel
// image every golden channel is compared to it (golden gray images are stacked x3).
static double maxDiffU8(const cv::Mat& img, const Array& g) {
    if (img.rows != g.shape[0] || img.cols != g.shape[1]) return 1e9;
    const uint8_t* ref = g.as<uint8_t>();
    const int ch = img.channels();
    double m = 0;
    for (int y = 0; y < img.rows; y++) {
        const uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < img.cols; x++)
            for (int c = 0; c < 3; c++) {
                const int v = row[x * ch + (ch == 1 ? 0 : c)];
                m = std::max(m, std::fabs(static_cast<double>(v) - ref[(static_cast<size_t>(y) * img.cols + x) * 3 + c]));
            }
    }
    return m;
}

static double maxDiffF32(const std::vector<float>& a, const Array& g) {
    if (a.size() != g.count()) return 1e9;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(static_cast<double>(a[i]) - g.as<float>()[i]));
    return m;
}

static double maxAbs(const Array& g) {
    double m = 0;
    for (size_t i = 0; i < g.count(); i++) m = std::max(m, std::fabs(static_cast<double>(g.as<float>()[i])));
    return m;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::printf("usage: %s <golden_dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const auto g = loadGolden(dir);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    const double* initBox = g.at("init_box").as<double>();
    const float factor = static_cast<float>(g.at("template_factor").as<double>()[0]);
    const bool templateGray = g.at("template_gray").as<double>()[0] > 0.5;
    const int T = g.at("template_crop").shape[0];
    const int SH = g.at("search_lb").shape[0], SW = g.at("search_lb").shape[1];

    // 1) template crop: sample_target on the gray (or RGB) frame
    cv::Mat frameT = cv::imread(dir + "/frame_t.png", cv::IMREAD_COLOR);
    cv::Mat srcT;
    if (templateGray) srcT = lightfc::toGray(frameT);
    else cv::cvtColor(frameT, srcT, cv::COLOR_BGR2RGB);
    cv::Mat crop = lightfc::sampleTarget(srcT, cv::Rect2f(static_cast<float>(initBox[0]), static_cast<float>(initBox[1]),
                                                          static_cast<float>(initBox[2]), static_cast<float>(initBox[3])),
                                         factor, T);
    // uint8 resize: identical OpenCV code gives 0; +-1 can appear across OpenCV versions/SIMD paths
    report("sample_target (template crop, u8)", maxDiffU8(crop, g.at("template_crop")), 1);

    // 2) search letterbox on the gray frame
    cv::Mat frameS = cv::imread(dir + "/frame_s.png", cv::IMREAD_COLOR);
    lightfc::LetterboxInfo info;
    cv::Mat lb = lightfc::letterbox(lightfc::toGray(frameS), SW, SH, info);
    report("letterbox (search image, u8)", maxDiffU8(lb, g.at("search_lb")), 1);
    const double* gInfo = g.at("lb_info").as<double>();
    report("letterbox scale/left/top",
           std::max({std::fabs(info.scale - gInfo[0]), std::fabs(info.left - gInfo[1]), std::fabs(info.top - gInfo[2])}),
           1e-12);

    // 3) normalization, fed with the golden uint8 images to isolate this step
    {
        const Array& gc = g.at("template_crop");
        cv::Mat golden(gc.shape[0], gc.shape[1], CV_8UC3, const_cast<uint8_t*>(gc.as<uint8_t>()));
        std::vector<float> out(static_cast<size_t>(gc.shape[0]) * gc.shape[1] * 3);
        lightfc::normalizeToNHWC(golden, mean, stdv, out.data());
        report("normalize template (f32 NHWC)", maxDiffF32(out, g.at("template_norm")), 1e-5);
    }
    {
        const Array& gs = g.at("search_lb");
        cv::Mat golden(gs.shape[0], gs.shape[1], CV_8UC3, const_cast<uint8_t*>(gs.as<uint8_t>()));
        std::vector<float> out(static_cast<size_t>(gs.shape[0]) * gs.shape[1] * 3);
        lightfc::normalizeToNHWC(golden, mean, stdv, out.data());
        report("normalize search (f32 NHWC)", maxDiffF32(out, g.at("search_norm")), 1e-5);
    }

    // 4) pixel-wise correlation from the golden backbone features
    const Array& z = g.at("z_feat");
    const Array& x = g.at("x_feat");
    const Array& gCorr = g.at("corr");
    std::vector<float> corr(gCorr.count());
    lightfc::pixelwiseCorr(z.as<float>(), z.shape[0], z.shape[1], x.as<float>(), x.shape[0], x.shape[1], x.shape[2],
                           corr.data());
    // fp32 sums of C products in a different order than cuDNN: relative tolerance
    report("pixel_wise_corr_fpga (f32 NHWC, rel)", maxDiffF32(corr, gCorr) / std::max(1.0, maxAbs(gCorr)), 1e-5);

    // 5) decode from the golden raw head outputs
    const Array& ctr = g.at("raw_ctr");
    const int fh = ctr.shape[0], fw = ctr.shape[1];
    lightfc::LetterboxInfo gl;
    gl.scale = gInfo[0];
    gl.left = static_cast<int>(gInfo[1]);
    gl.top = static_cast<int>(gInfo[2]);
    lightfc::Decoded d = lightfc::decode(ctr.as<float>(), g.at("raw_offset").as<float>(), g.at("raw_size").as<float>(),
                                         fh, fw, SW, SH, gl);
    const double* gBox = g.at("box").as<double>();
    report("decode argmax index", std::fabs(d.index - g.at("index").as<double>()[0]), 0);
    report("decode max_score", std::fabs(d.score - g.at("score").as<double>()[0]), 1e-6);
    report("decode box (original px)",
           std::max({std::fabs(d.box.x - gBox[0]), std::fabs(d.box.y - gBox[1]),
                     std::fabs(d.box.width - gBox[2]), std::fabs(d.box.height - gBox[3])}), 1e-2);

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
