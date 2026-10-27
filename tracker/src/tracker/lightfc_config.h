#ifndef LIGHTFC_CONFIG_H
#define LIGHTFC_CONFIG_H

#include <string>

/*
 * LightFC tracker configuration. Defaults mirror lib_orin; everything that depends on the
 * experiment yaml, the exported models or the target is read from a key=value file
 * (see tracker/config/lightfc.cfg), so nothing board-specific is compiled in.
 *
 * Source of each default:
 *   template_size / template_factor : cfg.TEST.TEMPLATE_SIZE / TEMPLATE_FACTOR (experiment yaml)
 *   search_width/height             : lib_orin/train/data/processing_utils.py:resize (640x480)
 *   stride, feat_channels           : MobileNetV2 truncated after the 96-ch stage (stride 16)
 *   mean/stdv                       : lib_orin/test/tracker/lightfc.py:lightFC.__init__ (ImageNet)
 */
struct LightFCConfig
{
    // models (paths relative to model_dir unless absolute)
    std::string model_dir = ".";
    std::string backbone_template_model = "backbone_template.xmodel";
    std::string backbone_search_model = "backbone_search.xmodel";
    std::string fusion_head_model = "fusion_head.xmodel";

    // tensor-name substrings used to bind fusion_head I/O (compilers may reorder tensors)
    std::string fusion_corr_input = "input_0";      // FusionHeadWrapper.forward(corr, x_feat)
    std::string fusion_xfeat_input = "input_1";
    std::string out_ctr = "conv5_ctr";
    std::string out_offset = "conv5_offset";
    std::string out_size = "conv5_size";

    // geometry
    int template_size = 128;
    float template_factor = -1.f;   // REQUIRED: no default in lib_orin outside the yaml
    int search_width = 640;
    int search_height = 480;
    int stride = 16;
    int feat_channels = 96;

    // preprocessing
    bool template_gray = true;      // training uses ToGrayscale(p=1); "rgb" reproduces lib_orin test initialize()
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};

    // runtime
    float score_threshold = -1.f;   // < 0: disabled (lib_orin has no lost-target logic)
    bool show = false;

    int templateFeat() const { return template_size / stride; }
    int numKernel() const { return templateFeat() * templateFeat(); }
    int featH() const { return search_height / stride; }
    int featW() const { return search_width / stride; }
    std::string modelPath(const std::string& name) const;

    // Parse `key = value` lines ('#' comments). Unknown keys are errors. Returns false + err.
    bool load(const std::string& path, std::string& err);
    // Consistency checks (template_factor set, sizes divisible by stride, ...).
    bool validate(std::string& err) const;
};

#endif // LIGHTFC_CONFIG_H
