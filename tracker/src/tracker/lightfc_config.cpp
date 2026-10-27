#include "lightfc_config.h"

#include <fstream>
#include <sstream>
#include <vector>

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

static bool parseFloats(const std::string& v, float* out, size_t n) {
    std::stringstream ss(v);
    std::string item;
    size_t i = 0;
    while (std::getline(ss, item, ',')) {
        if (i >= n) return false;
        out[i++] = std::stof(trim(item));
    }
    return i == n;
}

static bool parseBool(const std::string& v) {
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::string LightFCConfig::modelPath(const std::string& name) const {
    if (!name.empty() && name[0] == '/') return name;
    return model_dir + "/" + name;
}

bool LightFCConfig::load(const std::string& path, std::string& err) {
    std::ifstream file(path);
    if (!file.is_open()) {
        err = "cannot open config " + path;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(file, line)) {
        lineno++;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            err = path + ":" + std::to_string(lineno) + ": expected key = value";
            return false;
        }
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        try {
            if (key == "model_dir") model_dir = val;
            else if (key == "backbone_template_model") backbone_template_model = val;
            else if (key == "backbone_search_model") backbone_search_model = val;
            else if (key == "fusion_head_model") fusion_head_model = val;
            else if (key == "fusion_corr_input") fusion_corr_input = val;
            else if (key == "fusion_xfeat_input") fusion_xfeat_input = val;
            else if (key == "out_ctr") out_ctr = val;
            else if (key == "out_offset") out_offset = val;
            else if (key == "out_size") out_size = val;
            else if (key == "template_size") template_size = std::stoi(val);
            else if (key == "template_factor") template_factor = std::stof(val);
            else if (key == "search_width") search_width = std::stoi(val);
            else if (key == "search_height") search_height = std::stoi(val);
            else if (key == "stride") stride = std::stoi(val);
            else if (key == "feat_channels") feat_channels = std::stoi(val);
            else if (key == "template_mode") {
                if (val != "gray" && val != "rgb") throw std::invalid_argument("gray|rgb");
                template_gray = (val == "gray");
            }
            else if (key == "mean") { if (!parseFloats(val, mean, 3)) throw std::invalid_argument("3 floats"); }
            else if (key == "std") { if (!parseFloats(val, stdv, 3)) throw std::invalid_argument("3 floats"); }
            else if (key == "score_threshold") score_threshold = std::stof(val);
            else if (key == "show") show = parseBool(val);
            else {
                err = path + ":" + std::to_string(lineno) + ": unknown key '" + key + "'";
                return false;
            }
        } catch (const std::exception& e) {
            err = path + ":" + std::to_string(lineno) + ": bad value for '" + key + "' (" + e.what() + ")";
            return false;
        }
    }
    return validate(err);
}

bool LightFCConfig::validate(std::string& err) const {
    if (template_factor <= 0.f) {
        err = "template_factor is not set: copy TEST.TEMPLATE_FACTOR from the experiment yaml";
        return false;
    }
    if (stride <= 0 || template_size % stride != 0 || search_width % stride != 0 || search_height % stride != 0) {
        err = "template_size/search_width/search_height must be multiples of stride";
        return false;
    }
    for (int c = 0; c < 3; c++) {
        if (stdv[c] <= 0.f) {
            err = "std must be > 0";
            return false;
        }
    }
    return true;
}
