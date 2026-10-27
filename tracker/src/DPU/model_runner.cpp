#include "model_runner.h"

#include <sstream>

int ModelRunner::findInput(const std::string& part) const {
    for (size_t i = 0; i < inputCount(); i++) {
        if (inputName(i).find(part) != std::string::npos) return static_cast<int>(i);
    }
    return -1;
}

int ModelRunner::findOutput(const std::string& part) const {
    for (size_t i = 0; i < outputCount(); i++) {
        if (outputName(i).find(part) != std::string::npos) return static_cast<int>(i);
    }
    return -1;
}

size_t itemSize(const std::vector<int>& shape) {
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); i++) n *= static_cast<size_t>(shape[i]);
    return n;
}

std::string shapeToString(const std::vector<int>& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); i++) ss << (i ? "," : "") << shape[i];
    ss << "]";
    return ss.str();
}
