#include "dpu.h"

#include <cmath>
#include <stdexcept>

static int fixPoint(const xir::Tensor* tensor) {
    if (!tensor->has_attr("fix_point")) {
        throw std::runtime_error("tensor " + tensor->get_name() + " has no fix_point (not an int8 DPU tensor)");
    }
    return tensor->get_attr<int>("fix_point");
}

VART_DPU::VART_DPU(const std::string& model_path, int timeout) : model(model_path), timeout(timeout) {
    graph = xir::Graph::deserialize(model_path);
    auto subgraph = get_dpu_subgraph(graph.get());
    if (subgraph.size() != 1) {
        throw std::runtime_error(model_path + ": expected 1 DPU subgraph, found " + std::to_string(subgraph.size()) +
                                 " (ops not supported by the DPU were split out; check the compiler log)");
    }

    /*create runner*/
    runner = vart::Runner::create_runner(subgraph[0], "run");
    inputTensors = runner->get_input_tensors();
    outputTensors = runner->get_output_tensors();

    for (auto tensor : inputTensors) {
        inputBuffers.emplace_back(tensor->get_element_num(), 0);
        inputScales.push_back(std::ldexp(1.f, fixPoint(tensor)));
    }
    for (auto tensor : outputTensors) {
        outputBuffers.emplace_back(tensor->get_element_num(), 0);
        outputScales.push_back(std::ldexp(1.f, -fixPoint(tensor)));
    }

    std::cout << "VART DPU Object is initialized: " << model_path << std::endl;
}

std::vector<int> VART_DPU::inputShape(size_t index) const {
    return inputTensors.at(index)->get_shape();
}

std::vector<int> VART_DPU::outputShape(size_t index) const {
    return outputTensors.at(index)->get_shape();
}

std::string VART_DPU::inputName(size_t index) const {
    return inputTensors.at(index)->get_name();
}

std::string VART_DPU::outputName(size_t index) const {
    return outputTensors.at(index)->get_name();
}

void VART_DPU::run(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) {
    if (inputs.size() != inputTensors.size() || outputs.size() != outputTensors.size()) {
        throw std::runtime_error(model + ": wrong number of input/output buffers");
    }

    // float -> int8 (batch item 0; other batch items, if the DPU has batch > 1, stay zero).
    // Rounding: half-up (floor(x + 0.5)), which is the DPU rounding the Vitis AI quantizer simulates
    // for fix neurons. TARGET-DEPENDENT: confirm against your Vitis AI version.
    for (size_t i = 0; i < inputs.size(); i++) {
        const size_t n = itemSize(inputShape(i));
        const float scale = inputScales[i];
        int8_t* dst = inputBuffers[i].data();
        const float* src = inputs[i];
        for (size_t k = 0; k < n; k++) {
            const float v = std::floor(src[k] * scale + 0.5f);
            dst[k] = static_cast<int8_t>(std::min(127.f, std::max(-128.f, v)));
        }
    }

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputBufs, outputBufs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    for (size_t i = 0; i < inputTensors.size(); i++) {
        inputBufs.push_back(std::make_unique<CpuFlatTensorBuffer>(inputBuffers[i].data(), inputTensors[i]));
        inputsPtr.push_back(inputBufs.back().get());
    }
    for (size_t i = 0; i < outputTensors.size(); i++) {
        outputBufs.push_back(std::make_unique<CpuFlatTensorBuffer>(outputBuffers[i].data(), outputTensors[i]));
        outputsPtr.push_back(outputBufs.back().get());
    }

    /*run*/
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, this->timeout);

    // int8 -> float (batch item 0)
    for (size_t i = 0; i < outputs.size(); i++) {
        const size_t n = itemSize(outputShape(i));
        const float scale = outputScales[i];
        const int8_t* src = outputBuffers[i].data();
        float* dst = outputs[i];
        for (size_t k = 0; k < n; k++) dst[k] = static_cast<float>(src[k]) * scale;
    }
}

VART_DPU::~VART_DPU() {
    std::cout << "VART DPU object is destroyed" << std::endl;
}

std::unique_ptr<ModelRunner> createModelRunner(const std::string& modelPath) {
    return std::make_unique<VART_DPU>(modelPath);
}
