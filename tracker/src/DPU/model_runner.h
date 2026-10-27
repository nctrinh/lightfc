#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include <memory>
#include <string>
#include <vector>

/*
 * Accelerator-agnostic model runner used by the tracker. Tensors are exchanged as float32 NHWC
 * (batch item 0); quantization/dequantization, buffer layout and the runtime API stay inside the
 * implementation. The only implementation today is the Vitis AI VART one (dpu.cpp); another target
 * (other NPU runtime, ONNX Runtime on CPU, ...) only needs a new implementation of this interface
 * and its own createModelRunner().
 */
class ModelRunner
{
public:
    virtual ~ModelRunner() = default;

    virtual size_t inputCount() const = 0;
    virtual size_t outputCount() const = 0;
    // Full tensor shape as reported by the runtime (NHWC, batch first).
    virtual std::vector<int> inputShape(size_t index) const = 0;
    virtual std::vector<int> outputShape(size_t index) const = 0;
    virtual std::string inputName(size_t index) const = 0;
    virtual std::string outputName(size_t index) const = 0;

    // Index of the first input/output whose name contains `part`, or -1.
    int findInput(const std::string& part) const;
    int findOutput(const std::string& part) const;

    // inputs[i] / outputs[i]: float NHWC buffers of one batch item (product of shape[1:]).
    virtual void run(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) = 0;
};

// Number of elements of one batch item (product of shape[1:]).
size_t itemSize(const std::vector<int>& shape);
std::string shapeToString(const std::vector<int>& shape);

// Implemented by the selected backend (dpu.cpp for Vitis AI). Throws std::runtime_error on failure.
std::unique_ptr<ModelRunner> createModelRunner(const std::string& modelPath);

#endif // MODEL_RUNNER_H
