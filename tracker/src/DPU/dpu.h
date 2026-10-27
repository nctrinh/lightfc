#ifndef DPU_H
#define DPU_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <common.h>

#include "model_runner.h"

/*
 * Vitis AI (VART) implementation of ModelRunner for one compiled .xmodel.
 *
 * The DPU subgraph's tensors are XINT8 with a power-of-two scale ("fix_point" attribute):
 *   int8 = clamp(round(float * 2^fix_point), -128, 127),   float = int8 * 2^-fix_point.
 * The previous version handed float buffers to the DPU runner, which only works if the runtime
 * converts on its own; here the conversion is explicit.
 *
 * Target-dependent (verify on the board): the xmodel must contain exactly one DPU subgraph
 * (no CPU ops in between), and the rounding mode used for the input conversion.
 */
class VART_DPU : public ModelRunner
{
    private:
        std::string model;
        std::unique_ptr<xir::Graph> graph;
        std::unique_ptr<vart::Runner> runner;
        std::vector<const xir::Tensor*> inputTensors;
        std::vector<const xir::Tensor*> outputTensors;
        std::vector<std::vector<int8_t>> inputBuffers;   // whole batch
        std::vector<std::vector<int8_t>> outputBuffers;
        std::vector<float> inputScales;                  // 2^fix_point
        std::vector<float> outputScales;                 // 2^-fix_point
        int timeout;

    public:
        explicit VART_DPU(const std::string& model_path, int timeout = -1);
        ~VART_DPU() override;

        size_t inputCount() const override { return inputTensors.size(); }
        size_t outputCount() const override { return outputTensors.size(); }
        std::vector<int> inputShape(size_t index) const override;
        std::vector<int> outputShape(size_t index) const override;
        std::string inputName(size_t index) const override;
        std::string outputName(size_t index) const override;

        void run(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) override;
};

#endif /* DPU_H */
