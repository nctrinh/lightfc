#ifndef DPU_H
#define DPU_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <common.h>


class VART_DPU
{   
    private:
        std::string model;
        std::unique_ptr<xir::Graph> graph;
        GraphInfo shapes;

        float** outputData;
        TensorShape* inShapes;
        TensorShape* outShapes;
        size_t inputCnt;
        size_t outputCnt;
        std::vector<std::unique_ptr<xir::Tensor>> inputTensors;
        std::vector<std::unique_ptr<xir::Tensor>> outputTensors;
        std::unique_ptr<vart::Runner> runner;

        int timeout;
        
    public:
        VART_DPU();
        VART_DPU(std::string model_path, int timeout = -1);
        ~VART_DPU();
        void Erase_dynamic_memory(void);

        void Init(std::string model_path, int timeout = -1);
        void Run(float** inputData);

        /**
         * @brief Function get list of input Tensor shapes
         * @return
         */
        TensorShape* getInShapes();

        /**
         * @brief Function get list of output Tensor shapes
         * @return
         */
        TensorShape* getOutShapes();

        /**
         * @brief Function get number inputs of graph
         * @return
         */
        size_t getInputCnt();

        /**
         * @brief Function get number outputs of graph
         * @return
         */
        size_t getOutputCnt();

        /**
         * @brief Function get output data
         * @return
         */
        float** getOutputData();

};

#endif /* DPU_H */
