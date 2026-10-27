#include "dpu.h"


VART_DPU::VART_DPU() {
    std::cout << "VART DPU Object is created " << std::endl;
}

VART_DPU::VART_DPU(std::string model_path, int timeout) {
    Init(model_path, timeout);
}

void VART_DPU::Init(std::string model_path, int timeout) {
    this->timeout = timeout;
    graph = xir::Graph::deserialize(model_path);
    auto subgraph = get_dpu_subgraph(graph.get());

    /*create runner*/
    runner = vart::Runner::create_runner(subgraph[0], "run");
    /*get in/out tensor*/
    inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    outputTensors = cloneTensorBuffer(runner->get_output_tensors());

    /*get in/out tensor shape*/
    inputCnt = inputTensors.size();
    outputCnt = outputTensors.size();

    inShapes = new TensorShape[inputCnt];
    outShapes = new TensorShape[outputCnt];

    shapes.inTensorList = inShapes;
    shapes.outTensorList = outShapes;

    getTensorShape(runner.get(), &shapes, int(inputCnt), int(outputCnt));

    /*allocate outputData*/
    outputData = new float*[outputCnt];
    for (size_t index = 0; index < outputCnt; index++) {
        outputData[index] = new float[shapes.outTensorList[index].size];
    }

    std::cout << "VART DPU Object is initialized" << std::endl;
}

TensorShape* VART_DPU::getInShapes() {
    return this->inShapes;
}

TensorShape* VART_DPU::getOutShapes() {
    return this->outShapes;
}

size_t VART_DPU::getInputCnt() {
    return this->inputCnt;
}

size_t VART_DPU::getOutputCnt() {
    return this->outputCnt;
}

float** VART_DPU::getOutputData() {
    return this->outputData;
}

void VART_DPU::Run(float** inputDatas) {
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

    for (size_t index=0; index < this->inputCnt; index++) {
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>
                         (inputDatas[index], inputTensors[index].get()));
    }

    for (size_t index=0; index < this->outputCnt; index++) {
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>
                          (outputData[index], outputTensors[shapes.output_mapping[index]].get()));
    }

    /* tensor buffer input/output */
    for (size_t index=0; index < this->inputCnt; index++) {
        inputsPtr.push_back(inputs[index].get());
    }

    for (size_t index=0; index < this->outputCnt; index++) {
        outputsPtr.push_back(outputs[index].get());
    }
    
    /*run*/
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, this->timeout);
}


void VART_DPU::Erase_dynamic_memory(void){
    for (size_t index=0; index < this->outputCnt; index++) {
        delete[] outputData[index];
    }
    delete[] outputData;
    delete[] inShapes;
    delete[] outShapes;
}


VART_DPU::~VART_DPU() {
    for (size_t index=0; index < this->outputCnt; index++) {
        delete[] outputData[index];
    }
    delete[] outputData;
    delete[] inShapes;
    delete[] outShapes;
    std::cout << "VART DPU object is destroyed" << std::endl;
}
