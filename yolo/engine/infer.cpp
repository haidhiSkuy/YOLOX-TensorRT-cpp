#include "yolo_object_detect.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h" 

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include <vector> 
#include <iostream> 

#include <fstream>
#include <iterator>


void Yolo::infer(){ 

    void* buffers[2];
    cudaMalloc(&buffers[0], input_size * sizeof(float));  // input buffer
    cudaMalloc(&buffers[1], output_size * sizeof(float)); // output buffer

    cudaMemcpy(buffers[0], inputData.data(), input_size * sizeof(float), cudaMemcpyHostToDevice); 

    context->executeV2(buffers);

    outputData.resize(output_size);
    cudaMemcpy(outputData.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output size: " << outputData.size() << std::endl;
    

}