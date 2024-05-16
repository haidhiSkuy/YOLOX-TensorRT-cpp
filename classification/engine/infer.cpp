#include "classifier.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h" 

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include <vector> 
#include <iostream> 

void Classifier::infer(){ 

    void* buffers[2];
    cudaMalloc(&buffers[0], input_size * sizeof(float));  // input buffer
    cudaMalloc(&buffers[1], output_size * sizeof(float)); // output buffer

    cudaMemcpy(buffers[0], inputData.data(), input_size * sizeof(float), cudaMemcpyHostToDevice); 

    context->executeV2(buffers);

    outputData.resize(output_size);
    cudaMemcpy(outputData.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print Output
    for (int b = 0; b < 1; ++b) {
        std::cout << "Batch " << b << std::endl;
        for (int c = 0; c < 10; ++c) {
            auto prob =  outputData[b * output_size + c];
            std::cout << "class "<< c << ": " << prob << std::endl;
        }
    }

}