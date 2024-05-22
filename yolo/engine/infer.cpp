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

    bool status = context->executeV2(buffers.data()); 

    if (!status) {
        std::cerr << "Inference execution failed!" << std::endl;
        std::exit(EXIT_FAILURE);
    } else { 
        std::cout << "Inference Success" << std::endl;
    }

  
    // boxes 
    std::vector<float> floatBoxes(bufferSizes[1] / sizeof(float));
    cudaMemcpy(floatBoxes.data(), buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);
    // std::vector<cv::Rect> boxes = convertToRects(floatBoxes, true); 

    // scores
    std::vector<float> scores(bufferSizes[2] / sizeof(float));
    cudaMemcpy(scores.data(), buffers[2], bufferSizes[2], cudaMemcpyDeviceToHost);

    // classes
    std::vector<float> classes(bufferSizes[3] / sizeof(float));
    cudaMemcpy(classes.data(), buffers[3], bufferSizes[3], cudaMemcpyDeviceToHost);

    outputData = {floatBoxes, scores, classes};

}