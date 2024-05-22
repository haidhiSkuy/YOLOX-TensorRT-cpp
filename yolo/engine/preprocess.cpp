#include "yolo_object_detect.h"
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <vector>

#include "NvInfer.h"

cv::Mat letterbox(
    const cv::Mat& src, 
    int target_width, 
    int target_height, 
    cv::Scalar color = (0, 0, 0)) {
        
    int original_width = src.cols;
    int original_height = src.rows;
    float scale = std::min(static_cast<float>(target_width) / original_width, static_cast<float>(target_height) / original_height);

    int new_width = static_cast<int>(original_width * scale);
    int new_height = static_cast<int>(original_height * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_width, new_height));

    cv::Mat dst(cv::Size(target_width, target_height), src.type(), color);

    int x_offset = (target_width - new_width) / 2;
    int y_offset = (target_height - new_height) / 2;

    resized.copyTo(dst(cv::Rect(x_offset, y_offset, new_width, new_height)));

    return dst;
}


void Yolo::process_input(std::string image_path){ 
    // Read Image
    input_image = cv::imread(image_path);
    if (input_image.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int inputHeight = 640;
    const int inputWidth = 640;
    const int inputChannels = 3;
    const int inputSize = inputHeight * inputWidth * inputChannels;

    cv::Mat preprocessed_image = letterbox(input_image, 640, 640); 
    letterbox_image = preprocessed_image.clone();

    preprocessed_image.convertTo(preprocessed_image, CV_32F);
    cv::cvtColor(preprocessed_image, preprocessed_image, cv::COLOR_BGR2RGB);  

    std::vector<cv::Mat> inputChannelsVec(3); 
    cv::split(preprocessed_image, inputChannelsVec);


    numBindings = engine->getNbBindings();  

    auto inputDims = engine->getBindingDimensions(0); 
    for (int i = 0; i < inputDims.nbDims; ++i) {
        input_size *= inputDims.d[i];
    }  
        
    // Convert the image to NCHW format
    std::vector<float> chwImage(inputHeight * inputWidth * inputChannels);
    for (int c = 0; c < inputChannels; ++c) {
        for (int h = 0; h < inputHeight; ++h) {
            for (int w = 0; w < inputWidth; ++w) {
                chwImage[c * inputHeight * inputWidth + h * inputWidth + w] = preprocessed_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    inputData = chwImage; 
    
}   

void Yolo::create_buffers(){ 
    buffers.resize(numBindings);    
    bufferSizes.resize(numBindings);

    for(int binding_index = 0; binding_index < numBindings; binding_index++){         
        nvinfer1::Dims dims = engine->getBindingDimensions(binding_index);
        nvinfer1::DataType dtype = engine->getBindingDataType(binding_index);
        const char* bindingName = engine->getBindingName(binding_index); 
        std::cout << "binding index " << binding_index << " : " << bindingName << std::endl;

        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i]; 
        } 
        size *= (dtype == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(int32_t));
        bufferSizes[binding_index] = size;
        cudaMalloc(&buffers[binding_index], size);
    }

    // input buffer
    cudaMemcpy(buffers[0], inputData.data(), bufferSizes[0], cudaMemcpyHostToDevice); 

}