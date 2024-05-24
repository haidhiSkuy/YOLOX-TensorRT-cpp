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
    cv::Scalar color = (50, 50, 50)) {
        
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


void Yolo::process_input(cv::Mat frame){ 
    input_image = frame.clone();

    const int inputHeight = 640;
    const int inputWidth = 640;
    const int inputChannels = 3;
    const int inputSize = inputHeight * inputWidth * inputChannels;

    cv::Mat preprocessed_frame = letterbox(frame, 640, 640); 

    preprocessed_frame.convertTo(preprocessed_frame, CV_32F);
    cv::cvtColor(preprocessed_frame, preprocessed_frame, cv::COLOR_BGR2RGB);  

    std::vector<cv::Mat> inputChannelsVec(3); 
    cv::split(preprocessed_frame, inputChannelsVec);

      
    // Convert the image to NCHW format
    std::vector<float> chwImage(inputHeight * inputWidth * inputChannels);
    for (int c = 0; c < inputChannels; ++c) {
        for (int h = 0; h < inputHeight; ++h) {
            for (int w = 0; w < inputWidth; ++w) {
                chwImage[c * inputHeight * inputWidth + h * inputWidth + w] = preprocessed_frame.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    cudaMemcpy(buffers[0], chwImage.data(), bufferSizes[0], cudaMemcpyHostToDevice);
}   
