#include "yolo_object_detect.h"
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>

#include "NvInfer.h"

cv::Mat letterbox(
    const cv::Mat& src, 
    int target_width, 
    int target_height, 
    cv::Scalar color = (128, 128, 128)) {
        
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


void Yolo::process_input(char* image_path){ 

    // Get Input Dim
    int numBindings = engine->getNbBindings(); 
    int inputBindingIndex = engine->getBindingIndex("images");  
    auto inputDims = engine->getBindingDimensions(inputBindingIndex); 

    const int input_batch = inputDims.d[0];
    const int input_channels = inputDims.d[1]; 
    const int input_height = inputDims.d[2];
    const int input_width = inputDims.d[3];

    // Get Output Dim
    int outputBindingIndex = engine->getBindingIndex("output0");  
    auto outputDims = engine->getBindingDimensions(outputBindingIndex);

    const int output_batch = outputDims.d[0];
    const int output_feature = outputDims.d[1];
    const int output_prediction = outputDims.d[2]; 

     // Read Image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    image = letterbox(image, 640, 640); // letterbox

    image.convertTo(image, CV_32F, 1.0 / 255.0); // normalize
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // convert to rgb


    std::vector<cv::Mat> inputChannelsVec(input_channels); 
    cv::split(image, inputChannelsVec);

    input_size = input_batch * input_height * input_width * input_channels;
    inputData.resize(input_size); 

    output_size = output_batch * output_feature * output_prediction;
    outputData.resize(output_size);



    int channelSize = input_height * input_width;
    for (int b = 0; b < input_batch; ++b) {
        for (int c = 0; c < input_channels; ++c) {
            // Get pointer to input data for zthis batch, channel
            float* inputDataPtr = inputData.data() + (b * input_channels + c) * channelSize;

            // Copy image data to input buffer
            cv::Mat channel = inputChannelsVec[c];

            channel.convertTo(channel, CV_32FC1);
            memcpy(inputDataPtr, channel.ptr<float>(0), channelSize * sizeof(float));
        }
    }

    
}