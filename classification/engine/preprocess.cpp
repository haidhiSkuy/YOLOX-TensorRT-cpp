#include "classifier.h"
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>

#include "NvInfer.h"

void Classifier::process_input(char* image_path){ 

    // Get Output Dim
    int numBindings = engine->getNbBindings(); 
    int inputBindingIndex = engine->getBindingIndex("x");  
    auto inputDims = engine->getBindingDimensions(inputBindingIndex); 

    const int inputB = inputDims.d[0];
    const int inputH = inputDims.d[1];
    const int inputW = inputDims.d[2];
    const int inputC = inputDims.d[3]; 

    // Get Output Dim
    int outputBindingIndex = engine->getBindingIndex("dense");  
    auto outputDims = engine->getBindingDimensions(outputBindingIndex);

    const int outputB = outputDims.d[0];
    const int outputD = outputDims.d[1];
    

    // Read Image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        std::exit(EXIT_FAILURE);
    }


    
    cv::resize(image, image, cv::Size(inputW, inputH));
    image.convertTo(image, CV_32FC3);
    image /= 255.0; 

    std::vector<cv::Mat> inputChannelsVec(inputC); 
    cv::split(image, inputChannelsVec); 

    input_size = inputB * inputH * inputW * inputC;
    inputData.resize(input_size); 

    output_size = outputB * outputD;
    outputData.resize(output_size);


    int channelSize = inputH * inputW;
    for (int b = 0; b < inputB; ++b) {
        for (int c = 0; c < inputC; ++c) {
            // Get pointer to input data for zthis batch, channel
            float* inputDataPtr = inputData.data() + (b * inputC + c) * channelSize;

            // Copy image data to input buffer
            cv::Mat channel = inputChannelsVec[c];

            channel.convertTo(channel, CV_32FC1);
            memcpy(inputDataPtr, channel.ptr<float>(0), channelSize * sizeof(float));
        }
    }

}
