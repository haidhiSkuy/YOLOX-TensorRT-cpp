#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include <math.h>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using Severity = nvinfer1::ILogger::Severity;

using std::array;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;

class Logger : public ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};

class Yolo {
  public:
    Yolo(char* model_path, char* image_path);

  private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    Logger gLogger;
};


Yolo::Yolo(char* model_path, char* image_path) {

  // LOAD TRT ENGINE
  ifstream ifile(model_path, ios::in | ios::binary);
  if (!ifile) {
    std::cout << "read serialized file failed\n";
    std::abort();
  }

  ifile.seekg(0, ios::end);
  const int mdsize = ifile.tellg();
  ifile.clear();
  ifile.seekg(0, ios::beg);
  std::vector<char> buf(mdsize);
  ifile.read(&buf[0], mdsize);
  ifile.close();
  std::cout << "model size: " << mdsize << std::endl;

  runtime = nvinfer1::createInferRuntime(gLogger);
  initLibNvInferPlugins(&gLogger, "");
  engine = runtime -> deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
  std::cout << "Load Success" << std::endl;


  // GET INPUT DIMENSIONS
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


  // Input Image 
  std::cout << image_path << std::endl;
  cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return;
    }

  cv::resize(image, image, cv::Size(inputW, inputH));
  image.convertTo(image, CV_32FC3);
  image /= 255.0;

  std::vector<cv::Mat> inputChannelsVec(inputC); 
  cv::split(image, inputChannelsVec); 


  const int inputSize = inputB * inputH * inputW * inputC;
  std::vector<float> inputData(inputSize);

  const int outputSize = outputB * outputD;
  std::vector<float> outputData;

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

  // INFERENCE
  IExecutionContext* context = engine->createExecutionContext();

  // Allocate CUDA memory for input and output buffers
  void* buffers[2];
  cudaMalloc(&buffers[0], inputSize * sizeof(float));  // input buffer
  cudaMalloc(&buffers[1], outputSize * sizeof(float)); // output buffer

  cudaMemcpy(buffers[0], inputData.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

  context->executeV2(buffers);

  outputData.resize(outputSize);
  cudaMemcpy(outputData.data(), buffers[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost);


  // Print output probabilities
  std::vector<string> labels = {"Butterfly", "Cat", "Cow", "Dog", "Elephant","Hen", "Horse", "Sheep", "Spider", "Squirrel"};

  for (int b = 0; b < 1; ++b) {
    std::cout << "Batch " << b << std::endl;
    for (int c = 0; c < 10; ++c) {
        auto prob =  outputData[b * outputSize + c];
        std::cout << labels[c] << ": " << prob << std::endl;
    }

}

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
  context->destroy();
  engine->destroy();
  runtime->destroy();

}



int main(int argc, char* argv[]) {
  Yolo yolo("/workspaces/tensorrt/models/animals.trt", argv[1]);
}
