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

using cv::Mat;
using std::array;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;
using std::vector;

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
    Yolo(char* model_path);

  private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    Logger gLogger;
};


Yolo::Yolo(char* model_path) {
  ifstream ifile(model_path, ios::in | ios::binary);
  if (!ifile) {
    cout << "read serialized file failed\n";
    std::abort();
  }

  ifile.seekg(0, ios::end);
  const int mdsize = ifile.tellg();
  ifile.clear();
  ifile.seekg(0, ios::beg);
  vector<char> buf(mdsize);
  ifile.read(&buf[0], mdsize);
  ifile.close();
  cout << "model size: " << mdsize << endl;

  runtime = nvinfer1::createInferRuntime(gLogger);
  initLibNvInferPlugins(&gLogger, "");
  engine = runtime -> deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
  cout << "Load Success" << endl;
}



int main() {
  Yolo yolo("/workspaces/tensorrt/testing/onnx2rt/model.engine");
}
