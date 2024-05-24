#include "yolo_object_detect.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include <hiredis/adapters/libevent.h>

#include <math.h>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
}; 


Yolo::Yolo(std::string trt_path, char* redis_hostname, int redis_port){ 
    Logger gLogger;

    std::ifstream ifile(trt_path, std::ios::in | std::ios::binary);
    if (!ifile) {
      std::cout << "read serialized file failed\n";
      std::abort();
    }

    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    std::cout << "model size: " << mdsize << std::endl;

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime -> deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
    std::cout << "Load Success" << std::endl;    

    context = engine->createExecutionContext(); 

    // CREATE BUFFER
    numBindings = engine->getNbBindings();
    auto inputDims = engine->getBindingDimensions(0); 
    for (int i = 0; i < inputDims.nbDims; ++i) {
        input_size *= inputDims.d[i];
    }  

    buffers.resize(numBindings);    
    bufferSizes.resize(numBindings);
    for(int binding_index = 0; binding_index < numBindings; binding_index++){         
        nvinfer1::Dims dims = engine->getBindingDimensions(binding_index);
        nvinfer1::DataType dtype = engine->getBindingDataType(binding_index);
        const char* bindingName = engine->getBindingName(binding_index); 
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i]; 
        } 
        size *= (dtype == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(int32_t));
        bufferSizes[binding_index] = size;
        cudaMalloc(&buffers[binding_index], size);
    }

    //redis
    redis = redisConnect(redis_hostname, redis_port);
    if (redis->err) {
        printf("error: %s\n", redis->errstr);
        std::abort();
    } else { 
      std::cout << "Redis Connection Success" << std::endl;
    }
}