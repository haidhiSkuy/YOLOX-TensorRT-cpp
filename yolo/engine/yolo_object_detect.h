#include "NvInfer.h"
#include <vector>
#include <opencv2/opencv.hpp>


class Yolo{ 
    public: 
        Yolo(char* trt_path); 
        void process_input(char* image_path);
        void infer();
    
    private: 
        // inference engine
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
      
        // input and output size
        int input_size, output_size;
    
        std::vector<float> inputData;
        std::vector<float> outputData;
};


