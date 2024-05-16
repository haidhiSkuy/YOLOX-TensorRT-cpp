#include "NvInfer.h"
#include <vector>


class Classifier{ 
    public: 
        Classifier(char* trt_path); 
        void process_input(char* image_path); 
        void infer();
    
    private: 
      // inference engine
      nvinfer1::ICudaEngine* engine = nullptr;
      nvinfer1::IRuntime* runtime = nullptr;
      nvinfer1::IExecutionContext* context = nullptr;
      
      // input and output size
      int input_size, output_size;

      //preprocessed image 
      std::vector<float> inputData;
      std::vector<float> outputData;


};


