#include "NvInfer.h"
#include <vector>
#include <opencv2/opencv.hpp>


class Yolo{ 
    public: 
        Yolo(char* trt_path); 
        void process_input(char* image_path);
        void infer();
        void create_buffers();
        void post_process(); 
    
    private: 
        // inference engine
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        //ouput bindings 
        int numBindings; 
        std::vector<void*> buffers;    
        std::vector<int64_t> bufferSizes;
      
        // input and output size
        int input_size = 1;
        int output_size = 1;

        std::vector<float> inputData;
        std::vector<std::vector<float>> outputData; 

        // letterbox image 
        cv::Mat input_image;
        cv::Mat letterbox_image;
};


