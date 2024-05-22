#include "NvInfer.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <hiredis/adapters/libevent.h>
#include <string>

class Yolo{ 
    public: 
        Yolo(std::string trt_path, char* redis_hostname, int redis_port); 
        void process_input(std::string image_path);
        void infer();
        void create_buffers();
        void post_process(); 
    
    private: 
        // inference engine
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        //redis 
        redisContext *c; 

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


