#include "yolo_object_detect.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime_api.h>
#include <vector> 
#include <iostream> 
#include <string>
#include <fstream>
#include <iterator>

#include <hiredis/adapters/libevent.h>
#include <nlohmann/json.hpp>



std::string class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};


std::vector<cv::Rect> convertToRects(const std::vector<float>& floatBoxes, bool isXYWH) {
    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < floatBoxes.size(); i += 4) {
        if (isXYWH) {
            // If the format is (x, y, width, height)
            float x = floatBoxes[i];
            float y = floatBoxes[i + 1];
            float width = floatBoxes[i + 2];
            float height = floatBoxes[i + 3];
            boxes.emplace_back(cv::Rect(cv::Point(x, y), cv::Size(width, height)));
        } else {
            // If the format is (x1, y1, x2, y2)
            float x1 = floatBoxes[i];
            float y1 = floatBoxes[i + 1];
            float x2 = floatBoxes[i + 2];
            float y2 = floatBoxes[i + 3];
            boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }
    return boxes;
}


void performNMS(
    std::vector<cv::Rect>& boxes, 
    std::vector<float>& scores, 
    std::vector<float>& classes, 
    float scoreThreshold, 
    float nmsThreshold, 
    std::vector<int>& indices) {
    
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
}


void draw_bbox(cv::Mat& input_image,std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<float>& classes, std::vector<int>& indices){

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int x1_lb = static_cast<int>(box.x - box.width / 2);
        int y1_lb = static_cast<int>(box.y - box.height / 2);
        int x2_lb = static_cast<int>(box.x + box.width / 2);
        int y2_lb = static_cast<int>(box.y + box.height / 2);


        //scale to original image coor and convert to xyxy
        float scale = std::min(
            static_cast<float>(640) / input_image.cols, 
            static_cast<float>(640) / input_image.rows
            );
        float pad_x = (640 - input_image.cols * scale) / 2;
        float pad_y = (640 - input_image.rows * scale) / 2;

        int x1_original = static_cast<int>((x1_lb - pad_x) / scale);
        int y1_original = static_cast<int>((y1_lb - pad_y) / scale);
        int x2_original = static_cast<int>((x2_lb - pad_x) / scale);
        int y2_original = static_cast<int>((y2_lb - pad_y) / scale);
    
        x1_original = std::max(0, std::min(input_image.cols, x1_original));
        y1_original = std::max(0, std::min(input_image.rows, y1_original));
        x2_original = std::max(0, std::min(input_image.cols, x2_original));
        y2_original = std::max(0, std::min(input_image.rows, y2_original));

        
        cv::Point p1(x1_original, y1_original); 
        cv::Point p2(x2_original, y2_original);

        float score = scores[idx];
        int classId = classes[idx];
        std::string class_name = class_names[classId];

        cv::rectangle(input_image, p1, p2, cv::Scalar(0, 255, 0), 2);

        int baseLine;
        std::string label = class_name + ": " + cv::format("%.2f", score);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); 

    
        cv::rectangle(
            input_image, 
            cv::Rect(cv::Point(x1_original, y1_original - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)),
            cv::Scalar(0, 255, 0), cv::FILLED
            );

        cv::putText(
            input_image, label, 
            cv::Point(x1_original, y1_original), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1
        );
    }
}


std::string toJson(std::vector<cv::Rect>& boxes, std::vector<float>& classes, std::vector<int>& indices){
    nlohmann::json output; 

    int count = 0;
    for (int idx : indices) {
        cv::Rect rect = boxes[idx];
        int classId = classes[idx];
        std::string class_name = class_names[classId];
    
        nlohmann::json j;
        j["class"] = class_name;
        j["x"] = rect.x;
        j["y"] = rect.y;
        j["width"] = rect.width;
        j["height"] = rect.height;

        output[std::to_string(count)] = j;
        count += 1;
    }

    std::string jsonString = output.dump();
    return jsonString;
}


void Yolo::post_process(){ 
    std::vector<float> floatBoxes = outputData[0]; 
    std::vector<float> scores = outputData[1];
    std::vector<float> classes = outputData[2];

    // Perform NMS
    float scoreThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;   // NMS IoU threshold

    std::vector<int> indices;
    std::vector<cv::Rect> boxes = convertToRects(floatBoxes, true);
    performNMS(boxes, scores, classes, scoreThreshold, nmsThreshold, indices);

    // Draw Bboxes
    draw_bbox(input_image, boxes, scores, classes, indices);

    // send to redis
    const std::string channel = "yolox"; 
    std::string output_json = toJson(boxes, classes, indices);  
    // redisReply* reply = (redisReply*)redisCommand(c, "PUBLISH %s %s", channel.c_str(), output_json.c_str()); 

    cv::imwrite("/workspaces/tensorrt/result_image/result.jpg", input_image);

}