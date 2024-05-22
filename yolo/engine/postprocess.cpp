#include "yolo_object_detect.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h" 

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime_api.h>

#include <vector> 
#include <iostream> 
#include <string>

#include <fstream>
#include <iterator> 



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

 
cv::Mat draw_bbox(cv::Mat& image,std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<float>& classes, std::vector<int>& indices){

    // Display results
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float score = scores[idx];
        int classId = classes[idx];

        std::string class_name = class_names[classId];

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        std::string label = class_name + ": " + cv::format("%.2f", score);

        std::cout << "Kept box: " << box << " with score: " << score << " and class: " << classId << std::endl;

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); 

        // Draw the label background
        cv::rectangle(image, cv::Rect(cv::Point(box.x, box.y - labelSize.height),
                                      cv::Size(labelSize.width, labelSize.height + baseLine)),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        // Put the label text
        cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return image; 
}



void Yolo::post_process(){ 
    std::vector<float> floatBoxes = outputData[0]; 
    std::vector<float> scores = outputData[1];
    std::vector<float> classes = outputData[2];

    // Parameters for NMS
    float scoreThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;   // NMS IoU threshold

    // Indices of the boxes to keep
    std::vector<int> indices;

    // Perform NMS
    std::vector<cv::Rect> boxes = convertToRects(floatBoxes, true);
    performNMS(boxes, scores, classes, scoreThreshold, nmsThreshold, indices);



    cv::Mat drawed_image = draw_bbox(input_image, boxes, scores, classes, indices);

    cv::imwrite("/workspaces/tensorrt/result_image/dog_bike.jpg", drawed_image);

}