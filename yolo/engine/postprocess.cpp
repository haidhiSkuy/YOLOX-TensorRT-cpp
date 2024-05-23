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
    std::vector<int>& indices,
    std::vector<cv::Rect>& filtered_boxes, 
    std::vector<float>& filtered_scores, 
    std::vector<int>& filtered_classes
    ) {
    
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);
    
    for (int idx : indices) {
        filtered_boxes.push_back(boxes[idx]); 
        filtered_scores.push_back(scores[idx]);
        filtered_classes.push_back(classes[idx]);
    }

}

std::vector<std::vector<int>> scale_bbox(cv::Mat input_image, std::vector<cv::Rect> boxes){ 

    std::vector<std::vector<int>> scaled_coordinate; 
    for (int idx = 0; idx < boxes.size(); idx++) {
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

        std::vector<int> scaled = {x1_original, y1_original, x2_original, y2_original}; 
        scaled_coordinate.push_back(scaled);
    }

    return scaled_coordinate;
}


void draw_bbox(cv::Mat& input_image, std::vector<std::vector<int>> bbox, std::vector<float> scores, std::vector<int> classes){

    for(int i=0; i<classes.size(); i++) {
        std::vector<int> box = bbox[i];

        cv::Point p1(box[0], box[1]); 
        cv::Point p2(box[2], box[3]);

        float score = scores[i];
        int classId = classes[i];
        std::string class_name = class_names[classId];

        cv::rectangle(input_image, p1, p2, cv::Scalar(0, 255, 0), 2);

        int baseLine;
        std::string label = class_name + ": " + cv::format("%.2f", score);
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); 

        cv::rectangle(
            input_image, 
            cv::Rect(cv::Point(box[0], box[1]-labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)),
            cv::Scalar(0, 255, 0), cv::FILLED
            );

        cv::putText(
            input_image, label, 
            cv::Point(box[0], box[1]), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1
        );
    }
}


void redis_pubsub(
    redisContext *redis, 
    std::vector<std::vector<int>> bbox, 
    std::vector<float> scores, 
    std::vector<int> classes,
    std::string channel
){
    nlohmann::json output; 

    for(int i=0; i < classes.size(); i++) {
        std::vector<int> box = bbox[i];
        float score = scores[i];
        int classId = classes[i];
        std::string class_name = class_names[classId];
    
        nlohmann::json j;
        j["class"] = class_name;
        j["score"] = score;
        j["x1"] = box[0];
        j["y1"] = box[1];
        j["x2"] = box[2];
        j["y2"] = box[3];
        output["object_"+std::to_string(i)] = j;
    }
    std::string jsonString = output.dump();
    redisReply* reply = (redisReply*)redisCommand(redis, "PUBLISH %s %s", channel.c_str(), jsonString.c_str()); 
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

    std::vector<cv::Rect> filtered_boxes;
    std::vector<float> filtered_scores;
    std::vector<int> filtered_classes;
    performNMS(
        boxes, scores, classes, scoreThreshold, nmsThreshold, 
        indices, filtered_boxes, filtered_scores, filtered_classes
    );

    //scale bbox 
    std::vector<std::vector<int>> scaled_bbox = scale_bbox(input_image, filtered_boxes); 

    // Draw Bboxes
    draw_bbox(input_image, scaled_bbox, filtered_scores, filtered_classes);

    // send to redis
    const std::string channel = "yolox"; 
    redis_pubsub(redis, scaled_bbox, filtered_scores, filtered_classes, channel);

    cv::imwrite("/workspaces/tensorrt/result_image/result.jpg", input_image);

}