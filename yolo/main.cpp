#include <vector>
#include <iostream> 
#include <string>
#include <opencv2/opencv.hpp>
#include "engine/yolo_object_detect.h"



int main(int argc, char* argv[]) {
  Yolo yolo("/workspaces/tensorrt/models/yolov8n.trt");
  yolo.process_input("/workspaces/tensorrt/test_images/bus.jpg");
  yolo.infer();


}