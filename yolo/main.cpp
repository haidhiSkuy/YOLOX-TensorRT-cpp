#include <vector>
#include <iostream> 
#include <string>
#include <opencv2/opencv.hpp>
#include "engine/yolo_object_detect.h"



int main(int argc, char* argv[]) {
  Yolo yolo("/workspaces/tensorrt/models/yolox/yolox2.trt");

  yolo.process_input("/workspaces/tensorrt/test_images/dog_bike_car.jpg");
  yolo.create_buffers();
  yolo.infer(); 
  yolo.post_process();


}