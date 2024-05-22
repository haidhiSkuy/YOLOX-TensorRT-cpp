#include "engine/yolo_object_detect.h"
#include <string>

int main(int argc, char* argv[]) {
  
  std::string engine_path = "/workspaces/tensorrt/models/yolox/yolox2.trt"; 
  char* hostname = "localhost"; 
  int port = 6379; 

  Yolo yolo(engine_path, hostname, port);

  yolo.process_input("/workspaces/tensorrt/test_images/bus.jpg");
  yolo.create_buffers();
  yolo.infer(); 
  yolo.post_process();


}