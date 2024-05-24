#include "engine/yolo_object_detect.h"
#include <string>
#include <iostream>
#include <chrono>


void image_infer(Yolo yolo_class, char* image_path, char* out_path){ 
  cv::Mat in = cv::imread(image_path);
  yolo_class.process_input(in); 
  yolo_class.infer(); 
  cv::Mat out = yolo_class.post_process();
  cv::imwrite(out_path, out);
}

void video_infer(Yolo yolo_class, char* video_path, char* out_path){ 
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video file." << std::endl;
    std::abort();
  } 

  int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double vid_fps = cap.get(cv::CAP_PROP_FPS);

  int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
  cv::VideoWriter writer(out_path, fourcc, vid_fps, cv::Size(frameWidth, frameHeight));

  cv::Mat frame;
  int frameCount = 0;
  int fps = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto lastTime = start;
  
  while (true) {
      cap >> frame;
      if (frame.empty()) {break;} 

      yolo_class.process_input(frame);
      yolo_class.infer(); 
      cv::Mat result = yolo_class.post_process();

      frameCount++;
        
      // Calculate and display FPS every second
      auto currentTime = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = currentTime - lastTime;
      if (elapsed.count() >= 1.0) {
        fps = frameCount;
        frameCount = 0;
        lastTime = currentTime;

        std::string fpsText = "FPS: " + std::to_string(fps); 
        std::cout << fpsText << std::endl;
      }      
      writer.write(result);
  }
    cap.release();
    cv::destroyAllWindows();
}


int main(int argc, char* argv[]) {

  std::string engine_path = "/workspaces/tensorrt/models/yolox/yolox2.trt"; 
  Yolo yolo(engine_path, "localhost", 6379);

  std::string arg = argv[1];
  char* input_path = argv[2]; 
  char* output_path = argv[3];

  if(arg == "--image"){ 
    image_infer(yolo, input_path, output_path);
  } else if(arg == "--video"){ 
    video_infer(yolo, input_path, output_path);
  }

  return 0;

}