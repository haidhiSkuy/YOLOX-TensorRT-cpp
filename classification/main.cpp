#include <iostream> 
#include "engine/classifier.h"
#include <vector>



int main(int argc, char* argv[]) {
  // Yolo yolo("/workspaces/tensorrt/models/animals.trt", argv[1]);
  char* model_engine_path = argv[1]; 
  char* image_path = argv[2];

  Classifier classifier(model_engine_path);

  classifier.process_input(image_path);
  
  classifier.infer();

}