cmake_minimum_required(VERSION 3.18)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)



include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_path(HIREDIS_INCLUDE_DIRS hiredis/hiredis.h HINTS "/usr/include")
find_library(HIREDIS_LIB_HIREDIS NAMES hiredis HINTS "/usr/lib")
set(HIREDIS_LIBRARIES ${HIREDIS_LIB_HIREDIS}) 
include_directories(/usr/include/nlohmann) 

add_subdirectory(engine)

add_executable(yolo main.cpp)
target_link_libraries(yolo engine)
target_link_libraries(yolo ${HIREDIS_LIBRARIES})
target_link_libraries(yolo nvinfer)
target_link_libraries(yolo nvinfer_plugin)
target_link_libraries(yolo cudart)

target_link_libraries(yolo ${OpenCV_LIBS})


# target_include_directories(engine PUBLIC
#                           "${PROJECT_BINARY_DIR}"
#                           "${PROJECT_SOURCE_DIR}/engine"
#                           )

add_definitions(-O2 -pthread)