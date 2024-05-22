cmake_minimum_required(VERSION 3.18)

project(redis)

add_definitions(-std=c++11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)



include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")



find_path(HIREDIS_INCLUDE_DIRS hiredis/hiredis.h HINTS "/usr/include")
find_library(HIREDIS_LIB_HIREDIS NAMES hiredis HINTS "/usr/lib")

set(HIREDIS_LIBRARIES ${HIREDIS_LIB_HIREDIS})

add_executable(redis main.cpp)
target_link_libraries(redis ${HIREDIS_LIBRARIES})


add_definitions(-O2 -pthread)