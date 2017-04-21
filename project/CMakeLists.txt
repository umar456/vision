
cmake_minimum_required (VERSION 3.6)
project(FaceTracker)

add_definitions(-Wall -Werror -std=c++11)

if(POLICY CMP0037)
  cmake_policy(SET CMP0037 NEW)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES main.cpp)

add_executable(tracker ${SOURCE_FILES})
target_link_libraries(tracker ${OpenCV_LIBRARIES})

set_property(TARGET tracker PROPERTY CXX_STANDARD 17)
