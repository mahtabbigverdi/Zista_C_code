cmake_minimum_required(VERSION 3.21)
project(opencvcourse)

set(CMAKE_CXX_STANDARD 14)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)

add_executable(opencvcourse main.cpp)


target_link_libraries(opencvcourse ${OpenCV_LIBS})