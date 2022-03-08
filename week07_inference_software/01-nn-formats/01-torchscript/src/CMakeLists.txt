cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(vgg_cpp main.cpp)
target_link_libraries(vgg_cpp "${TORCH_LIBRARIES}")
set_property(TARGET vgg_cpp PROPERTY CXX_STANDARD 14)
