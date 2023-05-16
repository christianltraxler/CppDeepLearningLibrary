cmake_minimum_required(VERSION 3.24)
project(CppDeepLearningLib)


file(GLOB HEADER_LIST "${CppDeepLearningLib}/include/NeuralNetwork/NeuralNetwork.h")
file(GLOB SRC_Files "*.cpp")

add_library(DeepLearningLibrary
        ${HEADER_LIST}
        ${SRC_Files}
        )

target_include_directories(DeepLearningLibrary PUBLIC ../include)
