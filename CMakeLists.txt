cmake_minimum_required(VERSION 3.24)
project(CppDeepLearningLib)

set(CMAKE_CXX_STANDARD 17)

include(CTest)

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(sample_code)

# add_executable(CppDeepLearningLib "./sample_code/main.cpp")