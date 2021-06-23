cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(BoatDetector VERSION 1.0)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(BBoxSelector BBoxSelector.cpp Utils.cpp)
target_link_libraries(BBoxSelector ${OpenCV_LIBS})
target_link_libraries(BBoxSelector stdc++)

add_executable(bw_he BlackWhite_He.cpp)
target_link_libraries(bw_he ${OpenCV_LIBS})
target_link_libraries(bw_he stdc++)
