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

add_executable(sea_selector sea_selector.cpp)
target_link_libraries(sea_selector ${OpenCV_LIBS})
target_link_libraries(sea_selector stdc++)

add_executable(k_m_t k_m_t.cpp)
target_link_libraries(k_m_t ${OpenCV_LIBS})
target_link_libraries(k_m_t stdc++)
