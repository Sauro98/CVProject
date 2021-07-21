cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(BoatDetector VERSION 1.0)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(WaterSelector WaterSelector.cpp Utils.cpp DatasetHelper.cpp SegmentationHelper.cpp)
target_link_libraries(WaterSelector ${OpenCV_LIBS})
target_link_libraries(WaterSelector stdc++)

add_executable(BBoxSelector BBoxSelector.cpp Utils.cpp)
target_link_libraries(BBoxSelector ${OpenCV_LIBS})
target_link_libraries(BBoxSelector stdc++)

add_executable(bw_he BlackWhite_He.cpp Utils.cpp)
target_link_libraries(bw_he ${OpenCV_LIBS})
target_link_libraries(bw_he stdc++)

add_executable(BuildDataset BuildDataset.cpp Utils.cpp)
target_link_libraries(BuildDataset ${OpenCV_LIBS})
target_link_libraries(BuildDataset stdc++)

add_executable(DecisionTree DecisionTree_class_and_test.cpp)
target_link_libraries(DecisionTree stdc++)

add_executable(test_siftmasked test_siftMasked.cpp)
target_link_libraries(test_siftmasked ${OpenCV_LIBS})
target_link_libraries(test_siftmasked stdc++)

add_executable(bseadetector BoatSeaDetector.cpp Utils.cpp DatasetHelper.cpp SegmentationHelper.cpp KMeansClassifier.cpp)
target_link_libraries(bseadetector ${OpenCV_LIBS})
target_link_libraries(bseadetector stdc++)

add_executable(TrainAnn TrainAnn.cpp ANN.cpp Utils.cpp DatasetHelper.cpp SegmentationHelper.cpp)
target_link_libraries(TrainAnn ${OpenCV_LIBS})
target_link_libraries(TrainAnn stdc++)

add_executable(image_classifier ImageClassifier.cpp Utils.cpp DatasetHelper.cpp SegmentationHelper.cpp KMeansClassifier.cpp)
target_link_libraries(image_classifier ${OpenCV_LIBS})
target_link_libraries(image_classifier stdc++)