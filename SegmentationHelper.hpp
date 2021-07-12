#ifndef __SEGMENTATION_HELPER_HPP__
#define __SEGMENTATION_HELPER_HPP__

#define BBOX_EXT "*.txt"
#define SEA_MASK_EXT "*_mask.png"
#define BOAT_MASK_EXT "*_maskb.png"
#define MASK_TOKEN "_mask"

#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color);
void removeMasksFromImagesFnames(std::vector<cv::String>& fnames);

class SegmentationInfo {
    public:
        SegmentationInfo(cv::Mat image, cv::Mat seaMask, cv::Mat boatsMask, cv::Mat bgMask, std::vector<cv::Rect> bboxes): image(image), seaMask(seaMask), boatsMask(boatsMask), bgMask(bgMask), bboxes(bboxes) {};
        void computKeypoints(bool sharpen);
        void showLabeledKps();
        void performSegmentation(bool showResults);

    private: 
        std::vector<cv::Rect> bboxes;
        cv::Mat image, seaMask, boatsMask, bgMask;
        cv::Mat boatDescriptors, seaDescriptors, bgDescriptors;
        cv::Mat segmentationResult;
        std::vector<cv::KeyPoint> boatKps, seaKps, bgKps;
};

class SegmentationHelper {

    public:
        SegmentationHelper(cv::String& inputDirectory, cv::String& imagesExt);
        std::vector<SegmentationInfo> loadInfos(bool boatsFromBBoxes);

        
    private:
        std::vector<cv::String> filenames;
        std::vector<cv::String> bboxes_fnames;
        std::vector<cv::String> masks_fnames;
        std::vector<cv::String> boat_masks_fnames;
};



#endif