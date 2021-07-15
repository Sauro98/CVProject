#ifndef __SEGMENTATION_HELPER_HPP__
#define __SEGMENTATION_HELPER_HPP__

#define BBOX_EXT "*.txt"
#define SEA_MASK_EXT "*_mask.png"
#define BOAT_MASK_EXT "*_maskb.png"
#define MASK_TOKEN "_mask"

#define BOAT_LABEL 1
#define SEA_LABEL 2
#define BG_LABEL 3

#define BG_CH_INDEX 0 // blue channel
#define BOATS_CH_INDEX 1 // green channel
#define SEA_CH_INDEX 2 // red channel

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
        SegmentationInfo(cv::Mat image, cv::Mat seaMask, cv::Mat boatsMask, cv::Mat bgMask, std::vector<cv::Rect> bboxes, cv::String imageName): image(image), seaMask(seaMask), boatsMask(boatsMask), bgMask(bgMask), trueBboxes(bboxes), imageName(imageName) {
            estBboxes = std::vector<cv::Rect>();
        };
        void computKeypoints(bool sharpen);
        void showLabeledKps();
        void performSegmentation(bool showResults);
        std::vector<double> computeIOU(bool showBoxes);
        double computePixelAccuracy();
        cv::String& getName();

    private: 
        cv::String imageName;
        std::vector<cv::Rect> trueBboxes, estBboxes;
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