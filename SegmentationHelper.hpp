#ifndef __SEGMENTATION_HELPER_HPP__
#define __SEGMENTATION_HELPER_HPP__

#define BBOX_EXT "*.txt"
#define SEA_MASK_EXT "*_mask.png"
#define BOAT_MASK_EXT "*_maskb.png"
#define MASK_TOKEN "_mask"
#define DATASET_TOKEN "kp_dataset"
#define PARAMETERS_TOKEN "parameters"

// target values defined in DatasetHelper.hpp
#define BOAT_LABEL BOAT_TARGET
#define SEA_LABEL SEA_TARGET
#define BG_LABEL BG_TARGET

#define BG_CH_INDEX 0 // blue channel
#define BOATS_CH_INDEX 1 // green channel
#define SEA_CH_INDEX 2 // red channel

#define BOAT_GRID_INDEX 0
#define SEA_GRID_INDEX 1
#define BG_GRID_INDEX 2

#include <iostream>
#include <fstream>
#include <random>
#include <iterator>
#include <math.h>
#include <stdlib.h>

#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include "DatasetHelper.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/objdetect.hpp>

typedef unsigned int (*classFunc)(std::vector<double>&, void*);

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color);
void removeMasksFromImagesFnames(std::vector<cv::String>& fnames);
void removeDatasetsFromBBoxesFnames(std::vector<cv::String>& fnames);

class SegmentationInfo {
    public:
        SegmentationInfo(cv::Mat image, cv::Mat seaMask, cv::Mat boatsMask, cv::Mat bgMask, std::vector<cv::Rect> bboxes, cv::String imageName): image(image), seaMask(seaMask), boatsMask(boatsMask), bgMask(bgMask), trueBboxes(bboxes), imageName(imageName) {
            estBboxes = std::vector<cv::Rect>();
        };
        void computeKeypoints(bool sharpen, classFunc classify = nullptr, void* usrData = nullptr, unsigned int numThread = 1);
        void showLabeledKps();
        void performSegmentation(bool showResults, bool addBg, uint maxDim, double minNormVariance);
        void findBBoxes(bool showBoxes, double minPercArea, double maxOverlapMetric);
        std::vector<double> computeIOU(bool showBoxes, double minPercArea, double maxOverlapMetric, uint& falsePos, uint& falseNeg);
        double computePixelAccuracy();
        cv::String& getName();
        void appendBoatsDescriptors(std::vector<std::vector<double>>& vect, bool addEnc) const;
        void appendSeaDescriptors(std::vector<std::vector<double>>& vect, bool addEnc) const;
        void appendBgDescriptors(std::vector<std::vector<double>>& vect, bool addEnc) const;

        std::vector<cv::Rect>& getBBoxes(){ return estBboxes;}
        cv::Mat& getsegmentationResults(){return segmentationResult;}
        std::vector<cv::KeyPoint> getboatKps() {return boatKps;}

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
