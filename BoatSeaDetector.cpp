#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "SegmentationHelper.hpp"
#include "DatasetHelper.hpp"

double vectorAvg(std::vector<double>& v);
void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed);

int main(int argc, char** argv)
{
    cv::String input_directory = cv::String(argv[1]);
    cv::String images_ext = cv::String(argv[2]);

    SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);
    computeShowMetrics(segmentationInfos,false, false);
    std::vector<std::vector<double>> descriptors;
    for(const auto& info: segmentationInfos){
        info.appendBgDescriptors(descriptors);
        info.appendBoatsDescriptors(descriptors);
        info.appendSeaDescriptors(descriptors);
    }
    std::cout<<std::endl;
    std::cout<<"saving whole dataset..."<<std::endl;
    saveDataset(input_directory + "kp_dataset.txt", descriptors);

    descriptors.erase(descriptors.begin(), descriptors.end());

    std::vector<std::vector<double>> inputs, vInputs, tInputs;
    std::vector<uint> outputs, vOutputs, tOutputs;
    std::cout<<"loading whole dataset ..."<<std::endl;
    loadDataset(input_directory + "kp_dataset.txt", inputs, outputs, vInputs, vOutputs, tInputs, tOutputs, 100, 100, 100);
}

void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed){
    std::vector<double> allIous;
    std::vector<double> allPixAcc;
    std::cout<<std::endl;
    for(size_t i = 0; i < infos.size(); i++) {
        auto& imageInfo = infos[i];
        if(!detailed){
            std::cout<<"Image ("<<i+1<<"/"<<infos.size()<<")"<<std::endl;
        }
        imageInfo.computeKeypoints(true);
        if(displayImages){
            imageInfo.showLabeledKps();
        }
        imageInfo.performSegmentation(displayImages);
        auto ious = imageInfo.computeIOU(displayImages);
        if(detailed){
            std::cout<<imageInfo.getName()<<std::endl;
            for(const auto& iou: ious)
                std::cout<<"iou: "<<iou<<std::endl;
        }
        
        allIous.insert(allIous.end(), ious.begin(), ious.end());
        double pixAcc = imageInfo.computePixelAccuracy();
        allPixAcc.push_back(pixAcc);
        if(detailed){
            std::cout<<"Pixel accuracy: "<<pixAcc*100.<<"%"<<std::endl;
        }
        if(displayImages)
            cv::waitKey(0);
    }

    if(allIous.size() > 0){    
        std::cout<<std::endl;
        std::sort(allIous.begin(), allIous.end());
        double avgIou = vectorAvg(allIous);
        std::cout<<" - Iou (average, min, max) = ("<<avgIou<<", "<<allIous[0]<<", "<<allIous[allIous.size() - 1]<<")"<<std::endl;
    }

    if(allPixAcc.size() > 0){
        std::cout<<std::endl;
        std::sort(allPixAcc.begin(), allPixAcc.end());
        double avgPixacc = vectorAvg(allPixAcc);
        std::cout<<" - Pixel accuracy (average, min, max) = ("<<avgPixacc<<", "<<allPixAcc[0]<<", "<<allPixAcc[allPixAcc.size() - 1]<<")"<<std::endl;
    }
}

double vectorAvg(std::vector<double>& v){
    double avg = 0.;
    for(const auto& d: v)
        avg += d;
    return avg / v.size();
}
