#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "SegmentationHelper.hpp"
#include "DatasetHelper.hpp"
#include "kMeansClassifier.hpp"

double vectorAvg(std::vector<double>& v);
void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, KMeansClassifier* classifier = nullptr);

unsigned int kmeansCallback(std::vector<double>& input, void* usrData)
{
	KMeansClassifier* cl = (KMeansClassifier*)usrData;
    return cl->predictLabel(input);
}
	

int main(int argc, char** argv)
{
    cv::String input_directory = cv::String(argv[1]);
    cv::String images_ext = cv::String(argv[2]);

    /*SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);
    computeShowMetrics(segmentationInfos,false, false);
    std::vector<std::vector<double>> boatsDescriptors;
    std::vector<std::vector<double>> seaDescriptors;
    std::vector<std::vector<double>> bgDescriptors;
    for(const auto& info: segmentationInfos){
        info.appendBgDescriptors(bgDescriptors, true);
        info.appendBoatsDescriptors(boatsDescriptors, true);
        info.appendSeaDescriptors(seaDescriptors, true);
    }
    std::cout<<std::endl;
    std::cout<<"saving whole dataset..."<<std::endl;
    saveDataset(input_directory + "_bg_kp_dataset.txt", bgDescriptors);
    saveDataset(input_directory + "_boats_kp_dataset.txt", boatsDescriptors);
    saveDataset(input_directory + "_sea_kp_dataset.txt", seaDescriptors);*/

    //descriptors.erase(descriptors.begin(), descriptors.end());

    bool build = false;
    
    std::vector<std::vector<double>> seaInputs,boatsInputs, bgInputs, vInputs, tInputs;
    std::vector<uint> seaOutputs,boatsOutputs, bgOutputs, vOutputs, tOutputs;

    if(build){
        std::cout<<"loading bg dataset ..."<<std::endl;
        loadDataset(input_directory + "_bg_kp_dataset.txt", bgInputs, bgOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
        std::cout<<"loading boats dataset ..."<<std::endl;
        loadDataset(input_directory + "_boats_kp_dataset.txt", boatsInputs, boatsOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
        std::cout<<"loading sea dataset ..."<<std::endl;
        loadDataset(input_directory + "_sea_kp_dataset.txt", seaInputs, seaOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
    }
    

    KMeansClassifier classifier(100000000000.);
    
    if(build){
        classifier.clusterSeaKps(seaInputs,500, true);
        classifier.clusterboatsKps(boatsInputs, 500, true);
        classifier.clusterbgKps(bgInputs, 500, true);
        classifier.save(input_directory);
    }
    
    
    if(!build)
        classifier.load(input_directory,true);


    SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);
    computeShowMetrics(segmentationInfos,true, false, &classifier);

}

void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, KMeansClassifier* classifier){
    std::vector<double> allIous;
    std::vector<double> allPixAcc;
    std::cout<<std::endl;
    for(size_t i = 0; i < infos.size(); i++) {
        auto& imageInfo = infos[i];
        if(!detailed){
            std::cout<<"Image ("<<i+1<<"/"<<infos.size()<<"): "<<imageInfo.getName()<<std::endl;
        }
        std::cout<<"pre compute kp"<<std::endl;
        if(classifier == nullptr)
            imageInfo.computeKeypoints(true);
        else
            imageInfo.computeKeypoints(true,kmeansCallback,classifier,8);

        if(displayImages){
            imageInfo.showLabeledKps();
        }
        std::cout<<"pre segmentation"<<std::endl;
        imageInfo.performSegmentation(displayImages);
        std::cout<<"pre IOU"<<std::endl;
        auto ious = imageInfo.computeIOU(displayImages);
        std::cout<<"post IOU"<<std::endl;
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
