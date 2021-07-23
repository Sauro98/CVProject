#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "SegmentationHelper.hpp"
#include "DatasetHelper.hpp"
#include "kMeansClassifier.hpp"

typedef struct {
    bool addBg = true;
    unsigned int maxDim = 45;
    double minNormVariance = 0.01;
    double minPercArea = 0.02;
    double maxOverlapMetric = 1.5;
    double decisionRatio = 0.9;
} SegmentationParams;

SegmentationParams readParamsFromFile(const std::string& filename){
    SegmentationParams params;
    std::ifstream file;
    file.open(filename);
    std::string line;
    file >> params.addBg;
    file >> params.maxDim;
    file >> params.minNormVariance;
    file >> params.minPercArea;
    file >> params.maxOverlapMetric;
    file >> params.decisionRatio;
    file.close();
    return params;
}

void printParams(SegmentationParams& params){
    std::cout<<"--- PARAMS ---"<<std::endl;
    std::cout<<"Add background: "<<params.addBg<<std::endl;
    std::cout<<"Max grid size: "<<params.maxDim<<std::endl;
    std::cout<<"Minimum normalized variance: "<<params.minNormVariance<<std::endl;
    std::cout<<"Minimum area percentage: "<<params.minPercArea<<std::endl;
    std::cout<<"Maximum overlap for merging: "<<params.maxOverlapMetric<<std::endl;
    std::cout<<"Decision ratio: "<<params.decisionRatio<<std::endl;
    std::cout<<"--- ---"<<std::endl;
}

void writeParamsToFile(SegmentationParams& params, const std::string& filename) {
    std::ofstream ofs;
    ofs.open(filename);
    ofs << params.addBg << std::endl;
    ofs << params.maxDim << std::endl;
    ofs << params.minNormVariance << std::endl;
    ofs << params.minPercArea << std::endl;
    ofs << params.maxOverlapMetric << std::endl;
    ofs << params.decisionRatio << std::endl;
    ofs.close();
}


double vectorAvg(std::vector<double>& v);
void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, KMeansClassifier* classifier = nullptr, bool addBg = true, uint maxDim = 45, double minNormVariance = 0.01, double minPercArea = 0.02, double maxOverlapMetric = 1.5);

unsigned int kmeansCallback(std::vector<double>& input, void* usrData){
	KMeansClassifier* cl = (KMeansClassifier*)usrData;
    return cl->predictLabel(input);
}

void printHelp() {
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<" Usage: "<<std::endl;
    std::cout<<"  ./bseadetector input_directory format params_path build_kp_dataset build_k_centroids [k]"<<std::endl;
    std::cout<<"    - input_directory: directory containing the images to classify"<<std::endl;
    std::cout<<"    - format: expression for the images format (\"*.jpg\", \"*.png\", ...)"<<std::endl;
    std::cout<<"    - params_path: path to the txt params file (containing \"params\" in the filename)"<<std::endl;
    std::cout<<"    - build_kp_dataset: boolean, if true it builds a keypoint dataset from the input images"<<std::endl;
    std::cout<<"    - build_k_centroids: boolean, if true it performs k means on the keypoint dataset"<<std::endl;
    std::cout<<"    - [k]: if `build_k_centroids` is true, it specifies the number `k` of clusters"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
}

int main(int argc, char** argv)
{

    if(argc < 6){
        printHelp();
        return 1;
    } 
    cv::String input_directory = cv::String(argv[1]);
    cv::String images_ext = cv::String(argv[2]);
    SegmentationParams params = readParamsFromFile(cv::String(argv[3]));
    bool buildKpDataset = std::atoi(argv[4]);
    bool buildKcentroids = std::atoi(argv[5]);
    uint numClusters = 1000;
    if(buildKcentroids)
        if(argc == 7)
            numClusters = std::atoi(argv[6]);
        else{
            printHelp();
            return 1;
        }


    printParams(params);

    if(buildKpDataset){
        SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
        auto segmentationInfos = sHelper.loadInfos(false);
        computeShowMetrics(segmentationInfos,true, false, nullptr, params.addBg, params.maxDim, params.minNormVariance, params.minPercArea, params.maxOverlapMetric);
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
        saveDataset(input_directory + "_sea_kp_dataset.txt", seaDescriptors);
    }
    
    std::vector<std::vector<double>> seaInputs,boatsInputs, bgInputs, vInputs, tInputs;
    std::vector<uint> seaOutputs,boatsOutputs, bgOutputs, vOutputs, tOutputs;

    if(buildKcentroids){
        std::cout<<"loading bg dataset ..."<<std::endl;
        loadDataset(input_directory + "_bg_kp_dataset.txt", bgInputs, bgOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
        std::cout<<"loading boats dataset ..."<<std::endl;
        loadDataset(input_directory + "_boats_kp_dataset.txt", boatsInputs, boatsOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
        std::cout<<"loading sea dataset ..."<<std::endl;
        loadDataset(input_directory + "_sea_kp_dataset.txt", seaInputs, seaOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000000, 0, 0);
    }
    

    KMeansClassifier classifier(params.decisionRatio);
    
    if(buildKcentroids){
        classifier.clusterSeaKps(seaInputs,numClusters, true);
        classifier.clusterboatsKps(boatsInputs, numClusters, true);
        classifier.clusterbgKps(bgInputs, numClusters, true);
        classifier.save(input_directory);
    }
    
    
    if(!buildKcentroids)
        classifier.load(input_directory,true);


    SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);
    computeShowMetrics(segmentationInfos,true, true, &classifier, params.addBg, params.maxDim, params.minNormVariance, params.minPercArea, params.maxOverlapMetric);
    //writeParamsToFile(params, input_directory + "parameters.txt");
}

void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, KMeansClassifier* classifier,bool addBg,  uint maxDim, double minNormVariance, double minPercArea, double maxOverlapMetric){
    std::vector<double> allIous;
    std::vector<double> allPixAcc;
    uint falsePos = 0, falseNeg = 0;
    std::cout<<std::endl;
    for(size_t i = 0; i < infos.size(); i++) {
        auto& imageInfo = infos[i];
        if(!detailed){
            std::cout<<"Image ("<<i+1<<"/"<<infos.size()<<"): "<<imageInfo.getName()<<std::endl;
        }
        //std::cout<<"pre compute kp"<<std::endl;
        if(classifier == nullptr)
            imageInfo.computeKeypoints(true);
        else
            imageInfo.computeKeypoints(true,kmeansCallback,classifier,8);

        if(displayImages){
            imageInfo.showLabeledKps();
        }
        //std::cout<<"pre segmentation"<<std::endl;
        imageInfo.performSegmentation(displayImages, addBg, maxDim, minNormVariance);
        //std::cout<<"pre IOU"<<std::endl;
        uint falsePosCurr= 0;
        uint falseNegCurr = 0;
        auto ious = imageInfo.computeIOU(displayImages, minPercArea, maxOverlapMetric, falsePosCurr, falseNegCurr);
        falsePos += falsePosCurr;
        falseNeg += falseNegCurr;
        //std::cout<<"post IOU"<<std::endl;
        if(detailed){
            std::cout<<imageInfo.getName()<<std::endl;
            for(const auto& iou: ious)
                std::cout<<"iou: "<<iou<<std::endl;
            std::cout<<"False positives: "<<falsePosCurr<<std::endl;
            std::cout<<"False negatives: "<<falseNegCurr<<std::endl;
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
        std::cout<<"   Total false positives: "<<falsePos<<std::endl;
        std::cout<<"   Total false negatives: "<<falseNeg<<std::endl;
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
