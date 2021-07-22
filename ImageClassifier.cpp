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


//double vectorAvg(std::vector<double>& v);
//void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, KMeansClassifier* classifier = nullptr, bool addBg = true, uint maxDim = 45, double minNormVariance = 0.01, double minPercArea = 0.02, double maxOverlapMetric = 1.5);

unsigned int kmeansCallback(std::vector<double>& input, void* usrData){
	KMeansClassifier* cl = (KMeansClassifier*)usrData;
    return cl->predictLabel(input);
}

void printHelp() {
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
    std::cout<<" Usage: "<<std::endl;
    std::cout<<"  ./image_classifier image_path params_path centroids_path"<<std::endl;
    std::cout<<"    - image_path: image to classify"<<std::endl;
    std::cout<<"    - params_path: path to the txt params file (containing \"params\" in the filename)"<<std::endl;
    std::cout<<"    - centroids_path: path to the folder containing the \"kmclassifier\" centroids folder"<<std::endl;
    std::cout<<"--------------------------------------------------------------------------------------------"<<std::endl;
}

int main(int argc, char** argv){
    if(argc < 4){
        printHelp();
        return 1;
    } 

    cv::String params_path = cv::String(argv[3]);
    cv::String image_path = cv::String(argv[1]);
    std::vector<cv::Rect> dummy;
    cv::Mat image = cv::imread(image_path);
    
    SegmentationParams params = readParamsFromFile(cv::String(argv[2]));
    printParams(params);

    KMeansClassifier classifier(params.decisionRatio);
    classifier.load(params_path, true);
    SegmentationInfo imageInfo = SegmentationInfo(image, cv::Mat(), cv::Mat(), cv::Mat(), dummy, image_path);
    imageInfo.computeKeypoints(true, kmeansCallback, &classifier, 8);
    imageInfo.showLabeledKps();
    imageInfo.performSegmentation(true, params.addBg, params.maxDim, params.minNormVariance);
    imageInfo.findBBoxes(true, params.minPercArea, params.maxOverlapMetric);
    cv::waitKey(0);
}