#ifndef __KMEANS_CLASSIFIER_HPP__
#define __KMEANS_CLASSIFIER_HPP__

#include "Utils.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <numeric>
#include "DatasetHelper.hpp"

class KMeansClassifier {
    public:

        KMeansClassifier(double decisionRatio): decisionRatio(decisionRatio){}

        void clusterSeaKps(std::vector<std::vector<double>>& seaKps, int k, bool printComp);
        void clusterboatsKps(std::vector<std::vector<double>>& boatsKps, int k, bool printComp);
        void clusterbgKps(std::vector<std::vector<double>>& bgKps, int k, bool printComp);

        int predictLabel(std::vector<double>& descriptor);
        std::vector<int> predictBoatsBatch(cv::Mat& descriptors, float threshold);

        void save(cv::String& inputDirectory);
        void load(cv::String& inputDirectory, bool bg = true);
    private:
        double decisionRatio;
        //std::vector<std::vector<double>> seaCentroids;
        //std::vector<std::vector<double>> boatsCentroids;
        //std::vector<std::vector<double>> bgCentroids;

        cv::Mat seaCMat, boatsCMat,  bgCMat;
};

double cluster(std::vector<std::vector<double>>& input, int k,cv::Mat& output);

#endif