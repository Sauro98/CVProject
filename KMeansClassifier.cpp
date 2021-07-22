#include "kMeansClassifier.hpp"

template <class T>
cv::Mat matFromVecOfVec(const std::vector<std::vector<double>>& input, int matType){
    if(input.size() == 0)
        return cv::Mat(cv::Size(0,0), matType);
    cv::Mat matVec = cv::Mat(input.size(), input.at(0).size(), matType);
    for(int i=0; i<matVec.rows; ++i)
        for(int j=0; j<matVec.cols; ++j)
            matVec.at<T>(i, j) = (T)input.at(i).at(j);
    return matVec;
}

void fillDoubleVectorWithMat(cv::Mat& data, std::vector<std::vector<double>>& out){
    for(int i=0; i<data.rows; ++i){
        std::vector<double> row;
        for(int j=0; j<data.cols; ++j){
            row.push_back((double)data.at<double>(i, j));
        }
        out.push_back(row);
    }
}

double cluster(std::vector<std::vector<double>>& input, int k,cv::Mat& output){
    cv::Mat in = matFromVecOfVec<float>(input, CV_32F);
    cv::Mat labels;
    std::cout<<"Input matrix size: "<<in.rows<<"x"<<in.cols<<std::endl;
    output.convertTo(output, CV_32F);
    double comp = cv::kmeans(in, k, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 20, 0), 5, cv::KmeansFlags::KMEANS_PP_CENTERS, output);
    std::cout<<"Output matrix size: "<<output.rows<<"x"<<output.cols<<", channels: "<<output.channels()<<std::endl;
    output.convertTo(output, CV_64F);
    return comp;
}

double l2Dist(std::vector<double>& a, std::vector<double>& b){
    std::vector<double> auxiliary;
    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(auxiliary),//
        [](double element1, double element2) {return pow((element1-element2),2);});
    return std::sqrt(std::accumulate(auxiliary.begin(), auxiliary.end(), 0.0));
}

double closerInMat(cv::Mat& mat, std::vector<double>& descriptor){
    cv::Mat descMat;
    descMat.push_back( cv::Mat(descriptor, false).reshape(1,1));

    double minDist = cv::norm(mat.row(0), descMat, cv::NORM_L2);
    double secondDist = 1e15;
    for(int i = 1; i < mat.rows; i++){
        double dist = cv::norm(mat.row(i), descMat, cv::NORM_L2);
        if(dist < minDist){
            secondDist = minDist;
            minDist = dist;
        }
    }

    const double ratio = 2.;
    if(minDist < ratio * secondDist)
        return minDist;
    else
        return 1e15;

    return minDist;
}

double closerInVector(std::vector<std::vector<double>>& vector, std::vector<double>& descriptor){
    double minDist = l2Dist(vector[0], descriptor);
    for(int i = 1; i < vector.size(); i++){
        double dist = l2Dist(vector[i], descriptor);
        if(dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

void KMeansClassifier::clusterSeaKps(std::vector<std::vector<double>>& seaKps, int k, bool printComp = false){
    double comp = cluster(seaKps, k, seaCMat);
    if(printComp)
        std::cout<<"Compactness for sea kps: "<<comp<<std::endl;
}
void KMeansClassifier::clusterboatsKps(std::vector<std::vector<double>>& boatsKps, int k, bool printComp = false){
    double comp = cluster(boatsKps, k, boatsCMat);
    if(printComp)
        std::cout<<"Compactness for boat kps: "<<comp<<std::endl;
}
void KMeansClassifier::clusterbgKps(std::vector<std::vector<double>>& bgKps, int k, bool printComp = false){
    double comp = cluster(bgKps, k, bgCMat);
    if(printComp)
        std::cout<<"Compactness for bg kps: "<<comp<<std::endl;
}

double findBestMatch(cv::Mat& centroids, std::vector<double> descriptor){
    std::vector<std::vector<double>> temp;
    temp.push_back(descriptor);
    cv::Mat descrMat = matFromVecOfVec<float>(temp, CV_32F);
    //cv::Mat cvtCentroids;
    //centroids.convertTo(cvtCentroids, CV_32F);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(descrMat, centroids, knn_matches, 2 );
    //std::cout<<"size "<<knn_matches.size()<<std::endl;
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 2.f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    if(good_matches.size() > 0){
        return (double)good_matches[0].distance;
    } else {
        return 1e16;
    }
}

int KMeansClassifier::predictLabel(std::vector<double>& descriptor){
    double minSeaDist, minBoatDist,minBgDist;

    minSeaDist = closerInMat(seaCMat, descriptor);
    minBoatDist = closerInMat(boatsCMat, descriptor);
    minBgDist = closerInMat(bgCMat, descriptor);

    double bestDist = minSeaDist, secondBestDist = minSeaDist;
    unsigned int bestLabel = SEA_TARGET;

    if(minBoatDist < minBgDist){
        if(minBoatDist < bestDist) {
            bestDist = minBoatDist;
            bestLabel = BOAT_TARGET;
            if(minBgDist < secondBestDist)
                secondBestDist = minBgDist;
        } else {
            secondBestDist = minBoatDist;
        }
    } else {
        if(minBgDist < bestDist) {
            bestDist = minBgDist;
            bestLabel = BG_TARGET;
            if(minBoatDist < secondBestDist)
                secondBestDist = minBoatDist;
        } else {
            secondBestDist = minBgDist;
        }
    }

    if(bestDist < decisionRatio * secondBestDist)
        return bestLabel;
    else
        return 0;
}

std::vector<int> KMeansClassifier::predictBoatsBatch(cv::Mat& descriptors, float threshold){
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors, boatsCMat32, knn_matches, 2 );
    const float ratio_thresh = decisionRatio;
    std::vector<int> labels;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            if(knn_matches[i][0].distance < threshold)
            { labels.push_back(BOAT_TARGET);}
            else
                labels.push_back(SEA_TARGET);
        else
            labels.push_back(SEA_TARGET);
    }
    return labels;
}

void KMeansClassifier::save(cv::String& inputDirectory){
    std::vector<std::vector<double>> seaCentroids, boatsCentroids, bgCentroids;
    fillDoubleVectorWithMat(seaCMat, seaCentroids);
    fillDoubleVectorWithMat(bgCMat, bgCentroids);
    fillDoubleVectorWithMat(boatsCMat, boatsCentroids);
    saveDataset(inputDirectory + "/kmclassifier/seaCentroids.txt", seaCentroids);
    saveDataset(inputDirectory + "/kmclassifier/boatsCentroids.txt", boatsCentroids);
    saveDataset(inputDirectory + "/kmclassifier/bgCentroids.txt", bgCentroids);
}

void KMeansClassifier::load(cv::String& inputDirectory, bool bg){
    std::vector<std::vector<double>> seaCentroids, boatsCentroids, bgCentroids, vInputs, tInputs;
    std::vector<uint>  vOutputs, tOutputs;
    loadDataset(inputDirectory + "/kmclassifier/seaCentroids.txt", seaCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 10000000,0,0, false);
    loadDataset(inputDirectory + "/kmclassifier/boatsCentroids.txt", boatsCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 100000000,0,0, false);
    if(bg){
        loadDataset(inputDirectory + "/kmclassifier/bgCentroids.txt", bgCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 10000000,0,0, false);
    }/*loadDataset(inputDirectory + "_sea_kp_dataset.txt", seaCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000,0,0, true);
    loadDataset(inputDirectory + "_boats_kp_dataset.txt", boatsCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000,0,0, true);
    loadDataset(inputDirectory + "_bg_kp_dataset.txt", bgCentroids, vOutputs, vInputs, vOutputs, tInputs, tOutputs, 1000000,0,0, true);
    */
    std::cout<<"Start mat creation"<<std::endl;
    bgCMat = matFromVecOfVec<double>(bgCentroids, CV_64F);
    seaCMat = matFromVecOfVec<double>(seaCentroids, CV_64F);
    boatsCMat = matFromVecOfVec<double>(boatsCentroids, CV_64F);
    boatsCMat32 = matFromVecOfVec<float>(boatsCentroids, CV_32F);

    std::cout<<"End mat creation"<<std::endl;
}
