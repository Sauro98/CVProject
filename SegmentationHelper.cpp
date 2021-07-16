#include "SegmentationHelper.hpp"

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color){
    for (int i = 0; i < kps.size(); i++) {
        cv::Point2f pti = kps[i].pt;
        cv::circle( markers, pti,1, color, -1, 8, 0 );
    }
}

void removeMasksFromImagesFnames(std::vector<cv::String>& fnames){
    fnames.erase(std::remove_if(fnames.begin(), fnames.end(), [](const cv::String& f) {
        return f.find(cv::String(MASK_TOKEN)) != cv::String::npos;
    }), fnames.end());
}

void removeDatasetsFromBBoxesFnames(std::vector<cv::String>& fnames){
    fnames.erase(std::remove_if(fnames.begin(), fnames.end(), [](const cv::String& f) {
        return f.find(cv::String(DATASET_TOKEN)) != cv::String::npos;
    }), fnames.end());
}

SegmentationHelper::SegmentationHelper(cv::String& inputDirectory, cv::String& imagesExt){
    cv::utils::fs::glob(inputDirectory, cv::String(imagesExt), filenames);
    cv::utils::fs::glob(inputDirectory, cv::String(BBOX_EXT), bboxes_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(SEA_MASK_EXT), masks_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(BOAT_MASK_EXT), boat_masks_fnames);
    removeMasksFromImagesFnames(filenames);
    removeDatasetsFromBBoxesFnames(bboxes_fnames);

    std::sort(filenames.begin(), filenames.end());
    std::sort(bboxes_fnames.begin(), bboxes_fnames.end());
    std::sort(masks_fnames.begin(), masks_fnames.end());
    std::sort(boat_masks_fnames.begin(), boat_masks_fnames.end());

    if (filenames.size() != bboxes_fnames.size() || filenames.size() != masks_fnames.size() || filenames.size() != boat_masks_fnames.size()){
        std::cout<<"Some masks/bboxes are missing"<<std::endl;
        exit(1);
    }
}

std::vector<SegmentationInfo> SegmentationHelper::loadInfos(bool boatsFromBBoxes){
    std::vector<SegmentationInfo> infos;
    SiftMasked smasked= SiftMasked();
    for (int i = 0; i < filenames.size(); i++) {
        int check = 0;
        cv::Mat original_img = cv::imread(filenames[i], cv::IMREAD_COLOR);
        cv::Mat seaMask = cv::imread(masks_fnames[i], cv::IMREAD_GRAYSCALE);
        std::vector<cv::Rect> bboxes = smasked.checkFileBB(bboxes_fnames[i], check);
        if(check == 1) {
            exit(1);
        }
        cv::Mat boatMask;
        if(boatsFromBBoxes) {
            boatMask = smasked.findBinMask(original_img,bboxes);
        } else {
            boatMask = cv::imread(boat_masks_fnames[i], cv::IMREAD_GRAYSCALE);
        }
        boatMask.convertTo(boatMask, CV_32F);
        seaMask.convertTo(seaMask, CV_32F);
        cv::Mat intersection = boatMask.mul(seaMask);
        boatMask = boatMask - intersection;
        cv::Mat bgMask = 1 - seaMask - boatMask - intersection;

        boatMask.convertTo(boatMask, CV_8U);
        seaMask.convertTo(seaMask, CV_8U);
        bgMask.convertTo(bgMask, CV_8U);

        SegmentationInfo info = SegmentationInfo(original_img, seaMask, boatMask, bgMask, bboxes, filenames[i]);
        infos.push_back(info);
    }
    return infos;
}

void SegmentationInfo::computeKeypoints(bool sharpen, classFunc classify){
    SiftMasked smasked = SiftMasked();
    BlackWhite_He equalizer = BlackWhite_He();
    cv::Mat eq_img = equalizer.bgr_to_gray_HE(image, sharpen);
    
    if(classify)
    {
        cv::Mat allDescriptors;
        std::vector<cv::KeyPoint> allKP = smasked.findFeatures(eq_img, cv::Mat(), allDescriptors);
        std::vector<std::vector<double>> descVect;
        appendDescriptors(descVect, allDescriptors, 0, false);
        
        boatKps.clear();
        seaKps.clear();
        bgKps.clear();
        
        for(unsigned int i=0; i<allKP.size(); ++i)
        {
            const unsigned int classID = classify(descVect[i]);
            if (classID == BOAT_LABEL)
            {
                boatKps.push_back(allKP[i]);
            }
            else if (classID == SEA_LABEL)
            {
                seaKps.push_back(allKP[i]);
            }
            else
            {
                bgKps.push_back(allKP[i]);
            }
            
            smasked.findDescriptors(eq_img, boatKps, boatDescriptors);
            smasked.findDescriptors(eq_img, seaKps, seaDescriptors);
            smasked.findDescriptors(eq_img, bgKps, bgDescriptors);
        }
    }
    else
    {
        boatKps = smasked.findFeatures(eq_img, boatsMask, boatDescriptors);
        seaKps = smasked.findFeatures(eq_img, seaMask, seaDescriptors);
        bgKps = smasked.findFeatures(eq_img, bgMask, bgDescriptors);
    }
}

void SegmentationInfo::showLabeledKps(){
    cv::Mat kpImg = image.clone();
    cv::drawKeypoints(kpImg, boatKps, kpImg, cv::Scalar(0,255,0));
    cv::drawKeypoints(kpImg, seaKps, kpImg, cv::Scalar(0,0,255));
    cv::drawKeypoints(kpImg, bgKps, kpImg, cv::Scalar(255,0,0));
    cv::imshow("kps", kpImg);
}

void SegmentationInfo::performSegmentation(bool showResults) {
    cv::Mat markersMask = cv::Mat::zeros(image.size(), CV_8U);
    drawMarkers(markersMask,boatKps, cv::Scalar::all(BOAT_LABEL));
    drawMarkers(markersMask,seaKps, cv::Scalar::all(SEA_LABEL));
    drawMarkers(markersMask,bgKps, cv::Scalar::all(BG_LABEL));

    markersMask.convertTo(markersMask, CV_32S);
    
    cv::watershed(image, markersMask);

    Mat wshed(markersMask.size(), CV_8UC3);

    for(int r = 0; r < markersMask.rows; r++ )
        for(int c = 0; c < markersMask.cols; c++ )
        {
            int index = markersMask.at<int>(r,c);
            if( index == -1 )
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,255);
            else if (index == BOAT_LABEL)
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,0);
            else if (index == SEA_LABEL)
                wshed.at<Vec3b>(r,c) = Vec3b(0,0,255);
            else
                wshed.at<Vec3b>(r,c) = Vec3b(255,0,0);
        }

    segmentationResult = wshed.clone();

    if (showResults) {
        wshed = wshed*0.5 + image*0.5;
        imshow( "watershed transform", wshed );
    }
}

cv::Mat getBoatsMaskErodedDilated(cv::Mat segmentationResult){
    // keep only green (boats) channel
    cv::Mat boatsSegments = segmentationResult.clone();
    cv::Mat chs[3];
    cv::split(boatsSegments,chs);
    boatsSegments = chs[1];

    // erode mask with elements:
    // 1 1 1
    // 1 1 1
    // 1 1 1
    uchar erosionComponents[] = {1,1,1,1,1,1,1,1,1};
    cv::Mat erosionElement = cv::Mat(3,3,CV_8UC1, erosionComponents);
    cv::erode(boatsSegments, boatsSegments, erosionElement);
    // and then dilate
    cv::dilate(boatsSegments, boatsSegments, erosionElement);

    return boatsSegments;
}

void filterBoundingBoxesByArea(std::vector<cv::Rect>& bboxes, double ratio) {
    double avg_area = 0;
    for(auto& bbox: bboxes){
        avg_area += (double)bbox.area();
    }
    avg_area /= (double)bboxes.size();

    bboxes.erase(std::remove_if(bboxes.begin(), bboxes.end(), [avg_area, ratio](const cv::Rect& r) {
        return r.area() <= ratio*avg_area;
    }), bboxes.end());
}

std::vector<cv::Mat> masksForBoxes(std::vector<cv::Rect>& boxes, cv::Size img_size) {
    std::vector<cv::Mat> masks;
    for(const auto& b: boxes){
        cv::Mat mask = cv::Mat::zeros(img_size, CV_8UC1);
        cv::rectangle(mask,b,cv::Scalar(1),-1, LINE_8);
        masks.push_back(mask);
    }
    return masks;
}

std::vector<double> SegmentationInfo::computeIOU(bool showBoxes){
    std::vector<double> ious;
    SiftMasked smasked = SiftMasked();

    // extract boat-labeled pixels and perform erosion/dilation
    cv::Mat boatsSegments = getBoatsMaskErodedDilated(segmentationResult);
    // compute bounding boxes on the result
    smasked.binaryToBBoxes(boatsSegments, estBboxes, true);
    // filter bboxes with area <= 2% of the mean area
    filterBoundingBoxesByArea(estBboxes, 0.02);
    // precompute target bboxes masks
    auto targetBBoxesMasks = masksForBoxes(trueBboxes, image.size());
    // precompute estimated bboxes
    auto estBBoxesMasks = masksForBoxes(estBboxes, image.size());

    for(const auto& estBBoxMask: estBBoxesMasks){
        if(targetBBoxesMasks.size() == 0){
            std::cout<<"Warning, there are more estimated bboxes than real ones"<<std::endl;
            break;
        }
        int intersectionArea = cv::countNonZero(targetBBoxesMasks[0].mul(estBBoxMask));
        size_t best_index = 0;
        for(size_t i = 1; i < targetBBoxesMasks.size(); i++){
            int tempIntArea = cv::countNonZero(targetBBoxesMasks[i].mul(estBBoxMask));
            if(tempIntArea > intersectionArea){
                intersectionArea = tempIntArea;
                best_index = i;
            }
        }
        
        int unionArea = cv::countNonZero(targetBBoxesMasks[best_index] + estBBoxMask);
        double iou = (double)intersectionArea/(double)unionArea;
        ious.push_back(iou);
        targetBBoxesMasks.erase(targetBBoxesMasks.begin() + best_index);
    }

    // display bounding boxes
    if(showBoxes){
        cv::Mat bboxes_img = image.clone();
        for(auto& box: trueBboxes) {
            cv::rectangle(bboxes_img, box, cv::Scalar(255,0,0),3);
        }
        for(auto& box: estBboxes) {
            cv::rectangle(bboxes_img, box, cv::Scalar(0,255,0),2);
        }
        cv::imshow("bboxes", bboxes_img);
    }

    return ious;
}

double SegmentationInfo::computePixelAccuracy(){
    cv::Mat chs[3];
    cv::split(segmentationResult,chs);
    cv::Mat seaSegments = chs[SEA_CH_INDEX].clone();
    cv::Mat otherSegments = chs[BOATS_CH_INDEX].clone() + chs[BG_CH_INDEX].clone();
    cv::Mat correctSeaPixels = seaSegments.mul(seaMask);
    cv::Mat correctOtherPixels = otherSegments.mul(bgMask + boatsMask);
    int correctPixels = cv::countNonZero(correctSeaPixels) + cv::countNonZero(correctOtherPixels);
    int totalPixels = image.rows * image.cols;
    return (double) correctPixels / (double) totalPixels;
}

void SegmentationInfo::appendBoatsDescriptors(std::vector<std::vector<double>>& vect) const {
    appendDescriptors(vect, boatDescriptors, BOATS_1H_ENC);
}
void SegmentationInfo::appendSeaDescriptors(std::vector<std::vector<double>>& vect) const {
    appendDescriptors(vect, seaDescriptors, SEA_1H_ENC);
}
void SegmentationInfo::appendBgDescriptors(std::vector<std::vector<double>>& vect) const {
    appendDescriptors(vect, bgDescriptors, BG_1H_ENC);
}

cv::String& SegmentationInfo::getName(){
    return imageName;
}
