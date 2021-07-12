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

SegmentationHelper::SegmentationHelper(cv::String& inputDirectory, cv::String& imagesExt){
    cv::utils::fs::glob(inputDirectory, cv::String(imagesExt), filenames);
    cv::utils::fs::glob(inputDirectory, cv::String(BBOX_EXT), bboxes_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(SEA_MASK_EXT), masks_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(BOAT_MASK_EXT), boat_masks_fnames);
    removeMasksFromImagesFnames(filenames);

    std::sort(filenames.begin(), filenames.end());
    std::sort(bboxes_fnames.begin(), bboxes_fnames.end());
    std::sort(masks_fnames.begin(), masks_fnames.end());

    if (filenames.size() != bboxes_fnames.size() || filenames.size() != masks_fnames.size()){
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

        SegmentationInfo info = SegmentationInfo(original_img, seaMask, boatMask, bgMask, bboxes);
        infos.push_back(info);
    }
    return infos;
}

void SegmentationInfo::computKeypoints(bool sharpen){
    SiftMasked smasked = SiftMasked();
    BlackWhite_He equalizer = BlackWhite_He();

    cv::Mat eq_img = equalizer.bgr_to_gray_HE(image, sharpen);
    boatKps = smasked.findFeatures(eq_img, boatsMask, boatDescriptors);
    seaKps = smasked.findFeatures(eq_img, seaMask, seaDescriptors);
    bgKps = smasked.findFeatures(eq_img, bgMask, bgDescriptors);
}

void SegmentationInfo::showLabeledKps(){
    cv::Mat kpImg = image.clone();
    cv::drawKeypoints(kpImg, boatKps, kpImg, cv::Scalar(0,255,0));
    cv::drawKeypoints(kpImg, seaKps, kpImg, cv::Scalar(0,0,255));
    cv::drawKeypoints(kpImg, bgKps, kpImg, cv::Scalar(255,0,0));
    cv::imshow("kps", kpImg);
    cv::waitKey(0);
}

void SegmentationInfo::performSegmentation(bool showResults) {
    cv::Mat markersMask = cv::Mat::zeros(image.size(), CV_8U);
    drawMarkers(markersMask,boatKps, cv::Scalar::all(1));
    drawMarkers(markersMask,seaKps, cv::Scalar::all(2));
    drawMarkers(markersMask,bgKps, cv::Scalar::all(3));

    markersMask.convertTo(markersMask, CV_32S);
    
    cv::watershed(image, markersMask);

    Mat wshed(markersMask.size(), CV_8UC3);

    for(int r = 0; r < markersMask.rows; r++ )
        for(int c = 0; c < markersMask.cols; c++ )
        {
            int index = markersMask.at<int>(r,c);
            if( index == -1 )
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,255);
            else if (index == 1)
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,0);
            else if (index == 2)
                wshed.at<Vec3b>(r,c) = Vec3b(0,0,255);
            else
                wshed.at<Vec3b>(r,c) = Vec3b(255,0,0);
        }

    segmentationResult = wshed.clone();

    if (showResults) {
        wshed = wshed*0.5 + image*0.5;
        imshow( "watershed transform", wshed );
        waitKey(0);
    }
}