#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color);

int main(int argc, char** argv)
{
    // Expects at least one argument which is the path to a directory containing the images on which the user
	// needs to select bounding boxes. It exects jpg images and it will write one txt file for each image in the same
	// folder, with the same name as the image it refers to.
	std::vector<cv::String> filenames;
	std::vector<cv::String> bboxes_fnames;
    std::vector<cv::String> masks_fnames;
    	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.jpg"), filenames);
    	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.txt"), bboxes_fnames);
        cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.png"), masks_fnames);
    	//cv::namedWindow("BBoxSelector");
	std::cout<<filenames.size()<<" images found\n";

    std::sort(filenames.begin(), filenames.end());
    std::sort(bboxes_fnames.begin(), bboxes_fnames.end());
    std::sort(masks_fnames.begin(), masks_fnames.end());

    if (filenames.size() != bboxes_fnames.size() || filenames.size() != masks_fnames.size()){
        std::cout<<"Some masks/bboxes are missing"<<std::endl;
        return 1;
    }

    SiftMasked smasked = SiftMasked();
    BlackWhite_He equalizer = BlackWhite_He();


    for (int i = 0; i < filenames.size(); i++) {
        cv::Mat original_img = cv::imread(filenames[i], cv::IMREAD_COLOR);
        cv::Mat seaMask = cv::imread(masks_fnames[i], cv::IMREAD_GRAYSCALE);
        int check = 0;
        std::vector<cv::Rect> bboxes = smasked.checkFileBB(bboxes_fnames[i], check);
        if(check == 1) {
            return 1;
        }
        cv::Mat boatMask = smasked.findBinMask(original_img,bboxes);
        boatMask.convertTo(boatMask, CV_32F);
        seaMask.convertTo(seaMask, CV_32F);
        cv::Mat intersection = boatMask.mul(seaMask);
        boatMask = boatMask - intersection;
        cv::Mat bg_mask = 1 - seaMask - boatMask - intersection;

        boatMask.convertTo(boatMask, CV_8U);
        seaMask.convertTo(seaMask, CV_8U);
        bg_mask.convertTo(bg_mask, CV_8U);

        cv::Mat eq_img = equalizer.bgr_to_gray_HE(original_img, true);

        cv::Mat boat_descriptors, sea_descriptors, bg_descriptors;
        std::vector<cv::KeyPoint> boat_kps = smasked.findFeatures(eq_img, boatMask, boat_descriptors);
        std::vector<cv::KeyPoint> sea_kps = smasked.findFeatures(eq_img, seaMask, sea_descriptors);
        std::vector<cv::KeyPoint> bg_kps = smasked.findFeatures(eq_img, bg_mask, bg_descriptors);

        cv::Mat kp_img = original_img.clone();
        cv::drawKeypoints(kp_img, boat_kps, kp_img, cv::Scalar(0,255,0));
        cv::drawKeypoints(kp_img, sea_kps, kp_img, cv::Scalar(0,0,255));
        cv::drawKeypoints(kp_img, bg_kps, kp_img, cv::Scalar(255,0,0));
        cv::imshow("kps", kp_img);
        cv::waitKey(0);

        cv::Mat markers_mask = cv::Mat::zeros(original_img.size(), CV_8U);
        drawMarkers(markers_mask,boat_kps, cv::Scalar::all(1));
        drawMarkers(markers_mask,sea_kps, cv::Scalar::all(2));
        drawMarkers(markers_mask,bg_kps, cv::Scalar::all(3));

        markers_mask.convertTo(markers_mask, CV_32S);
        
        cv::watershed(original_img, markers_mask);

        Mat wshed(markers_mask.size(), CV_8UC3);

        for(int r = 0; r < markers_mask.rows; r++ )
            for(int c = 0; c < markers_mask.cols; c++ )
            {
                int index = markers_mask.at<int>(r,c);
                if( index == -1 )
                    wshed.at<Vec3b>(r,c) = Vec3b(0,255,255);
                else if (index == 1)
                    wshed.at<Vec3b>(r,c) = Vec3b(0,255,0);
                else if (index == 2)
                    wshed.at<Vec3b>(r,c) = Vec3b(0,0,255);
                else
                    wshed.at<Vec3b>(r,c) = Vec3b(255,0,0);
            }
    
        wshed = wshed*0.5 + original_img*0.5;
        imshow( "watershed transform", wshed );
        waitKey(0);
    }


}

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color){
    for (int i = 0; i < kps.size(); i++) {
        cv::Point2f pti = kps[i].pt;
        cv::circle( markers, pti,1, color, -1, 8, 0 );
    }
}