//
// Created by Anna Zuccante on 20/06/2021.
//

#ifndef BLACK_WHITE_HE_H
#define BLACK_WHITE_HE_H

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "Utils.hpp"



using namespace cv;
using namespace std;

class BlackWhite_He{
    public:
        BlackWhite_He() = default;

        //converting an image to BGR to GRAYScale and doing the histogram equalization of it; if the bool
        //parameter is true than also a sharpening of the image is performed.
        Mat bgr_to_gray_HE(Mat image, bool shouldSharpen, int laplacianWeigth = 1){

            Mat gray_img, grayhist_equal;
            cvtColor(image,gray_img,COLOR_BGR2GRAY);
            equalizeHist(gray_img, grayhist_equal); 
            if(shouldSharpen)
                {
                    sharpen(grayhist_equal, grayhist_equal, laplacianWeigth);
                }
            return grayhist_equal;
        };
    
        //this function do hist_eq on RGB space and then the conversion on Grayscale; if the bool
        //parameter is true than also a sharpening of the image is performed.
        Mat bgr_HE_to_gray(Mat image, bool shouldSharpen){

            Mat histchannel_eq, gray_img, equalRGB_img ;
            vector<Mat> rgb_plane, rgbimg_eq;
            split(image,rgb_plane);
            for (int i = 0; i<3; i++){
                equalizeHist(rgb_plane[i],histchannel_eq);
                rgbimg_eq.push_back(histchannel_eq);

            }
            merge(rgbimg_eq, equalRGB_img);
            cvtColor(equalRGB_img,gray_img,COLOR_BGR2GRAY);
            if(shouldSharpen)
            {
                sharpen(gray_img, gray_img);
            }
            return gray_img;

        };
    
    
        //this function do hist_eq on HSV space and then the conversion on Grayscale; if the bool
        //parameter is true than also a sharpening of the image is performed.
        Mat hsv_HE_to_gray(Mat image, bool shouldSharpen){

            Mat hsv_img,result_imgInV;
            vector<Mat> hsv_plane,hsvimg_eq;
            cvtColor(image,hsv_img,COLOR_BGR2HSV);
            split(hsv_img,hsv_plane);
            equalizeHist(hsv_plane[2],result_imgInV); //the equalization should be done in the V channel,
                // and a conversion to gray is not necessary since this channel is actually the grayscale of the image
            if(shouldSharpen)
            {
                sharpen(result_imgInV, result_imgInV);
            }
            return result_imgInV;

        };

};

#endif



