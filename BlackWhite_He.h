//
// Created by Anna Zuccante on 20/06/2021.
//
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>



using namespace cv;
using namespace std;

class BlackWhite_He{
    public:
        BlackWhite_He() = default;

        //converting an image to BGR to GRAYScale and doing the histogram equalization of it
        Mat bgr_to_gray_HE(Mat image){

            Mat gray_img, grayhist_equal;
            cvtColor(image,gray_img,COLOR_BGR2GRAY);
            equalizeHist(gray_img, grayhist_equal); //this part should be commented if you are using hist_eq on RGB or HSV space
            return grayhist_equal;
        };
    
        //this function do hist_eq on RGB space and then the conversion on Grayscale
        Mat bgr_HE_to_gray(Mat image){

            Mat histchannel_eq, gray_img, equalRGB_img ;
            vector<Mat> rgb_plane, rgbimg_eq;
            split(image,rgb_plane);
            for (int i = 0; i<3; i++){
                equalizeHist(rgb_plane[i],histchannel_eq);
                rgbimg_eq.push_back(histchannel_eq);

            }
            merge(rgbimg_eq, equalRGB_img);
            cvtColor(equalRGB_img,gray_img,COLOR_BGR2GRAY);
            return gray_img;

        };
    
    
        //this function do hist_eq on HSV space and then the conversion on Grayscale
        Mat hsv_HE_to_gray(Mat image){

            Mat hsv_img,result_imgInV;
            vector<Mat> hsv_plane,hsvimg_eq;
            cvtColor(image,hsv_img,COLOR_BGR2HSV);
            split(hsv_img,hsv_plane);
            equalizeHist(hsv_plane[2],result_imgInV); //the equalization should be done in the V channel,
                // and a conversion to gray is not necessary since this channel is actually the grayscale of the image
            return result_imgInV;

        };

};



