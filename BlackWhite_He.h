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

            //decomment this part if you want to first do hist_eq on HSV space and then the conversion on Grayscale

            /*
            Mat hsv_img, hchannel_eq,result_imgInV;
            vector<Mat> hsv_plane,hsvimg_eq;
            cvtColor(image,hsv_img,COLOR_BGR2HSV);
            split(hsv_img,hsv_plane);
            equalizeHist(hsv_plane[2],hchannel_eq);

            hsvimg_eq.push_back(hsv_plane[0]);
            hsvimg_eq.push_back(hsv_plane[1]);
            hsvimg_eq.push_back(hchannel_eq);
            merge(hsvimg_eq, result_imgInV);
            cvtColor(result_imgInV,image,COLOR_HSV2BGR);
            */


            //decomment this part if you want to first do hist_eq on RGB space and then the conversion on Grayscale

            /*
            Mat histchannel_eq ;
            vector<Mat> rgb_plane, rgbimg_eq;
            split(image,rgb_plane);
            for (int i = 0; i<3; i++){
                equalizeHist(rgb_plane[i],histchannel_eq);
                rgbimg_eq.push_back(histchannel_eq);

            }
            merge(rgbimg_eq, image);
            */

            cvtColor(image,gray_img,COLOR_BGR2GRAY);
            equalizeHist(gray_img, grayhist_equal); //this part should be commented if you are using hist_eq on RGB or HSV space
            return grayhist_equal;
        };

};



