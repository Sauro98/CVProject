
#include <iostream>
#include <fstream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "SiftMasked.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    vector<cv::Rect> rects;
    vector<KeyPoint> keypoints, keypoints2;
    Mat col, col2, descriptor, descriptor2, res_img, res_img2, maskSea;
    int check;
    bool secMask = false;


    SiftMasked featImg = SiftMasked();
    namedWindow("Original Image");

    Mat img = imread(argv[1],IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Couldn't find or open the image!\n";
    }

    imshow("Original Image", img);
    waitKey(0);

    if(!argv[2]){
        featImg.binaryToBBoxes(img,rects, false);
    }

    else{
        rects = featImg.checkFileBB(argv[2],check);

        if(check)
        {
            return -1;
        }
        
        if(argc == 4)
        {
            secMask = true;
            maskSea = imread(argv[3],IMREAD_GRAYSCALE);
        }

    }

    col = featImg.findBinMask(img,rects);
    
    if(secMask)
    {
        imshow("Second Mask", maskSea);
        waitKey(0);
        keypoints2 = featImg.findFeatures(img, maskSea,descriptor2, res_img2);
        drawKeypoints(img, keypoints2, res_img2);
        imshow("Second Keypoint image", res_img2);
        waitKey(0);

        col = featImg.findIntersectionMask(col, maskSea);
    }

    imshow("First Mask", col);
    waitKey(0);

    keypoints = featImg.findFeatures(img, col,descriptor, res_img);
    drawKeypoints(img, keypoints, res_img);
    imshow("First Keypoint image", res_img);
    waitKey(0);





    return 0;
}

