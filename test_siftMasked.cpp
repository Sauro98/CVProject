
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
    vector<KeyPoint> keypoints;
    Mat col, descriptor, res_img;
    int check;


    SiftMasked featImg = SiftMasked();
    namedWindow("Original Image");

    Mat img = imread(argv[1],IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Couldn't find or open the image!\n";
    }

    imshow("Original Image", img);
    waitKey(0);

    if(!argv[2]){
        featImg.binaryToBBoxes(img,rects);
    }

    else{
        rects = featImg.checkFileBB(argv[2],check);

        if(check)
        {
            return -1;
        }

    }


    col = featImg.findBinMask(img,rects);

    imshow("Mask", col);
    waitKey(0);

    keypoints = featImg.findFeatures(img, col,descriptor);
    drawKeypoints(img, keypoints, res_img);
    imshow("Keypoint image", res_img);
    waitKey(0);





    return 0;
}

