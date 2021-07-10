
#ifndef SIFT_MASKED_H
#define SIFT_MASKED_H

#include <iostream>
#include <fstream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

class SiftMasked {
public:

    //costructor
    SiftMasked() = default;;

    //if bounding boxes in the image have already been selected and their coordinates are written into a txt file,
    //we save them into a vector of rect and return this vector
    vector<Rect> checkFileBB(const String& bboxes, int &check)
    {
        fstream my_file;
        vector<Rect> out_rects;
        my_file.open(bboxes, ios::in);
        check = 0;

        if (!my_file) {
            cout << "No such file!\n";
            check = 1;
        }

        else {

            int x,y,width,height;
            
            while (my_file >> x >> y >> width >> height) {

                //cout << x << " " << y << " " << width << " " << height << endl;

                out_rects.push_back(cv::Rect(x, y, width, height));
            }

        }
        my_file.close();
        return out_rects;
    }

    //if we have a binary image we detect the contours and create the zone of interest with a bounding box
    void binaryToBBoxes(const cv::Mat &img, std::vector<cv::Rect> &out, bool ignore_internal) {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        if(ignore_internal)
            findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        else
            findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &cnt: contours) {
            out.push_back(boundingRect(cnt));
        }
    }

    //draw a binary mask based on rects
    Mat findBinMask(Mat img, vector<Rect>& rects)
    {
        Mat col = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);

        for(const auto& rect: rects)
        {
            rectangle(col,rect,cv::Scalar(255),-1, LINE_8);
        }

        return col;
    }

    //find and return features of an image given a binary mask
    vector<KeyPoint> findFeatures(const Mat &img, const Mat& mask, Mat &descriptor_img) {
        vector<KeyPoint> keypoints_img;

        Ptr<Feature2D> f2d = SIFT::create();
        if(not mask.empty())
        {
            f2d->detectAndCompute(img, mask, keypoints_img, descriptor_img);
        }
        else
        {
            f2d->detectAndCompute(img, noArray(), keypoints_img, descriptor_img);
        }
        

        return keypoints_img;

    }

    void findDescriptors(const Mat &img, std::vector<KeyPoint> & keypoints, Mat &descriptor_img) {
        vector<KeyPoint> keypoints_img;

        Ptr<Feature2D> f2d = SIFT::create();
        f2d->compute(img, keypoints, descriptor_img);
    }

};

#endif
