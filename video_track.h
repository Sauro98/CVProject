#ifndef VIDEO_TRACK_H
#define VIDEO_TRACK_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <chrono>
#include "Utils.hpp"
#include "SiftMasked.h"
#include "kMeansClassifier.hpp"
#include "SegmentationHelper.hpp"
#include "DatasetHelper.hpp"


using namespace cv;
using namespace std;

class video_track {

    public:
    video_track() = default;

    Mat preproc(const Mat& frame)
    {
        Mat output;
        //some preprocessing for every frame
        //convert to grayscale
        cvtColor(frame, output, COLOR_BGR2GRAY);
        GaussianBlur(output, output, Size(21, 21), 0);

        return output;
    }

    VideoWriter prep_video(const VideoCapture& input_video,const String& filename)
    {
        int frame_width=   input_video.get(CAP_PROP_FRAME_WIDTH);
        int frame_height=  input_video.get(CAP_PROP_FRAME_HEIGHT);
        int codec = VideoWriter::fourcc('h', '2', '6', '4');
        int len_str = (int)filename.size()-(3+1);
        String outfile = filename.substr(0,len_str) + "_TrackVid.mp4";
        VideoWriter outvideo(outfile,codec,10,Size(frame_width,frame_height),true);

        return outvideo;
    }



    //compute difference between first frame and current frame, considering first frame is the static one
    vector<Rect> findBBMovement(const Mat& firstFrame, const Mat& currentFrame)
    {
        Mat frameDelta, thresh;
        vector<vector<Point>> mov_contours;
        vector<Rect> bboxes;

        absdiff(firstFrame, currentFrame, frameDelta);
        threshold(frameDelta, thresh, 25, 255, THRESH_BINARY);
        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        findContours(thresh, mov_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < mov_contours.size(); i++) {

            //if the area of the motion is below a constant then do nothing
            if (contourArea(mov_contours[i]) < 3000) {
                continue;
            }

            //else create a rect where there is motion
            bboxes.push_back(boundingRect(mov_contours[i]));

        }

        return bboxes;
    }


    int checkBoats(const Rect& ROI, const Mat& currentFrame, KMeansClassifier classifier, int &descSize, vector<Point2f> &keyboatsBBpoints)
    {

        Mat ROIMat = currentFrame(ROI);

        if(ROIMat.empty())
            return 0;

        vector<Point2f> keyBBpoints;
        vector<KeyPoint> keypframe;
        int label;
        Mat descrframe;

        SiftMasked featImg = SiftMasked();
        int num_boats = 0;

        Mat colframe = Mat::ones(ROIMat.size(),CV_8U);
        keypframe = featImg.findFeatures(ROIMat, colframe, descrframe);
        KeyPoint::convert(keypframe,keyBBpoints);
        vector<int> labels = classifier.predictBoatsBatch(descrframe,250);
        descSize = labels.size();

        for (int i = 0; i < labels.size(); i++) {
            label = labels[i];

            if (label == BOAT_LABEL) {
                num_boats++;

                keyboatsBBpoints.push_back(Point2f(keyBBpoints[i].x+ROI.x,keyBBpoints[i].y+ROI.y));
            }

        }

        return num_boats;
    }




    vector<Point2f> track(Mat& prevFrame, Mat& currentFrame, vector<Point2f> keypBB, Point2f& delta)
    {

        vector<Point2f> newKp;
        vector<unsigned char> status;
        vector<float> err;
        //TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

        calcOpticalFlowPyrLK(prevFrame,currentFrame, keypBB,newKp,status,err, cv::Size(21, 21),0);


        vector<Point2f> good_new;
        delta = Point2f (0,0);

        for(uint k = 0; k < keypBB.size(); k++)
        {
            // Select good points
            if(status[k] == 1 && err[k] < 20 && norm(newKp[k] - keypBB[k]) < 100000) {

                delta += newKp[k] - keypBB[k];
                good_new.push_back(newKp[k]);
            }
        }

        if(good_new.size())
            {delta /= (float)good_new.size();}

        return good_new;

    }

};


#endif
