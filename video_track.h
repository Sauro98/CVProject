

#ifndef VIDEO_TRACK_H
#define VIDEO_TRACK_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <unistd.h>
#include "Utils.hpp"
#include "SiftMasked.h"
#include "kMeansClassifier.hpp"
#include "SegmentationHelper.hpp"


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
        int len_str = (int)filename.size()-(4+1);
        String outfile = filename.substr(1,len_str) + "_TrackVid.mp4";
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
            if (contourArea(mov_contours[i]) < 5000) {
                continue;
            }

            //else create a rect where there is motion
            bboxes.push_back(boundingRect(mov_contours[i]));

        }

        return bboxes;
    }

  

    int checkBoats(const Rect& ROI, const Mat& currentFrame, KMeansClassifier classifier, int &descSize, vector<Point2f> &keyboatsBBpoints)
    {


        vector<Point2f> allkeyBBpoints;
        vector<KeyPoint> keypframe;
        vector<vector<double>> descVect;
        bool label;
        Mat descrframe;
        vector<Rect> singleROI;
        SiftMasked featImg = SiftMasked();
        int num_boats = 0;

        singleROI.push_back(ROI);
        Mat colframe = featImg.findBinMask(currentFrame, singleROI);
        keypframe = featImg.findFeatures(currentFrame, colframe, descrframe);
        KeyPoint::convert(keypframe,allkeyBBpoints);
        appendDescriptors(descVect, descrframe, 0, false);
        descSize = descVect.size();

        for (int i = 0; i < descVect.size(); i++) {
            label = classifier.predictLabel(descVect[i]);

            if (label == BOAT_LABEL) {
                num_boats++;
                keyboatsBBpoints.push_back(allkeyBBpoints[i]);

            }

        }

        return num_boats;
    }




    vector<Point2f> track(Mat& prevFrame, Mat& currentFrame, vector<Point2f> keypBB)
    {

        vector<Point2f> newKp;
        vector<unsigned char> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

        calcOpticalFlowPyrLK(prevFrame,currentFrame, keypBB,newKp,status,err, cv::Size(21, 21),0,criteria);

        vector<Point2f> good_new;
        for(uint k = 0; k < keypBB.size(); k++)
        {
            // Select good points
            if(status[k] == 1) {
                good_new.push_back(newKp[k]);
            }
        }
        return good_new;

    }



};


#endif
