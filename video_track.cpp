#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <unistd.h>

#include "Utils.hpp"
#include "DatasetHelper.hpp"
#include "video_track.h"



using namespace cv;
using namespace std;


struct traceble
        {vector<Point2f> tracKeyp;
            Rect tracRect;
            bool isTracked = false;
            int lifeTime = 0;
        };


int main(int argc, char **argv) {

    Mat frame, gray, oldFrame, grayframe(Size(0,0),CV_8UC1), grayoldFrame(Size(0,0),CV_8UC1), grayold;




    if (argc != 3 )
    {
        printf("usage: ./filename <Video> <path_directory>\n");
        return -1;
    }
    String filename = argv[1];

    video_track trck_vd = video_track();
    VideoCapture video(filename); //open video given by command line


    if (!video.isOpened()) {
        cout << "Couldn't find or open the video!\n" << endl;
        return -1;
    }

    String input_directoryKM = argv[2];

    //classifier for boat detection
    KMeansClassifier classifier(0.9);
    classifier.load(input_directoryKM,true);


    sleep(3);
    video.read(frame);



    //preparing output video
    VideoWriter outvideo = trck_vd.prep_video(video,filename);

    oldFrame = frame.clone();

    vector<traceble> tracVect;
    int countFrame = 0;
    int limitFrame = 3;


    while(video.read(frame)) {

        vector<Rect> outBB;

        int num_boats;


        //some preprocessing for every frame
        gray = trck_vd.preproc(frame);
        grayold = trck_vd.preproc(oldFrame);

        if(countFrame>limitFrame)
            outBB = trck_vd.findBBMovement(grayold,gray);

        if(!tracVect.empty()) {


            cvtColor(oldFrame, grayoldFrame, COLOR_BGR2GRAY);
            cvtColor(frame, grayframe, COLOR_BGR2GRAY);

            for (int k = 0; k < tracVect.size(); k++) {

                Point2f delta;

                tracVect[k].tracKeyp = trck_vd.track(grayoldFrame, grayframe, tracVect[k].tracKeyp, delta);
                tracVect[k].isTracked = true;
               // tracVect[k].tracRect = boundingRect(tracVect[k].tracKeyp);

                tracVect[k].tracRect.x = tracVect[k].tracRect.x + delta.x;
                tracVect[k].tracRect.y = tracVect[k].tracRect.y + delta.y;

                //tracVect[k].lifeTime++;
                //if(tracVect[k].tracRect.x <= 1 || tracVect[k].tracRect.y <= 1 || tracVect[k].tracRect.x + tracVect[k].tracRect.width >= frame.cols - 1 || tracVect[k].tracRect.y + tracVect[k].tracRect.height >= frame.rows - 1 )
                  //  tracVect[k].lifeTime = 10;
                if(norm(delta) < 2)
                    tracVect[k].lifeTime = 10;



            }

            tracVect.erase(std::remove_if(tracVect.begin(), tracVect.end(), [](const traceble& f) {
                return  (f.tracKeyp.size() < 3) || f.lifeTime >= 10;
            }), tracVect.end());
        }

        drawROIs(frame, outBB);
        for (int i = 0; i < outBB.size(); i++) {
            int sizeROI;


            double maxIoU = -1;
            int maxIndex;
            double minArea;

            for(int j = 0; j < tracVect.size() ; j++)
            {
                minArea = outBB[i].area();
                if(minArea > tracVect[j].tracRect.area())
                    minArea = tracVect[j].tracRect.area();

                double IoU = ((outBB[i] & tracVect[j].tracRect).area())/minArea;
                if(IoU > maxIoU)
                { maxIoU = IoU;
                    maxIndex = j;}


            }

            cout << "maxIoU" << maxIoU << endl;


            if(maxIoU < 0.5) {

                vector<Point2f> kp;
                num_boats = trck_vd.checkBoats(outBB[i], frame, classifier, sizeROI,kp);

                if (num_boats > (sizeROI / 6)) {
                    traceble temp ;
                    temp.tracKeyp = kp;
                    temp.tracRect = outBB[i];
                    tracVect.push_back(temp);
                }


            }

            else
            {
                tracVect[maxIndex].tracRect |= outBB[i];
            }



        }


        //finally draw the rects found
        for(const auto &tr : tracVect)
        {
            if(tr.isTracked)
                rectangle(frame,tr.tracRect,Scalar(0,255,0),2);
            //else
                //rectangle(frame,tr.tracRect,Scalar(255,0,0),2);



        }




        //show on a window the video frame by frame
        imshow("Camera", frame);
        outvideo.write(frame);



        if(waitKey(1) == 27){
            //exit if ESC is pressed
            break;
        }



        if(countFrame > limitFrame)
        {   oldFrame = frame.clone();
            countFrame = 0;}

        countFrame++;



    }

    return 0;
}