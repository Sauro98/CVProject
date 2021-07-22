#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <unistd.h>

#include "Utils.hpp"
#include "DatasetHelper.hpp"
#include "video_track.h"


using namespace cv;
using namespace std;
 
int main(int argc, char **argv) {

    Mat frame, gray, oldFrame, grayframe, firstFrame, grayoldFrame;
    vector<vector<Point2f>> oldkp;

    if (argc != 3 )
    {
        printf("usage: ./filename <Video> <path_directory_classifier>\n");
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

    //convert to grayscale and set the first frame
    firstFrame = trck_vd.preproc(frame);

    //preparing output video
    VideoWriter outvideo = trck_vd.prep_video(video,filename);

    oldFrame = firstFrame;


    while(video.read(frame)) {

        vector<Rect> outBB, boats, oldoutBB;
        vector<Point2f> kp;
        vector<vector<Point2f>> tempKeyp;
        int num_boats;
     
        //some preprocessing for every frame
        gray = trck_vd.preproc(frame);

        outBB = trck_vd.findBBMovement(firstFrame,gray);

       

            for (int i = 0; i < outBB.size(); i++) {
            int sizeROI;
            int num_keyponBB = 0;
            
            //cout<<oldkp.size()<<endl;

            if(!oldkp.empty()){

                cvtColor(frame, grayframe, COLOR_BGR2GRAY);
                cvtColor(oldFrame, grayoldFrame, COLOR_BGR2GRAY);

                for(int k = 0; k < oldkp.size(); k++) {
                    kp = trck_vd.track(oldFrame, grayframe, oldkp[k]);
                    //cout << "num_keyponBB " << num_keyponBB << endl;
                    for (int j = 0; j < kp.size(); j++) {
                        if (outBB[i].contains(kp[j])) {
                            num_keyponBB++;

                        }
                    }
                }
            }
             
            if(num_keyponBB < 50) {
                num_boats = trck_vd.checkBoats(outBB[i], frame, classifier, sizeROI,kp);

                if (num_boats > (sizeROI / 6)) {
                    boats.push_back(outBB[i]);
                    tempKeyp.push_back(kp);
                }
                else
                    {
                    kp.clear();
                }
                //cout << "sizeROI " << i << " " << sizeROI << endl;
                //cout << "num_boats " << i << " " << num_boats << endl;

            }
            else
            {
                boats.push_back(outBB[i]);
            }
             
         }
        oldkp = tempKeyp;

            //finally draw the rects found
            drawROIs(frame, boats);

        //show on a window the video frame by frame
        imshow("Camera", frame);
        outvideo.write(frame);

        
        if(waitKey(1) == 27){
            //exit if ESC is pressed
            break;
        }
        
        oldFrame = frame;
    
    }

    return 0;
}
