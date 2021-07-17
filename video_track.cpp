#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <unistd.h>

#include "Utils.hpp"
//#include "Utils.cpp"


using namespace cv;
using namespace std;
 
int main(int argc, char **argv) {

    Mat frame, gray, frameDelta, thresh, firstFrame;
    vector<vector<Point>> contours;

    if (argc != 2 )
    {
        printf("usage: ./filename <Video>\n");
        return -1;
    }

    VideoCapture video(argv[1]); //open video given by command line

    if (!video.isOpened()) {
        cout << "Couldn't find or open the video!\n" << endl;
        return -1;
    }

    sleep(3);
    video.read(frame);

    //convert to grayscale and set the first frame
    cvtColor(frame, firstFrame, COLOR_BGR2GRAY);
    GaussianBlur(firstFrame, firstFrame, Size(21, 21), 0);


    while(video.read(frame)) {

        vector<Rect> out;

        //some preprocessing for every frame
        //convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        //compute difference between first frame and current frame, considering first frame is the static one
        absdiff(firstFrame, gray, frameDelta);
        threshold(frameDelta, thresh, 25, 255, THRESH_BINARY);
        
        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

       

            for (int i = 0; i < contours.size(); i++) {

                //if the area of the motion is below a constant then do nothing
                if (contourArea(contours[i]) < 5000) {
                    continue;
                }

                //else create a rect where there is motion
                out.push_back(boundingRect(contours[i]));

            }

            //finally draw the rects found
            drawROIs(frame, out);

        //show on a window the video frame by frame
        imshow("Camera", frame);

        
        if(waitKey(1) == 27){
            //exit if ESC is pressed
            break;
        }
    
    }

    return 0;
}