#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

typedef struct {
    std::vector<cv::KeyPoint> original;
    std::vector<cv::KeyPoint> selected;
} KeyPointsHolder;

const cv::Scalar RED(0,0,255);
const cv::Scalar GREEN(0,255,0);

static void onMouse( int event, int x, int y, int, void* userdata) {
    if( event != EVENT_LBUTTONDOWN )
        return;

    std::cout<<"click"<<std::endl;

    cv::Point2f seed = cv::Point2f(x,y);

    KeyPointsHolder* kps = (KeyPointsHolder*) userdata;

    int min_i = 0;
    double min_d = -1.;

    for(int i = 0; i < kps->original.size(); i++){
        double dist_i = cv::norm(seed -(kps->original)[i].pt);
        if (dist_i < min_d || min_d < 0.) {
            min_d = dist_i;
            min_i = i;
        }
    }

    cv::KeyPoint selected = kps->original[min_i];
    kps->selected.push_back(selected);
    kps->original.erase(kps->original.begin() + min_i);

    std::cout<<"click, min_i "<<min_i<<" or. size "<<kps->original.size()<<" sel. size "<<kps->selected.size()<<std::endl;

}


int main(int argc, char** argv) {


    if (argc != 2) {
        printf("usage: ./sea_selector image_path\n");
        return -1;
    }

    cv::Mat img = cv::imread(std::string(argv[1]));

    if (img.empty()) {
        cout << "Couldn't find or open the image!\n" << endl;
        return 1;
    }

    cv::namedWindow("sea_selector");
    cv::imshow("sea_selector",img);

    KeyPointsHolder kps;

    
    cv::Ptr<Feature2D> sift = cv::SIFT::create();
    sift->detect(img, kps.original);

    setMouseCallback( "sea_selector", onMouse, (void*) &kps );

    while(true) {

        cv::Mat kp_img;
        cv::drawKeypoints(img,kps.original,kp_img, RED);
        cv::drawKeypoints(kp_img,kps.selected,kp_img, GREEN);

        cv::imshow("sea_selector",kp_img);

        cv::waitKey(50);
    }

    return 0;
}
