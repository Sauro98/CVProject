/*#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;



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

cv::Mat sharpen_img(cv::Mat& input){
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    Mat imgLaplacian;
    filter2D(input, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    input.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    return imgResult;
}

cv::Mat getMarkers(cv::Mat& img, KeyPointsHolder& kps) {
    cv::Mat prepared_img = Mat::zeros(img.size(), CV_8UC1);
    for(const auto& kp: kps.selected){
        for(int x = 0;  x < 10; x++){
            for(int y = 0;  y < 10; y++){
                prepared_img.at<cv::Scalar>(kp.pt + cv::Point2f(x,y)) = cv::Scalar(255);
            }
        }
    }

    vector<vector<Point> > contours;
    cv::findContours(prepared_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    cv::Mat markers = Mat::zeros(img.size(), CV_32SC1);
    for(int i = 0; i < contours.size(); i++){
        cv::drawContours(markers, contours, i, cv::Scalar(50), -1);
    }
    //prepared_img.convertTo(markers, 32);
    return prepared_img;
}

cv::Mat getSegmentation(cv::Mat& markers) {
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC1);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0)
            {
                dst.at<uchar>(i,j) = (uchar)index;
            }
        }
    }
    return dst;
}

int main(int argc, char** argv) {


    if (argc != 2) {
        printf("usage: ./sea_selector image_path\n");
        return -1;
    }

    cv::Mat img = cv::imread(std::string(argv[1]));
    img = sharpen_img(img);

    if (img.empty()) {
        cout << "Couldn't find or open the image!\n" << endl;
        return 1;
    }

    cv::namedWindow("sea_selector");
    cv::namedWindow("markers");
    cv::namedWindow("segm");
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

        cv::Mat markers = getMarkers(img, kps);

        Mat markers8u;
        markers.convertTo(markers8u, CV_8U, 32);
        imshow("markers", markers8u);

        cv::watershed(img, markers);
        cv::Mat segm = getSegmentation(markers);

        
        cv::imshow("segm",segm);

        cv::waitKey(0);
    }

    return 0;
}*/
  
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <cstdio>
#include <algorithm>

using namespace cv;
using namespace std;

const cv::Scalar RED(0,0,255);
const cv::Scalar GREEN(0,255,0);

Mat markerMask, img;
Point prevPt(-1, -1);


typedef struct {
    std::vector<cv::KeyPoint> original;
    std::vector<cv::KeyPoint> selected;
} KeyPointsHolder;

static void onMouse( int event, int x, int y, int flags, void* userdata)
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        //if( prevPt.x < 0 )
        //    prevPt = pt;
        //line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        //line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        //prevPt = pt;
        KeyPointsHolder* kps = (KeyPointsHolder*) userdata;

        int min_i = 0;
        std::vector<int> selected_idxs;

        for(int i = 0; i < kps->original.size(); i++){
            double dist_i = cv::norm((cv::Point2f)pt -(kps->original)[i].pt);
            if (dist_i < 20) {
                selected_idxs.push_back(i);
            }
        }

        for(const auto& i: selected_idxs){
            cv::KeyPoint selected = kps->original[i];
            kps->selected.push_back(selected);
        }
        for(int j = selected_idxs.size() - 1; j >=0; j--){ 
            kps->original[selected_idxs[j]] = kps->original.back();
            kps->original.pop_back();
        }
        
        cv::drawKeypoints(img,kps->selected,img, GREEN);

        imshow("image", img);
    }
}


static void help()
{
    cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
            "Usage:\n"
            "./watershed [image_name -- default is fruits.jpg]\n" << endl;


    cout << "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tw or SPACE - run watershed segmentation algorithm\n"
        "\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
        "\t  (before that, roughly outline several markers on the image)\n";
}


cv::Mat sharpen_img(cv::Mat& input){
    Mat imgLaplacian, sharp;
    cv::Laplacian(input, imgLaplacian, CV_32F);
    cv::GaussianBlur(input, sharp,cv::Size(5,5),0);
    cv::Laplacian(sharp, imgLaplacian, CV_32F);
    sharp.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC1);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC1);
    imgResult.copyTo(input);
    return imgLaplacian;
}

cv::Mat drawMarkers(KeyPointsHolder& kps, cv::Size img_size) {
    cv::Mat markers = cv::Mat::zeros(img_size, CV_8U);
    for (int i = 0; i < kps.original.size() - 1; i++) {
            cv::Point2f pti = kps.original[i].pt;
        cv::circle( markers, pti,1, Scalar::all(255), -1, 8, 0 );
        /*for (int j = i; j < kps.original.size(); j++) {
            cv::Point2f ptj = kps.original[j].pt;
            double dist = cv::norm(pti - ptj);
            //if (dist < 50. && dist > 5.) {
            //    cv::line( markers, pti,ptj, Scalar::all(1.), 5, 8, 0 );
            //}
        }*/
    }

    for (int i = 0; i < kps.selected.size() - 1; i++) {
        cv::Point2f pti = kps.selected[i].pt;
        cv::circle( markers, pti,1, Scalar::all(255), -1, 8, 0 );
        /*for (int j = i; j < kps.selected.size(); j++) {
            cv::Point2f ptj = kps.selected[j].pt;
            double dist = cv::norm(pti - ptj);
            /*if (dist < 50.) {
                cv::line( markers, pti,ptj, Scalar::all(255), 5, 8, 0 );
            }
        }*/
    }

    return markers;
}

std::vector<int> get_sea_indexes(KeyPointsHolder& kps, cv::Mat& markers){
    std::vector<int> sea_indexes;
    for (const auto& kp: kps.selected){
        int index = markers.at<int>(kp.pt);
        if(std::find(sea_indexes.begin(), sea_indexes.end(), index) == sea_indexes.end()) {
            /* sea_indexes does not contain index */
            sea_indexes.push_back(index);
        } 
    }
    return sea_indexes;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main( int argc, char** argv )
{
    char* filename = argv[1];
    Mat img0 = imread(filename), imgGray;
    if( img0.empty() )
    {
        cout << "Couldn'g open image " << filename << ". Usage: watershed <image_name>\n";
        return 0;
    }
    help();
    std::cout<<img0.channels()<<std::endl;
    namedWindow( "image", 1 );
    cvtColor(img0, markerMask, COLOR_BGR2GRAY);
    markerMask.copyTo(imgGray);
    //cvtColor(img0, img0, COLOR_BGR2GRAY);
    Mat edges = sharpen_img(imgGray);
    imgGray.copyTo(img);
    Canny(imgGray, edges, 20, 100, 3);
    equalizeHist(edges, edges);
    threshold(edges, edges, 10, 255, THRESH_BINARY);
    cvtColor(edges, edges, COLOR_GRAY2BGR);
    namedWindow( "edges");
    imshow( "edges", edges );
    
    /*threshold(edges, edges, 10, 255, THRESH_BINARY);
    edges = ~edges;
    //namedWindow( "thr");
    //imshow( "thr", edges );
    // Perform the distance transform algorithm
    Mat dist;
    cv::distanceTransform(edges, dist, DIST_L2, 5, CV_32FC1);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    dist = dist;
    namedWindow( "dst");
    imshow( "dst", dist );
    threshold(dist, dist, 0.3, 1.0, THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(5, 5, CV_8U);
    erode(dist, dist, kernel1);
    namedWindow( "pks");
    imshow( "pks", dist );*/
    
    KeyPointsHolder kps;
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
    sift->detect(img0, kps.original);
    markerMask = Scalar::all(0);
    cv::drawKeypoints(img,kps.original,img, RED);
    cv::drawKeypoints(img,kps.selected,img, GREEN);
    imshow( "image", img );
    setMouseCallback( "image", onMouse, &kps );

    for(;;)
    {
        int c = waitKey(0);

        if( (char)c == 27 )
            break;

        if( (char)c == 'r' )
        {
            markerMask = Scalar::all(0);
            imgGray.copyTo(img);
            sift->detect(img, kps.original);
            kps.selected.clear();
            cv::drawKeypoints(img,kps.original,img, RED);
            cv::drawKeypoints(img,kps.selected,img, GREEN);
            imshow( "image", img );
        }

        if( (char)c == 'w' || (char)c == ' ' )
        {


            int i, j, compCount = 0;
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            //std::cout<<"mmtype "<<type2str(markerMask.type())<<std::endl;
            //std::cout<<"disttype "<<type2str(dist.type())<<std::endl;
            markerMask = drawMarkers(kps, img0.size())/* dist.convertTo(markerMask, CV_8UC1)*/;
            Mat cvt;

            //markerMask.convertTo(cvt, CV_8U);
            //namedWindow( "mmmask");
            //imshow( "mmmask", markerMask );
            
            findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            if( contours.empty() )
                continue;
            Mat markers(markerMask.size(), CV_32S);
            markers = Scalar::all(0);
            int idx = 0;
            for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
                drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

            

            if( compCount == 0 )
                continue;

            vector<Vec3b> colorTab;
            for( i = 0; i < compCount; i++ )
            {
                int b = theRNG().uniform(0, 255);
                int g = theRNG().uniform(0, 255);
                int r = theRNG().uniform(0, 0);

                colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            double t = (double)getTickCount();
            watershed( img0, markers );
            t = (double)getTickCount() - t;
            printf( "execution time = %gms\n", t*1000./getTickFrequency() );

            //markers.convertTo(cvt, CV_8U);
            //namedWindow( "mmask");
            //imshow( "mmask", cvt );

            Mat wshed(markers.size(), CV_8UC3);

            auto sea_indexes = get_sea_indexes(kps, markers);

            // paint the watershed image
            for( i = 0; i < markers.rows; i++ )
                for( j = 0; j < markers.cols; j++ )
                {
                    int index = markers.at<int>(i,j);
                    if(std::find(sea_indexes.begin(), sea_indexes.end(), index) != sea_indexes.end()) {
                        wshed.at<Vec3b>(i,j) = Vec3b(0,0,255);
                    } 
                    else 
                    if( index == -1 )
                        wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
                    else if( index <= 0 || index > compCount )
                        wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
                    else
                        wshed.at<Vec3b>(i,j) = colorTab[index - 1];
                }
            Mat bg;
            cvtColor(imgGray, bg, COLOR_GRAY2BGR);
            wshed = wshed*0.5 + bg*0.5;
            imshow( "watershed transform", wshed );
            waitKey(0);
        }
    }

    return 0;
}
