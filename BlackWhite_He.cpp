#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "BlackWhite_He.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv) {


        if (argc != 3) {
            printf("usage: ./filename <Image_Directory> <Format_of_images: ex. *.bmp or *.png or *.jpg>\n");
            return -1;
        }

    std::vector<cv::String> filenames;
    std::vector<cv::String> output_names;
    std::vector<cv::Mat> images;
    string newfolder = String(argv[1]) + "/BnW";
    std::cout<<newfolder<<std::endl;
    if(!cv::utils::fs::createDirectory(newfolder)){
        std::cout<<"Failed to create directory"<<std::endl;
    }

    cv::utils::fs::glob( String(argv[1]), String(argv[2]), filenames);

    for (const auto &fn: filenames) {
        cv::Mat img = cv::imread(fn);

        if (img.empty()) {
            cout << "Couldn't find or open the image!\n" << endl;
            break;
        }

        //create a new object of the class BlackWhite_He
        BlackWhite_He preproc_img = BlackWhite_He();
        Mat result_img = preproc_img.bgr_to_gray_HE(img);

        images.push_back(result_img);
        int beg_str = String(argv[1]).size() + 1;
        int len_str = (int)fn.size()-(4+beg_str);
        String name =  "BnW_" + fn.substr(beg_str,len_str);
        output_names.push_back(name);

    }
    for(int i = 0; i<output_names.size(); i++)
    {
        stringstream ss;
        ss<<newfolder<<"/"<<output_names[i]<<String(argv[2]).substr(1);
        string fullPath = ss.str();
        cout<<fullPath<<endl;
        ss.str("");

        if(!imwrite(fullPath, images[i])){
            std::cout<<"Failed to write image"<<std::endl;
        }
    }

    return 0;
}
