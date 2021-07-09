#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "BlackWhite_He.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
       
     bool sharp = false;
       
     if (argc != 3 ) {
            if(argc == 4)
            {
                   sharp = strcmp(argv[3],"-sh")==0;
            }
            else if(argc == 5)
            {
                   sharp = strcmp(argv[4],"-sh")==0;
            }
            else
            {
                    printf("usage: ./filename <Image_Directory> <Format_of_images: ex. *.bmp or *.png> <Optional command (-hsv or -rgb or/and -sh)>\n");
                    return -1;
            }
        }

    std::vector<cv::String> filenames;
    std::vector<cv::String> output_names;
    std::vector<cv::Mat> images;
    Mat result_img;
    string optionalcmd; //the 4th optional string on the command line
    

    cv::utils::fs::glob( String(argv[1]), String(argv[2]), filenames);

    for (const auto &fn: filenames) {
        cv::Mat img = cv::imread(fn);

        if (img.empty()) {
            cout << "Couldn't find or open the image!\n" << endl;
            break;
        }

        //create a new object of the class BlackWhite_He
        BlackWhite_He preproc_img = BlackWhite_He();
            
        //calling functions of the class
        if((argc == 4 || argc == 5) and strcmp(argv[3],"-rgb")==0)
        {
            if(sharp)
            {
                optionalcmd = "RGBhe_sh_";
            }
            else {
                optionalcmd = "RGBhe_";
            }
            result_img = preproc_img.bgr_HE_to_gray(img, sharp);
            images.push_back(result_img);
            int beg_str = String(argv[1]).size() + 1;
            int len_str = (int)fn.size()-(4+beg_str);
            String name =  optionalcmd + "BnW_" + fn.substr(beg_str,len_str);
            output_names.push_back(name);
        }

        else if((argc == 4 || argc == 5) and strcmp(argv[3],"-hsv")==0)
        {
            if(sharp)
            {
                optionalcmd = "HSVhe_sh_";
            }
            else {
                optionalcmd = "HSVhe_";
            }
            result_img = preproc_img.hsv_HE_to_gray(img,sharp);
            images.push_back(result_img);
            int beg_str = String(argv[1]).size() + 1;
            int len_str = (int)fn.size()-(4+beg_str);
            String name =  optionalcmd + "BnW_" + fn.substr(beg_str,len_str);
            output_names.push_back(name);
        }

        else
        {
            if(sharp)
            {
                optionalcmd = "Sh_";
            }
            else {
                optionalcmd = "";
            }
            
            result_img = preproc_img.bgr_to_gray_HE(img, sharp);
            images.push_back(result_img);
            int beg_str = String(argv[1]).size() + 1;
            int len_str = (int)fn.size()-(4+beg_str);
            String name =  optionalcmd + "BnW_" + fn.substr(beg_str,len_str);
            output_names.push_back(name);
        }

    }
        
    //creation of a folder to contain the new images    
    string newfolder = String(argv[1])  + "/"+ optionalcmd + "BnW";
    std::cout<<newfolder<<std::endl;
    if(!cv::utils::fs::createDirectory(newfolder)){
        std::cout<<"Failed to create directory"<<std::endl;
    }
    
    //writing the images resulting on the new folder
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
