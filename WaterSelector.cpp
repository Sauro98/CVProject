#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>

#include "Utils.hpp"
#include "SegmentationHelper.hpp"

int main(int argc, char** argv)
{

	if(argc < 3) {
		std::cout<<std::endl;
		std::cout<<"Usage:"<<std::endl;
		std::cout<<"./WaterSelector directory_path image_extension [-b] [-r]"<<std::endl;
		std::cout<<"  image_extension is of the type \"*.jpg\" / \"*.png\""<<std::endl;
		std::cout<<"  -b is optional and used to generate boat masks instead of sea masks"<<std::endl;
		std::cout<<"  -r is optional and used to modify existing masks"<<std::endl;
		return 1;
	}

	// -b can anly be at 3rd place
	bool boatSelector = (argc >= 4)? cv::String(argv[3]) == cv::String("-b") : false;

	// -r can be either at third or fourth place
	bool revise = (argc >= 4)? cv::String(argv[3]) == cv::String("-r") : false;
	if (argc >=5) {
		if (cv::String(argv[4]) == cv::String("-r") )
			revise = true;
	}
	// Expects at least one argument which is the path to a directory containing the images on which the user
	// needs to select water. It exects jpg images and it will write one png mask file for each image in the same
	// folder, with the same name as the image it refers to.
	std::vector<cv::String> filenames;
	std::vector<cv::String> done;
	cv::utils::fs::glob(cv::String(argv[1]), cv::String(argv[2]), filenames);
	if(boatSelector)
		cv::utils::fs::glob(cv::String(argv[1]), cv::String(BOAT_MASK_EXT), done);
	else
		cv::utils::fs::glob(cv::String(argv[1]), cv::String(SEA_MASK_EXT), done);

	removeMasksFromImagesFnames(filenames);
	
	cv::namedWindow("WaterSelector");

	if (revise)
		std::cout<<filenames.size()<<" images found\n\n";
	else
		std::cout<<filenames.size()-done.size()<<" images found\n\n";
	
	unsigned int brushSize = 20;

	if(true)//else
	{
		// for all images found in the directory
		for(const auto& fn: filenames)
		{
			bool skip = false;
			cv::Mat img = cv::imread(fn);
			cv::Mat temp_mask = cv::Mat(img.size(), CV_8UC3);

			// check the png files found to see if there is one for this image
			for(const auto& dn: done)
			{
				// if there is then the water was already selected and this image can be skipped or revised
				if(maskFilenamesMatch(dn,fn, boatSelector))
				{
					if(revise){
						std::cout<<"Found "<<dn<<std::endl;
						cv::Mat readMask = cv::imread(dn, cv::IMREAD_GRAYSCALE);
						cvtColor(readMask, temp_mask, cv::COLOR_GRAY2BGR);
					}
					else
						skip = true;

					break;
				}
			}
			
			if(skip)
			{
				std::cout<<"skipping "<<fn<<"\n";
			}
			else
			{
				// draw over the image
				brushSize = selectSea("WaterSelector", img, temp_mask, brushSize, boatSelector);
				
				cv::cvtColor(temp_mask, temp_mask, cv::COLOR_BGR2GRAY);
				cv::Mat mask;
				temp_mask.convertTo(mask, CV_8UC1);
				
				// After the user is done save the new mask
				// to a png file with the same name as the image

				saveMask(fn.substr(0,fn.size()-4), mask, boatSelector);
			}
		}
	}
	return 0;
}
