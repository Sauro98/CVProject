#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>

#include "Utils.hpp"

int main(int argc, char** argv)
{
	// Expects at least one argument which is the path to a directory containing the images on which the user
	// needs to select water. It exects jpg images and it will write one png mask file for each image in the same
	// folder, with the same name as the image it refers to.
	std::vector<cv::String> filenames;
	std::vector<cv::String> done;
	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.jpg"), filenames);
	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.png"), done);
	cv::namedWindow("WaterSelector");
	std::cout<<filenames.size()<<" images found\n\n";
	
	unsigned int brushSize = 20;

	if(true)//else
	{
		// for all images found in the directory
		for(const auto& fn: filenames)
		{
			bool skip = false;
			// check the png files found to see if there is one for this image
			for(const auto& dn: done)
			{
				// if there is then the water was already selected and this image can be skipped.
				if(filenamesMatch(fn,dn))
				{
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
				cv::Mat img = cv::imread(fn);
				cv::Mat temp_mask = cv::Mat(img.size(), CV_8UC3);

				// draw over the image
				brushSize = selectSea("WaterSelector", img, temp_mask, brushSize);
				
				cv::cvtColor(temp_mask, temp_mask, cv::COLOR_BGR2GRAY);
				cv::Mat mask;
				temp_mask.convertTo(mask, CV_8UC1);
				
				// After the user is done save the new mask
				// to a png file with the same name as the image

				saveMask(fn.substr(0,fn.size()-4), mask);
			}
		}
	}
	return 0;
}
