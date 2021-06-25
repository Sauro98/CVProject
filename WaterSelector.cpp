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
				std::vector<cv::KeyPoint> keypoints;
				std::vector<cv::KeyPoint> selected;
				
				// find keypoints
				
				findAllKeypoints(img, keypoints);
				std::cout<<keypoints.size()<<" keypoints found\n";
				
				// allow the user to select water keypoints over the image

				brushSize = selectKeypoints("WaterSelector", img, keypoints, selected, brushSize);
				std::cout<<"  of which "<<selected.size()<<" water and "<<keypoints.size()<<" non-water.\n\n";
				
				// transform keypoints to a mask of segements
				
				cv::Mat mask;
				createMask(img, keypoints, selected, mask);
				
				// show results
				showMask("WaterSelector", img, mask);

				// After the user is done save the new mask
				// to a png file with the same name as the image

				saveMask(fn.substr(0,fn.size()-4), mask);
			}
		}
	}
	return 0;
}
