#include <iostream>
#include <fstream>

#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>

int main(int argc, char** argv)
{
	// Expects at least one argument which is the path to a directory containing the images on which the user
	// needs to select bounding boxes. It exects jpg images and it will write one txt file for each image in the same
	// folder, with the same name as the image it refers to.
	std::vector<cv::String> filenames;
	std::vector<cv::String> done;
    cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.jpg"), filenames);
    cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.txt"), done);
    cv::namedWindow("BBoxSelector");
	std::cout<<filenames.size()<<" images found\n";
	
	// -S option: allows the user to see the bounding boxes they selected
	// over the original images. Images will be shown one at a time and 
	// any input will make the program go to the next image. No interaction with 
	// the images is possible.
	if(argc == 3 and strcmp(argv[2],"-S")==0)
	{
		// for all images found in the directory
		for(const auto& fn: filenames)
		{
			std::vector<cv::Rect> ROIs;
			bool show = false;
			// check the txt files found to see if there is one for this image
			for(const auto& dn: done)
			{
				// if there is then load the ROIs from the txt and setup the image to
				// be dispalyed
				if(filenamesMatch(fn,dn))
				{
					show = true;
					loadROIs(dn,ROIs);
					break;
				}
			}
			
			// if the corresponding txt file was found, then 
			// draw the loaded ROIs over the image and wait for user input before going to the next image.
			if(show)
			{
				cv::Mat img = cv::imread(fn);
				drawROIs(img, ROIs);
				cv::imshow("BBoxSelector", img);
				cv::waitKey(0);
			}
			// otherwise just go to the next image in the folder
			else
			{
				std::cout<<"ROIs not found: "<<fn<<"\n";
			}
		}
	}
	// -R option: allows the user to see the bounding boxes they selected
	// over the original images and draw additional boxes. Images will be shown one at a time, 
	// and navigation is performed as in the general case. The difference with the general case is that 
	// it will also show images for which the bounding boxes were already selected.
	else if(argc == 3 and strcmp(argv[2],"-R")==0)
	{
		// for all images found in the directory
		for(const auto& fn: filenames)
		{
			std::vector<cv::Rect> ROIs;
			cv::Mat img = cv::imread(fn);
			// check the txt files found to see if there is one for this image
			for(const auto& dn: done)
			{
				// if there is then load the ROIs from the txt and draw them over the image.
				if(filenamesMatch(fn,dn))
				{
					loadROIs(dn, ROIs);
					drawROIs(img, ROIs);
					break;
				}
			}

			// Regardless of whether any previous bounding boxes were found, allow the user to
			// draw more bounding boxes over the image

			selectROIs("BBoxSelector", img, ROIs);
			std::cout<<"\n"<<ROIs.size()<<" ROIs selected\n\n\n";
			
			// After the user is done save the rois (old and new)
			// to a txt file with the same name as the image

			saveROIs(fn.substr(0,fn.size()-4), ROIs);
		}
	}
	// General case: Images are shown one at a time and the user is allowed to draw bounding boxes over them.
	// Once the user is satisfied with the drawn bounding box, they can press enter or space to confirm it. When
	// the user is done selecting bounding boxes for the current image, they can press enter or space without drawing
	// any box to go to the next image. Images for which bounding boxes already exist will be skipped, propting the user
	// with the first image they haven't seen yet.
	else
	{
		// for all images found in the directory
		for(const auto& fn: filenames)
		{
			bool skip = false;
			// check the txt files found to see if there is one for this image
			for(const auto& dn: done)
			{
				// if there is then the ROIs were already selected and this image can be skipped.
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
				std::vector<cv::Rect> ROIs;
				
				//allow the user to draw bounding boxes over the image

				selectROIs("BBoxSelector", img, ROIs);
				std::cout<<"\n"<<ROIs.size()<<" ROIs selected\n\n\n";

				// After the user is done save the rois (old and new)
				// to a txt file with the same name as the image

				saveROIs(fn.substr(0,fn.size()-4), ROIs);
			}
		}
	}
    return 0;
}
