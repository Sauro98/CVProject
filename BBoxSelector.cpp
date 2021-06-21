#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>

#include "Utils.hpp"

void selectROIs(cv::String name, cv::Mat& img, std::vector<cv::Rect>& ROIs)
{
	bool loop = true;
	while(loop)
	{
		cv::Rect ROI = cv::selectROI(name, img, false);
		if(ROI.x == 0 and ROI.y == 0 and ROI.width == 0 and ROI.height == 0)
		{
			loop = false;
		}
		else
		{
			cv::rectangle(img,ROI,cv::Scalar(0,0,255));
			ROIs.push_back(ROI);
		}
	}
}

void saveROIs(cv::String baseName, std::vector<cv::Rect>& ROIs)
{
	std::ofstream output;
	output.open(baseName+".txt");
	for(const auto& ROI: ROIs)
	{
		output<<ROI.x<<" "<<ROI.y<<" "<<ROI.width<<" "<<ROI.height<<"\n";
	}
	output.close();
}

bool filenamesMatch(const cv::String& f1, const cv::String& f2)
{
	return f1.size() == f2.size() and
			f1.substr(0,f1.size()-4) == f2.substr(0,f1.size()-4);
}

void drawROIs(cv::Mat& img, std::vector<cv::Rect>& ROIs)
{
	for(const auto& ROI: ROIs)
	{
		cv::rectangle(img,ROI,cv::Scalar(0,0,255));
	}
}

int main(int argc, char** argv)
{
	std::vector<cv::String> filenames;
	std::vector<cv::String> done;
    cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.jpg"), filenames);
    cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.txt"), done);
    cv::namedWindow("BBoxSelector");
	std::cout<<filenames.size()<<" images found\n";
	
	if(argc == 3 and strcmp(argv[2],"-S")==0)
	{
		for(const auto& fn: filenames)
		{
			std::vector<cv::Rect> ROIs;
			bool show = false;
			for(const auto& dn: done)
			{
				if(filenamesMatch(fn,dn))
				{
					show = true;
					loadROIs(dn,ROIs);
					break;
				}
			}
			
			if(show)
			{
				cv::Mat img = cv::imread(fn);
				drawROIs(img, ROIs);
				cv::imshow("BBoxSelector", img);
				cv::waitKey(0);
			}
			else
			{
				std::cout<<"ROIs not found: "<<fn<<"\n";
			}
		}
	}
	else if(argc == 3 and strcmp(argv[2],"-R")==0)
	{
		for(const auto& fn: filenames)
		{
			std::vector<cv::Rect> ROIs;
			cv::Mat img = cv::imread(fn);
			
			for(const auto& dn: done)
			{
				if(filenamesMatch(fn,dn))
				{
					loadROIs(dn, ROIs);
					drawROIs(img, ROIs);
					break;
				}
			}

			selectROIs("BBoxSelector", img, ROIs);
			std::cout<<"\n"<<ROIs.size()<<" ROIs selected\n\n\n";
			
			saveROIs(fn.substr(0,fn.size()-4), ROIs);
		}
	}
	else
	{
		for(const auto& fn: filenames)
		{
			bool skip = false;
			for(const auto& dn: done)
			{
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
				
				selectROIs("BBoxSelector", img, ROIs);
				std::cout<<"\n"<<ROIs.size()<<" ROIs selected\n\n\n";
				saveROIs(fn.substr(0,fn.size()-4), ROIs);
			}
		}
	}
    return 0;
}
