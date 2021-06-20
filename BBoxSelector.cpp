#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>

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
			cv::String baseFn = fn.substr(0,fn.size()-4);
			std::vector<cv::Rect> ROIs;
			bool show = false;
			for(const auto& dn: done)
			{
				if(
					fn.size() == dn.size() and
					baseFn == dn.substr(0,dn.size()-4)
				)
				{
					show = true;
					std::ifstream input;
					input.open(dn);
					int x, y, w, h;
					while (input >> x >> y >> w >> h)
					{
						std::cout<<x<<" "<<y<<" "<<w<<" "<<h<<"\n";
						ROIs.push_back(cv::Rect(x,y,w,h));
					}
					std::cout<<"\n";
					input.close();
					break;
				}
			}
			
			if(show)
			{
				cv::Mat img = cv::imread(fn);
				for(const auto& ROI: ROIs)
				{
					cv::rectangle(img,ROI,cv::Scalar(0,0,255));
				}
				cv::imshow("BBoxSelector", img);
				cv::waitKey(0);
			}
			else
			{
				std::cout<<"ROIs not found: "<<fn<<"\n";
			}
		}
	}
	else
	{
		for(const auto& fn: filenames)
		{
			cv::String baseFn = fn.substr(0,fn.size()-4);
			bool skip = false;
			for(const auto& dn: done)
			{
				if(
					fn.size() == dn.size() and
					baseFn == dn.substr(0,dn.size()-4)
				)
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
				cv::selectROIs("BBoxSelector", img, ROIs, false);
				std::cout<<"\n"<<ROIs.size()<<" ROIs selected\n\n";
				
				std::ofstream output;
				output.open(baseFn+".txt");
				for(const auto& ROI: ROIs)
				{
					output<<ROI.x<<" "<<ROI.y<<" "<<ROI.width<<" "<<ROI.height<<"\n";
				}
				output.close();
			}
		}
	}
    return 0;
}