#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <random>

#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core.hpp>

cv::Mat createBinMask(cv::Mat img, std::vector<cv::Rect> rects)
{
	cv::Mat col = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);
	for(const auto& rect: rects)
	{
		rectangle(col,rect,cv::Scalar(255),-1, cv::LINE_8);
	}
	return col;
}

void invertBinMask(cv::Mat& mask)
{
	cv::bitwise_not(mask,mask);
}

cv::Mat findFeatures(const cv::Mat& img, const cv::Mat& mask)
{
	std::vector<cv::KeyPoint> keypoints_img;
	
	cv::Ptr<cv::Feature2D> f2d = cv::SIFT::create();
	cv::Mat descriptors;
	f2d->detectAndCompute(img, mask, keypoints_img, descriptors);
	//cv::imshow("pippo",descriptors);
	//cv::waitKey(0);
	
	return descriptors;
}

void appendDescriptors(std::vector<std::vector<double>>& vect, cv::Mat descriptors)
{
	std::vector<double> line;
	for(unsigned int r=0; r<descriptors.rows; ++r)
	{
		line.clear();
		for(unsigned int c=0; c<descriptors.cols; ++c)
		{
			line.push_back(descriptors.at<float>(r,c));
		}
		vect.push_back(line);
	}
}

void saveDataset(
					const std::string& name,
					std::vector<std::vector<double>>& boats,
					std::vector<std::vector<double>>& other
				)
{
	for(unsigned int i=0; i<boats.size(); ++i)
	{
		boats[i].push_back(1);
		boats[i].push_back(0);
	}
	for(unsigned int i=0; i<other.size(); ++i)
	{
		other[i].push_back(0);
		other[i].push_back(1);
	}
	
	boats.insert(boats.end(), other.begin(), other.end());
	auto rng = std::default_random_engine(42);
	std::shuffle(std::begin(boats), std::end(boats), rng);
	
	std::ofstream output;
	output.open(name, std::ofstream::binary);
	std::ostream_iterator<char> outIt(output);
	for(unsigned int i=0; i<boats.size(); ++i)
	{
		const char* byte_s = (char*)&boats[i][0];
		const char* byte_e = (char*)&boats[i].back() + sizeof(double);
		std::copy(byte_s, byte_e, outIt);
	}
	output.close();
}

int main(int argc, char** argv)
{
	std::vector<std::vector<double>> boats;
	std::vector<std::vector<double>> none;
	
	std::vector<cv::String> filenames;
	std::vector<cv::String> rectFiles;
	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.jpg"), filenames);
	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.png"), filenames);
	cv::utils::fs::glob(cv::String(argv[1]), cv::String("*.txt"), rectFiles);
	std::cout<<rectFiles.size()<<" images found\n";
	
	unsigned int index = 0;
	std::cout<<index;
	for(const auto& fn: filenames)
	{
		std::cout<<"\r"<<index;
		++index;
		// check the txt files found to see if there is one for this image
		for(const auto& dn: rectFiles)
		{
			if(filenamesMatch(fn,dn))
			{
				std::vector<cv::Rect> ROIs;
				cv::Mat img = cv::imread(fn);
				loadROIs(dn, ROIs);
				
				cv::Mat mask = createBinMask(img,ROIs);
				cv::Mat tmp = findFeatures(img,mask);
				appendDescriptors(boats,tmp);
				
				invertBinMask(mask);
				tmp = findFeatures(img,mask);
				appendDescriptors(none,tmp);
				
				break;
			}
		}
	}
	std::cout<<"\n";

	std::cout<<boats.size()<<" boats keypoints\n"
				<<none.size()<<" other keypoints\n";
	
	saveDataset("dataset.txt",boats,none);
	return 0;
}