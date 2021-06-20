#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>


void binaryToBBoxes(const cv::Mat& img, std::vector<cv::Rect>& out)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	
	cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	
	for (const auto& cnt: contours)
	{
		out.push_back(cv::boundingRect(cnt));
    }
}

int main(int argc, char** argv)
{
    cv::namedWindow("TestBinaryToBBox");
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	std::vector<cv::Rect> rects;
	cv::imshow("TestBinaryToBBox", img);
	cv::waitKey(0);
	binaryToBBoxes(img,rects);
	
	cv::Mat col;
	cv::cvtColor(img, col, cv::COLOR_GRAY2BGR);
	
	for(const auto& rect: rects)
	{
		cv::rectangle(col,rect,cv::Scalar(0,0,255));
	}
	cv::imshow("TestBinaryToBBox", col);
	cv::waitKey(0);
	
    return 0;
}
