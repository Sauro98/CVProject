#include "Utils.hpp"

void loadROIs(std::string name, std::vector<cv::Rect>& ROIs)
{
	std::ifstream input;
	input.open(name);
	int x, y, w, h;
	while (input >> x >> y >> w >> h)
	{
		ROIs.push_back(cv::Rect(x,y,w,h));
	}
	input.close();
}


void selectROIs(cv::String name, cv::Mat& img, std::vector<cv::Rect>& ROIs)
{
	bool loop = true;
	// Loop until the user presses enter (or space) without selecting any ROIs
	// (or until they press c as indicated in opencv's docs)
	while(loop)
	{
		cv::Rect ROI = cv::selectROI(name, img, false);
		// stop looping if no ROI is selected
		if(ROI.x == 0 and ROI.y == 0 and ROI.width == 0 and ROI.height == 0)
		{
			loop = false;
		}
		// otherwise draw the ROI just selected in red over the image, add the ROI to the output vector
		// and loop again to select the next ROI
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