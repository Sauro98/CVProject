#include "Utils.hpp"
#include <fstream>

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