#ifndef __UTILS_HPP_INCLUDED__
#define __UTILS_HPP_INCLUDED__

#include <string>
#include <vector>
#include <opencv2/core.hpp>

void loadROIs(std::string name, std::vector<cv::Rect>& ROIs);

#endif // __UTILS_HPP_INCLUDED__