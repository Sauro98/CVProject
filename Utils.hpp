#ifndef __UTILS_HPP_INCLUDED__
#define __UTILS_HPP_INCLUDED__

#include <string>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


/// Utility function to load a set of bounding boxes from a txt file.
/// The txt file must have the same name as the image it refers to and it has
/// to contain a bounding box in each line. Bounding boxes are specified by
/// starting x, starting y, width and height, all separated by a whitespace.
void loadROIs(std::string name, std::vector<cv::Rect>& ROIs);

/// Utility function which allows to select a number of ROIs from an image.
///
/// Parameters:
/// -------
/// `name`: the name of the window used for the selection
/// `img`: the image on which to select the ROIs 
/// `ROIs`: a vector that will hold the selected ROIs after the function terminates
void selectROIs(cv::String name, cv::Mat& img, std::vector<cv::Rect>& ROIs);

/// Utility function which saves all the ROIs contained in `ROIs` in a txt file named
/// according to `basename`. `basename` should not end in ".txt" as it will be automatically appended;  
/// its value should instead be the name of the image from which the ROIs were extracted, stripped of 
/// its file extension. 
///
/// Rois will be saved one per line ("\\n" line separator) by their starting x, starting y, width and height, all
/// separated by a whitespace.
void saveROIs(cv::String baseName, std::vector<cv::Rect>& ROIs);

/// Returns true if the filenames match excluding their extensions. Assumes 3 charachters
/// long extensions.
bool filenamesMatch(const cv::String& f1, const cv::String& f2);

/// Updates `img` by drawing all the ROIs contained in `ROIs` with a thin red line.
void drawROIs(cv::Mat& img, std::vector<cv::Rect>& ROIs);

#endif // __UTILS_HPP_INCLUDED__
