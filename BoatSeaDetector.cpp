#include <iostream>
#include <fstream>
#include "SiftMasked.h"
#include "BlackWhite_He.h"
#include "Utils.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include "SegmentationHelper.hpp"

int main(int argc, char** argv)
{
    cv::String input_directory = cv::String(argv[1]);
    cv::String images_ext = cv::String(argv[2]);

    SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);

    for(auto& imageInfo: segmentationInfos) {
        imageInfo.computKeypoints(true);
        imageInfo.showLabeledKps();
        imageInfo.performSegmentation(true);
    }
}
