#include "SegmentationHelper.hpp"

void drawMarkers(cv::Mat& markers, std::vector<cv::KeyPoint> kps, cv::Scalar color){
    for (int i = 0; i < kps.size(); i++) {
        cv::Point2f pti = kps[i].pt;
        cv::circle( markers, pti,1, color, -1, 8, 0 );
    }
}

void removeMasksFromImagesFnames(std::vector<cv::String>& fnames){
    fnames.erase(std::remove_if(fnames.begin(), fnames.end(), [](const cv::String& f) {
        return f.find(cv::String(MASK_TOKEN)) != cv::String::npos;
    }), fnames.end());
}

void removeDatasetsFromBBoxesFnames(std::vector<cv::String>& fnames){
    fnames.erase(std::remove_if(fnames.begin(), fnames.end(), [](const cv::String& f) {
        return f.find(cv::String(DATASET_TOKEN)) != cv::String::npos;
    }), fnames.end());
}

SegmentationHelper::SegmentationHelper(cv::String& inputDirectory, cv::String& imagesExt){
    cv::utils::fs::glob(inputDirectory, cv::String(imagesExt), filenames);
    cv::utils::fs::glob(inputDirectory, cv::String(BBOX_EXT), bboxes_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(SEA_MASK_EXT), masks_fnames);
    cv::utils::fs::glob(inputDirectory, cv::String(BOAT_MASK_EXT), boat_masks_fnames);
    removeMasksFromImagesFnames(filenames);
    removeDatasetsFromBBoxesFnames(bboxes_fnames);

    std::sort(filenames.begin(), filenames.end());
    std::sort(bboxes_fnames.begin(), bboxes_fnames.end());
    std::sort(masks_fnames.begin(), masks_fnames.end());
    std::sort(boat_masks_fnames.begin(), boat_masks_fnames.end());

    if (filenames.size() != bboxes_fnames.size() || filenames.size() != masks_fnames.size() || filenames.size() != boat_masks_fnames.size()){
        std::cout<<"Some masks/bboxes are missing"<<std::endl;
        exit(1);
    }
}

std::vector<SegmentationInfo> SegmentationHelper::loadInfos(bool boatsFromBBoxes){
    std::vector<SegmentationInfo> infos;
    SiftMasked smasked= SiftMasked();
    for (int i = 0; i < filenames.size(); i++) {
        int check = 0;
        cv::Mat original_img = cv::imread(filenames[i], cv::IMREAD_COLOR);
        cv::Mat seaMask = cv::imread(masks_fnames[i], cv::IMREAD_GRAYSCALE);
        std::vector<cv::Rect> bboxes = smasked.checkFileBB(bboxes_fnames[i], check);
        if(check == 1) {
            exit(1);
        }
        cv::Mat boatMask;
        if(boatsFromBBoxes) {
            boatMask = smasked.findBinMask(original_img,bboxes);
        } else {
            boatMask = cv::imread(boat_masks_fnames[i], cv::IMREAD_GRAYSCALE);
        }
        boatMask.convertTo(boatMask, CV_32F);
        seaMask.convertTo(seaMask, CV_32F);
        cv::Mat intersection = boatMask.mul(seaMask);
        boatMask = boatMask - intersection;
        cv::Mat bgMask = 1 - seaMask - boatMask - intersection;

        boatMask.convertTo(boatMask, CV_8U);
        seaMask.convertTo(seaMask, CV_8U);
        bgMask.convertTo(bgMask, CV_8U);

        SegmentationInfo info = SegmentationInfo(original_img, seaMask, boatMask, bgMask, bboxes, filenames[i]);
        infos.push_back(info);
    }
    return infos;
}

void SegmentationInfo::computeKeypoints(bool sharpen, classFunc classify, void* usrData){
    SiftMasked smasked = SiftMasked();
    BlackWhite_He equalizer = BlackWhite_He();
    cv::Mat eq_img = equalizer.bgr_to_gray_HE(image, sharpen);
    
    if(classify)
    {
        cv::Mat allDescriptors;
        std::vector<cv::KeyPoint> allKP = smasked.findFeatures(eq_img, cv::Mat(), allDescriptors);
        std::vector<std::vector<double>> descVect;
        appendDescriptors(descVect, allDescriptors, 0, false);
        
        boatKps.clear();
        seaKps.clear();
        bgKps.clear();
        //std::cout<<"#kps "<<allKP.size()<<std::endl;
        for(unsigned int i=0; i<allKP.size(); ++i)
        {   
            //if(i%100 == 0)
            //    std::cout<<"KP #"<<i<<std::endl;
            const unsigned int classID = classify(descVect[i], usrData);
            if (classID == BOAT_LABEL)
            {
                boatKps.push_back(allKP[i]);
            }
            else if (classID == SEA_LABEL)
            {
                seaKps.push_back(allKP[i]);
            }
            else if (classID == BG_LABEL)
            {
                bgKps.push_back(allKP[i]);
            }
        }
        smasked.findDescriptors(eq_img, boatKps, boatDescriptors);
        smasked.findDescriptors(eq_img, seaKps, seaDescriptors);
        smasked.findDescriptors(eq_img, bgKps, bgDescriptors);
    }
    else
    {
        boatKps = smasked.findFeatures(eq_img, boatsMask, boatDescriptors);
        seaKps = smasked.findFeatures(eq_img, seaMask, seaDescriptors);
        bgKps = smasked.findFeatures(eq_img, bgMask, bgDescriptors);
    }
}

void SegmentationInfo::showLabeledKps(){
    cv::Mat kpImg = image.clone();
    cv::drawKeypoints(kpImg, boatKps, kpImg, cv::Scalar(0,255,0));
    cv::drawKeypoints(kpImg, seaKps, kpImg, cv::Scalar(0,0,255));
    cv::drawKeypoints(kpImg, bgKps, kpImg, cv::Scalar(255,0,0));
    cv::imshow("kps", kpImg);
}

struct bin3u
{
	unsigned int b1 = 0;
	unsigned int b2 = 0;
	unsigned int b3 = 0;
};

void matErodeDilate3x3(cv::Mat& toClose){
    uchar erosionComponents[] = {1,1,1,1,1,1,1,1,1};
    cv::Mat erosionElement = cv::Mat(3,3,CV_8UC1, erosionComponents);
    cv::dilate(toClose, toClose, erosionElement);
    cv::erode(toClose, toClose, erosionElement);
}

void drawGridOnMat(cv::Mat& img, cv::Mat& grid, double deltaX, double deltaY, cv::Scalar color) {
    for(int r = 0; r < grid.rows; r++){
        const double y0 = deltaY*r + deltaY/2;
        for(int c = 0; c < grid.cols; c++){
            const double x0 = deltaX*c + deltaX/2;
            if(grid.at<uchar>(r,c) > 0){
                cv::circle(img, cv::Point2f(x0,y0),1, color, -1, 8, 0 );
            }
        }
    }
}

void SegmentationInfo::performSegmentation(bool showResults) {
    cv::Mat markersMask = cv::Mat::zeros(image.size(), CV_8U);
	if(false)
	{
		drawMarkers(markersMask,boatKps, cv::Scalar::all(BOAT_LABEL));
		drawMarkers(markersMask,seaKps, cv::Scalar::all(SEA_LABEL));
		drawMarkers(markersMask,bgKps, cv::Scalar::all(BG_LABEL));
	}
	else
	{   
        cv::Mat denseMarkers = image.clone();
		unsigned int cels_x = 50;
		unsigned int cels_y = 50;
        cv::Mat accumulator = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_32FC3);
        cv::Mat grid = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_8UC3);
		double delta_x = image.cols/cels_x;
		double delta_y = image.rows/cels_y;
		
		for(const auto& kp: boatKps)
		{
			unsigned int x = kp.pt.x/delta_x;
			unsigned int y = kp.pt.y/delta_y;
			x = x<cels_x?x:cels_x-1;
			y = y<cels_y?y:cels_y-1;
            accumulator.at<cv::Vec3f>(y,x)(0)+= 1.;
		}
		for(const auto& kp: seaKps)
		{
			unsigned int x = kp.pt.x/delta_x;
			unsigned int y = kp.pt.y/delta_y;
			x = x<cels_x?x:cels_x-1;
			y = y<cels_y?y:cels_y-1;
            accumulator.at<cv::Vec3f>(y,x)(1) += 1.;
		}
		for(const auto& kp: bgKps)
		{
			unsigned int x = kp.pt.x/delta_x;
			unsigned int y = kp.pt.y/delta_y;
			x = x<cels_x?x:cels_x-1;
			y = y<cels_y?y:cels_y-1;
            accumulator.at<cv::Vec3f>(y,x)(2) += 1.;
		}
		
		for(unsigned int x=0; x<cels_x; ++x)
		{
			const double x0 = delta_x*x + delta_x/2;
			for(unsigned int y=0; y<cels_y; ++y)
			{
				const double y0 = delta_y*y + delta_y/2;
                const cv::Vec3f binValue = accumulator.at<cv::Vec3f>(y,x);
				if(binValue[0]>=3 or binValue[1]>=3 or binValue[2]>=3)
				{
					if(binValue[0]>binValue[1] and binValue[0]>binValue[2])
					{   
                        grid.at<cv::Vec3b>(y,x)(0) = 255;
						//cv::circle( markersMask, cv::Point2f(x0,y0),1, cv::Scalar::all(BOAT_LABEL), -1, 8, 0 );
						//cv::circle( denseMarkers, cv::Point2f(x0,y0),1, cv::Scalar(0,255,0), -1, 8, 0 );
					}
					if(binValue[1]>binValue[0] and binValue[1]>binValue[2])
					{
                        grid.at<cv::Vec3b>(y,x)(1) = 255;
						//cv::circle( markersMask, cv::Point2f(x0,y0),1, cv::Scalar::all(SEA_LABEL), -1, 8, 0 );
						//cv::circle( denseMarkers, cv::Point2f(x0,y0),1, cv::Scalar(0,0,255), -1, 8, 0 );
					}
					if(binValue[2]>binValue[0] and binValue[2]>binValue[1])
					{
                        grid.at<cv::Vec3b>(y,x)(2) = 255;
						//cv::circle( markersMask, cv::Point2f(x0,y0),1, cv::Scalar::all(BG_LABEL), -1, 8, 0 );
						//cv::circle( denseMarkers, cv::Point2f(x0,y0),1, cv::Scalar(255,0,0), -1, 8, 0 );
					}
				}
			}
		}
        cv::Mat chs[3];
        cv::split(grid, chs);
        matErodeDilate3x3(chs[0]);
        matErodeDilate3x3(chs[1]);
        matErodeDilate3x3(chs[2]);

        drawGridOnMat(markersMask, chs[0], delta_x, delta_y, cv::Scalar::all(BOAT_LABEL));
        drawGridOnMat(markersMask, chs[1], delta_x, delta_y, cv::Scalar::all(SEA_LABEL));
        drawGridOnMat(markersMask, chs[2], delta_x, delta_y, cv::Scalar::all(BG_LABEL));

        drawGridOnMat(denseMarkers, chs[0], delta_x, delta_y, cv::Scalar(0,255,0));
        drawGridOnMat(denseMarkers, chs[1], delta_x, delta_y, cv::Scalar(0,0,255));
        drawGridOnMat(denseMarkers, chs[2], delta_x, delta_y, cv::Scalar(255,0,0));


        if(showResults){
            cv::imshow("dense markers", denseMarkers);
        }
	}
    markersMask.convertTo(markersMask, CV_32S);
    cv::Mat sharp;
    sharpen(image, sharp, 1);
    /*
    BlackWhite_He eql = BlackWhite_He();
    cv::Mat mask = eql.bgr_to_gray_HE(image, true, 1);
	cv::blur(mask, mask, cv::Size(3,3));
	
	double med = median(mask);
	double lower = 0.67*med;
	double upper = 1.33*med;
	lower = lower<0?0:lower;
	upper = upper>255?255:upper;
	cv::Canny(mask, mask, lower, upper);
	cv::bitwise_not(mask, mask);
	cv::distanceTransform(mask, mask, cv::DIST_L2, 3);
	cv::normalize(mask, mask, 0, 1.0, cv::NORM_MINMAX);
	mask.convertTo(mask, CV_8UC3, 255, 0);
	cv::subtract(cv::Scalar::all(255),mask,mask);
    cv::cvtColor(mask,mask,cv::COLOR_GRAY2BGR);*/

    cv::watershed(sharp, markersMask);

    Mat wshed(markersMask.size(), CV_8UC3);

    for(int r = 0; r < markersMask.rows; r++ )
        for(int c = 0; c < markersMask.cols; c++ )
        {
            int index = markersMask.at<int>(r,c);
            if( index == -1 )
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,255);
            else if (index == BOAT_LABEL)
                wshed.at<Vec3b>(r,c) = Vec3b(0,255,0);
            else if (index == SEA_LABEL)
                wshed.at<Vec3b>(r,c) = Vec3b(0,0,255);
            else
                wshed.at<Vec3b>(r,c) = Vec3b(255,0,0);
        }

    segmentationResult = wshed.clone();

    if (showResults) {
        wshed = wshed*0.5 + image*0.5;
        imshow( "watershed transform", wshed );
    }
}



cv::Mat getBoatsMaskErodedDilated(cv::Mat segmentationResult){
    // keep only green (boats) channel
    cv::Mat boatsSegments = segmentationResult.clone();
    cv::Mat chs[3];
    cv::split(boatsSegments,chs);
    boatsSegments = chs[1];

    // erode mask with elements:
    // 1 1 1
    // 1 1 1
    // 1 1 1
    uchar erosionComponents[] = {1,1,1,1,1,1,1,1,1};
    cv::Mat erosionElement = cv::Mat(3,3,CV_8UC1, erosionComponents);
    cv::erode(boatsSegments, boatsSegments, erosionElement);
    // and then dilate
    cv::dilate(boatsSegments, boatsSegments, erosionElement);

    return boatsSegments;
}

void filterBoundingBoxesByArea(std::vector<cv::Rect>& bboxes, double ratio) {
    double avg_area = 0;
    for(auto& bbox: bboxes){
        avg_area += (double)bbox.area();
    }
    avg_area /= (double)bboxes.size();

    bboxes.erase(std::remove_if(bboxes.begin(), bboxes.end(), [avg_area, ratio](const cv::Rect& r) {
        return r.area() <= ratio*avg_area;
    }), bboxes.end());
}

std::vector<cv::Mat> masksForBoxes(std::vector<cv::Rect>& boxes, cv::Size img_size) {
    std::vector<cv::Mat> masks;
    for(const auto& b: boxes){
        cv::Mat mask = cv::Mat::zeros(img_size, CV_8UC1);
        cv::rectangle(mask,b,cv::Scalar(1),-1, LINE_8);
        masks.push_back(mask);
    }
    return masks;
}

std::vector<double> SegmentationInfo::computeIOU(bool showBoxes){
    std::vector<double> ious;
    SiftMasked smasked = SiftMasked();

    // extract boat-labeled pixels and perform erosion/dilation
    cv::Mat boatsSegments = getBoatsMaskErodedDilated(segmentationResult);
    // compute bounding boxes on the result
    smasked.binaryToBBoxes(boatsSegments, estBboxes, true);
    // filter bboxes with area <= 2% of the mean area
    filterBoundingBoxesByArea(estBboxes, 0.02);
    // precompute target bboxes masks
    auto targetBBoxesMasks = masksForBoxes(trueBboxes, image.size());
    // precompute estimated bboxes
    auto estBBoxesMasks = masksForBoxes(estBboxes, image.size());

    for(const auto& estBBoxMask: estBBoxesMasks){
        if(targetBBoxesMasks.size() == 0){
            //std::cout<<"Warning, there are more estimated bboxes than real ones"<<std::endl;
            //break;
            ious.push_back(0.);
            continue;
        }
        int intersectionArea = cv::countNonZero(targetBBoxesMasks[0].mul(estBBoxMask));
        size_t best_index = 0;
        for(size_t i = 1; i < targetBBoxesMasks.size(); i++){
            int tempIntArea = cv::countNonZero(targetBBoxesMasks[i].mul(estBBoxMask));
            if(tempIntArea > intersectionArea){
                intersectionArea = tempIntArea;
                best_index = i;
            }
        }
        if(intersectionArea == 0){
            ious.push_back(0.);
        } else {
            int unionArea = cv::countNonZero(targetBBoxesMasks[best_index] + estBBoxMask);
            double iou = (double)intersectionArea/(double)unionArea;
            ious.push_back(iou);
            targetBBoxesMasks.erase(targetBBoxesMasks.begin() + best_index);
        }
    }

    // display bounding boxes
    if(showBoxes){
        cv::Mat bboxes_img = image.clone();
        for(auto& box: trueBboxes) {
            cv::rectangle(bboxes_img, box, cv::Scalar(255,0,0),3);
        }
        for(auto& box: estBboxes) {
            cv::rectangle(bboxes_img, box, cv::Scalar(0,255,0),2);
        }
        cv::imshow("bboxes", bboxes_img);
    }

    return ious;
}

double SegmentationInfo::computePixelAccuracy(){
    cv::Mat chs[3];
    cv::split(segmentationResult,chs);
    cv::Mat seaSegments = chs[SEA_CH_INDEX].clone();
    cv::Mat otherSegments = chs[BOATS_CH_INDEX].clone() + chs[BG_CH_INDEX].clone();
    cv::Mat correctSeaPixels = seaSegments.mul(seaMask);
    cv::Mat correctOtherPixels = otherSegments.mul(bgMask + boatsMask);
    int correctPixels = cv::countNonZero(correctSeaPixels) + cv::countNonZero(correctOtherPixels);
    int totalPixels = image.rows * image.cols;
    return (double) correctPixels / (double) totalPixels;
}

void SegmentationInfo::appendBoatsDescriptors(std::vector<std::vector<double>>& vect, bool addEnc = true) const {
    appendDescriptors(vect, boatDescriptors, BOATS_1H_ENC, addEnc);
}
void SegmentationInfo::appendSeaDescriptors(std::vector<std::vector<double>>& vect, bool addEnc = true) const {
    appendDescriptors(vect, seaDescriptors, SEA_1H_ENC, addEnc);
}
void SegmentationInfo::appendBgDescriptors(std::vector<std::vector<double>>& vect, bool addEnc = true) const {
    appendDescriptors(vect, bgDescriptors, BG_1H_ENC, addEnc);
}

cv::String& SegmentationInfo::getName(){
    return imageName;
}
