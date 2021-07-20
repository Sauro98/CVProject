#include "SegmentationHelper.hpp"
#include <thread>

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

void classifyKeypoints(std::vector<std::vector<double>>& descVect, std::vector<unsigned int>& IDs, unsigned int index, unsigned int n, classFunc classify, void* usrData)
{
	const unsigned int k = descVect.size()/n;
	const unsigned int maxIt = (index==(n-1))?(descVect.size()):(k*(index+1));
	for(unsigned int i=k*index; i<maxIt; ++i)
	{
        if ((i - (k * index)) % 100 == 0)
            std::cout<<""<<i - (k * index) + 1<<" of "<< maxIt - (k*index)<<std::endl; 
		IDs[i] = classify(descVect[i], usrData);
	}
}

void SegmentationInfo::computeKeypoints(bool sharpen, classFunc classify, void* usrData, unsigned int numThread){
    SiftMasked smasked = SiftMasked();
    //BlackWhite_He equalizer = BlackWhite_He();
    //cv::Mat eq_img = equalizer.bgr_to_gray_HE(image, sharpen);
    cv::Mat eq_img = image.clone();
    if(classify)
    {
        cv::Mat allDescriptors;
        std::vector<cv::KeyPoint> allKP = smasked.findFeatures(eq_img, cv::Mat(), allDescriptors);
        std::vector<std::vector<double>> descVect;
        appendDescriptors(descVect, allDescriptors, 0, false);
        
        boatKps.clear();
        seaKps.clear();
        bgKps.clear();
		
		std::vector<unsigned int> IDs(allKP.size(), 0);
		std::vector<std::thread> threads;
		for(unsigned int i=0; i<(numThread-1); ++i)
		{
			threads.push_back(std::thread(classifyKeypoints, std::ref(descVect), std::ref(IDs), i, numThread, classify, usrData));
		}
		classifyKeypoints(descVect, IDs, (numThread-1), numThread, classify, usrData);
		
		for(unsigned int i=0; i<threads.size(); ++i)
		{
			threads[i].join();
		}
				
        //std::cout<<"#kps "<<allKP.size()<<std::endl;
        for(unsigned int i=0; i<allKP.size(); ++i)
        {   
            //if(i%100 == 0)
            //    std::cout<<"KP #"<<i<<std::endl;
            const unsigned int classID = IDs[i];
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

void matDilateErode3x3(cv::Mat& toClose){
    uchar erosionComponents[] = {1,1,1,1,1,1,1,1,1};
    cv::Mat erosionElement = cv::Mat(3,3,CV_8UC1, erosionComponents);
    cv::dilate(toClose, toClose, erosionElement);
    cv::erode(toClose, toClose, erosionElement);
}

void matErodeDilate3x3(cv::Mat& toClose){
    uchar erosionComponents[] = {1,1,1,1,1,1,1,1,1};
    cv::Mat erosionElement = cv::Mat(3,3,CV_8UC1, erosionComponents);
    cv::erode(toClose, toClose, erosionElement);
    cv::dilate(toClose, toClose, erosionElement);
}

void matDilateErode2x2(cv::Mat& toClose){
    cv::Mat erosionElement = cv::Mat::ones(2,2,CV_8UC1);
    cv::dilate(toClose, toClose, erosionElement);
    cv::erode(toClose, toClose, erosionElement);
}

void matErodeDilate2x2(cv::Mat& toClose){
    cv::Mat erosionElement = cv::Mat::ones(cv::Size(2,2),CV_8UC1);
    cv::erode(toClose, toClose, erosionElement);
    cv::dilate(toClose, toClose, erosionElement);
}

void removeIsolatedPixels(cv::Mat& image){
    for(int r = 1; r < image.rows - 1; r++){
        for(int c = 1; c < image.cols - 1; c++){
            int sum = image.at<uchar>(r-1,c-1) + image.at<uchar>(r-1,c) + image.at<uchar>(r-1,c+1);
            sum = sum + image.at<uchar>(r,c-1) + image.at<uchar>(r,c+1);
            sum = sum + image.at<uchar>(r+1,c-1) + image.at<uchar>(r+1,c) + image.at<uchar>(r+1,c+1);
            if(sum == 0)
                image.at<uchar>(r,c) = 0;
        }
    }
}

void drawMarkersFromGrid(cv::Mat& img, cv::Mat& grid, cv::Mat& coms, double deltaX, double deltaY, cv::Scalar color){
    for(int r = 0; r < grid.rows; r++){
        const double y0 = deltaY*r + deltaY/2;
        for(int c = 0; c < grid.cols; c++){
            const double x0 = deltaX*c + deltaX/2;
            if(grid.at<uchar>(r,c) > 0){
                cv::Vec2f com = coms.at<cv::Vec2f>(r,c);
                if(com[0] > 0. && com[1] > 0.)
                    cv::circle(img, cv::Point2f(com[0], com[1]),1, color, -1, 8, 0 );
                else
                    cv::circle(img, cv::Point2f(x0, y0),1, color, -1, 8, 0 );
            }
        }
    }
}

void drawGridOnMat(cv::Mat& img, cv::Mat& grid,cv::Mat& coms, double deltaX, double deltaY, cv::Scalar color) {
    for(int r = 0; r < grid.rows; r++){
        const double y0 = deltaY*r + deltaY/2;
        for(int c = 0; c < grid.cols; c++){
            const double x0 = deltaX*c + deltaX/2;
            if(grid.at<uchar>(r,c) > 0){
                cv::rectangle(img, cv::Rect(x0 - deltaX/2, y0 - deltaY/2, deltaX, deltaY),color, -1, 8);
                cv::Vec2f com = coms.at<cv::Vec2f>(r,c);
                if(com[0] > 0. && com[1] > 0.)
                    cv::circle(img, cv::Point2f(com[0], com[1]),1, cv::Scalar(255,255,255), -1, 8, 0 );
                else
                    cv::circle(img, cv::Point2f(x0, y0),1, cv::Scalar(255,255,255), -1, 8, 0 );
            }
        }
    }
}

void fillBg(cv::Mat& bg,cv::Mat& sea,const cv::Mat& boats, cv::Mat& laplacian){
    cv::Mat adder = cv::Mat::zeros(bg.size(), bg.type());
    cv::Mat seaAdder;
    cv::bitwise_or(boats, bg, adder);
    cv::bitwise_or(sea, adder, adder);
    cv::Mat dilationElement = cv::Mat::ones(cv::Size(5,5), CV_8UC1);
    cv::dilate(adder, adder, dilationElement);
    cv::bitwise_not(adder, adder);
    seaAdder = adder.clone();
    dilationElement = cv::Mat::ones(cv::Size(7,7), CV_8UC1);
    cv::dilate(laplacian, laplacian, dilationElement);
    cv::bitwise_not(laplacian, laplacian);
    cv::bitwise_and(laplacian, adder, adder);
    cv::bitwise_or(adder, bg, bg);
    /*cv::bitwise_not(adder, adder);
    cv::bitwise_and(seaAdder, adder, seaAdder);
    cv::bitwise_or(seaAdder, sea, sea);*/
}

void fillKpAccumulator(std::vector<cv::KeyPoint>& kps, cv::Mat& accumulator, cv::Mat& coms, int index, double delta_x, double delta_y, uint cels_x, uint cels_y){
    for(const auto& kp: kps){
        unsigned int x = kp.pt.x/delta_x;
        unsigned int y = kp.pt.y/delta_y;
        x = x<cels_x?x:cels_x-1;
        y = y<cels_y?y:cels_y-1;
        accumulator.at<cv::Vec3f>(y,x)(index)+= 1.;
        coms.at<cv::Vec2f>(y,x) += cv::Vec2f(kp.pt.x, kp.pt.y);
    }
    for(int r = 0; r < coms.rows; r++){
        for(int c = 0; c < coms.cols; c++){
            float acc = accumulator.at<cv::Vec3f>(r,c)(index);
            if(acc>= 1.)
                coms.at<cv::Vec2f>(r,c) /= acc;
        }
    }
}

void fillGrid(cv::Mat& accumulator, cv::Mat& grid){
    for(unsigned int x=0; x<accumulator.cols; ++x)
		{
			for(unsigned int y=0; y<accumulator.rows; ++y)
			{
                const cv::Vec3f binValue = accumulator.at<cv::Vec3f>(y,x);
                float tot = binValue[BOAT_GRID_INDEX] + binValue[SEA_GRID_INDEX] + binValue[BG_GRID_INDEX];
                if(tot  == 0)
                    continue;
                float density = 1. / tot;
                
                if(binValue[BOAT_GRID_INDEX] / density > 0.33 || binValue[SEA_GRID_INDEX] / density > 0.33 || binValue[BG_GRID_INDEX] / density > 0.33)
				{
					if(binValue[BOAT_GRID_INDEX]>binValue[SEA_GRID_INDEX] && binValue[BOAT_GRID_INDEX]>binValue[BG_GRID_INDEX])
					{   
                        grid.at<cv::Vec3b>(y,x)(BOAT_GRID_INDEX) = 255;
					}
					if(binValue[SEA_GRID_INDEX]>binValue[BOAT_GRID_INDEX] && binValue[SEA_GRID_INDEX]>binValue[BG_GRID_INDEX])
					{
                        grid.at<cv::Vec3b>(y,x)(SEA_GRID_INDEX) = 255;
					}
					if(binValue[BG_GRID_INDEX]>binValue[BOAT_GRID_INDEX] && binValue[BG_GRID_INDEX]>binValue[SEA_GRID_INDEX])
					{
                        grid.at<cv::Vec3b>(y,x)(BG_GRID_INDEX) = 255;
					}
				}
			}
		}
}

cv::Mat getLaplacianMask(cv::Mat& image, uint cels_x, uint cels_y){
    cv::Mat gray, laplacian;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(gray,gray, cv::Size(5,5), 0);
    
    cv::Laplacian(gray, laplacian, CV_32FC1);
    cv::normalize(laplacian, laplacian, cv::NORM_MINMAX);
    cv::threshold(laplacian,laplacian, 0.01, 1., cv::THRESH_BINARY);
    laplacian *= 255;
    laplacian.convertTo(laplacian, CV_8UC1);
    cv::resize(laplacian, laplacian, cv::Size(cels_x, cels_y), cv::INTER_MAX);
    return laplacian;
}

void morphMask(cv::Mat& mask){
    cv::Size largeSize = cv::Size(mask.cols*2, mask.rows*2);
    cv::Size origSize = mask.size();

    cv::resize(mask,mask, largeSize, cv::INTER_NEAREST);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
    cv::resize(mask, mask, origSize);
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

        unsigned int maxDim = 50;
		unsigned int cels_x = maxDim;
		unsigned int cels_y = maxDim;

        if(image.rows >= image.cols){
            cels_x = (maxDim * image.cols)/(image.rows);
        } else {
            cels_y = (maxDim * image.rows)/(image.cols);
        }

		double delta_x = image.cols/cels_x;
		double delta_y = image.rows/cels_y;
		

        cv::Mat accumulator = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_32FC3);
        cv::Mat grid = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_8UC3);
        cv::Mat boatsComs = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_32FC2);
        cv::Mat seaComs = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_32FC2);
        cv::Mat bgComs = cv::Mat::zeros(cv::Size(cels_x, cels_y), CV_32FC2);

        fillKpAccumulator(boatKps, accumulator, boatsComs, BOAT_GRID_INDEX, delta_x, delta_y, cels_x, cels_y);
        fillKpAccumulator(seaKps, accumulator, seaComs, SEA_GRID_INDEX, delta_x, delta_y, cels_x, cels_y);
        fillKpAccumulator(bgKps, accumulator, bgComs, BG_GRID_INDEX, delta_x, delta_y, cels_x, cels_y);
		fillGrid(accumulator, grid);

        cv::Mat chs[3];
        cv::split(grid, chs);

        
        /*matDilateErode3x3(chs[BOAT_GRID_INDEX]);
        matDilateErode3x3(chs[SEA_GRID_INDEX]);
        matDilateErode3x3(chs[BG_GRID_INDEX]);


        //removeIsolatedPixels(chs[BOAT_GRID_INDEX]);
        if(boatKps.size() > 15)
            matErodeDilate3x3(chs[BOAT_GRID_INDEX]);
        matErodeDilate3x3(chs[SEA_GRID_INDEX]);
        matErodeDilate3x3(chs[BG_GRID_INDEX]);*/
        morphMask(chs[BOAT_GRID_INDEX]);
        morphMask(chs[SEA_GRID_INDEX]);
        morphMask(chs[BG_GRID_INDEX]);


        cv::Mat laplacian = getLaplacianMask(image, cels_x, cels_y);
        fillBg(chs[BG_GRID_INDEX], chs[SEA_GRID_INDEX], chs[BOAT_GRID_INDEX], laplacian);

        // BG has lowest priority, drawn first
        drawMarkersFromGrid(markersMask, chs[BG_GRID_INDEX],bgComs, delta_x, delta_y, cv::Scalar::all(BG_LABEL));
        // sea drawn next
        drawMarkersFromGrid(markersMask, chs[SEA_GRID_INDEX], seaComs, delta_x, delta_y, cv::Scalar::all(SEA_LABEL));
        // boats have highest priority and so gat drawn last
        drawMarkersFromGrid(markersMask, chs[BOAT_GRID_INDEX],boatsComs,  delta_x, delta_y, cv::Scalar::all(BOAT_LABEL));
        
        
        
        drawGridOnMat(denseMarkers, chs[BG_GRID_INDEX],bgComs, delta_x, delta_y, cv::Scalar(255,0,0));
        drawGridOnMat(denseMarkers, chs[SEA_GRID_INDEX],seaComs, delta_x, delta_y, cv::Scalar(0,0,255));
        drawGridOnMat(denseMarkers, chs[BOAT_GRID_INDEX],boatsComs, delta_x, delta_y, cv::Scalar(0,255,0));


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
