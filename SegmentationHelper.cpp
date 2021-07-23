#include "SegmentationHelper.hpp"
#include <thread>

/////////////////////////////////////////////////
//                                            //
//            SEGMENTATION HELPER             //
//                                            //
////////////////////////////////////////////////

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
        return (f.find(cv::String(DATASET_TOKEN)) != cv::String::npos) || (f.find(cv::String(PARAMETERS_TOKEN)) != cv::String::npos);
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

/////////////////////////////////////////////////
//                                            //
//            SEGMENTATION INFO               //
//                                            //
////////////////////////////////////////////////


// >>>>>
// >>>>> KEYPOINT CLASSIFICATION
// >>>>>


void classifyKeypoints(std::vector<std::vector<double>>& descVect, std::vector<unsigned int>& IDs, unsigned int index, unsigned int n, classFunc classify, void* usrData)
{
	const unsigned int k = descVect.size()/n;
	const unsigned int maxIt = (index==(n-1))?(descVect.size()):(k*(index+1));
	for(unsigned int i=k*index; i<maxIt; ++i)
	{
        //if ((i - (k * index)) % 100 == 0)
        //    std::cout<<""<<i - (k * index) + 1<<" of "<< maxIt - (k*index)<<std::endl; 
		IDs[i] = classify(descVect[i], usrData);
	}
}

void SegmentationInfo::computeKeypoints(bool shouldSharpen, classFunc classify, void* usrData, unsigned int numThread){
    SiftMasked smasked = SiftMasked();
    //BlackWhite_He equalizer = BlackWhite_He();
    //cv::Mat eq_img = equalizer.bgr_to_gray_HE(image, sharpen);
    cv::Mat eq_img = image.clone();
    //sharpen(image, eq_img, 1);
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
				
        for(unsigned int i=0; i<allKP.size(); ++i)
        {   
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
    //cv::imwrite(imageName.substr(0,imageName.size()-4) + "_labeled_kps.png", kpImg);
}


// >>>>>
// >>>>> GRID COMPUTATION
// >>>>>

bool inHollowMat(int r, int c, const cv::Size& size){
    return r > 0 && c>0 && r < size.height && c < size.width && (r!=0 || c!=0);
}

void drawSingleMarker(cv::Mat& img, cv::Mat& grid, cv::Vec2f& com, int r, int c, int layer,  double deltaX, double deltaY, double x0, double y0, cv::Scalar color){
    
    if(com[0] > 0. && com[1] > 0.){
        cv::circle(img, cv::Point2f(com[0], com[1]),1, color, -1, 8, 0 );
    }else{
        for(int nr = -1; nr <=1; nr++){
            for(int nc = -1; nc <=1; nc++){
                if(inHollowMat(r+nr, c+nc, img.size())){
                    if(grid.at<Vec3b>(r+nr,c+nc)(layer) > 0){
                        double x01 = x0 + nc*(deltaX/2);
                        double y01 = y0 + nr*(deltaY/2);
                        cv::circle(img, cv::Point2f(x01, y01),1, color, -1, 8, 0 );
                    }
                }
            }
        }
        //cv::circle(img, cv::Point2f(x0, y0),1, color, -1, 8, 0 );
    }
}

void drawMarkersFromGrid(cv::Mat& img, cv::Mat& grid, cv::Mat& bgComs, cv::Mat& seaComs, cv::Mat& boatsComs, double deltaX, double deltaY){
    for(int r = 0; r < grid.rows; r++){
        const double y0 = deltaY*r + deltaY/2;
        for(int c = 0; c < grid.cols; c++){
            const double x0 = deltaX*c + deltaX/2;
            if(grid.at<Vec3b>(r,c)(BOAT_GRID_INDEX) > 0){
                cv::Vec2f com = boatsComs.at<cv::Vec2f>(r,c);
                drawSingleMarker(img, grid, com, r, c, BOAT_GRID_INDEX, deltaX, deltaY, x0, y0, cv::Scalar::all(BOAT_LABEL)); 
            } else if (grid.at<Vec3b>(r,c)(SEA_GRID_INDEX) > 0){
                cv::Vec2f com = seaComs.at<cv::Vec2f>(r,c);
                drawSingleMarker(img, grid, com, r, c, SEA_GRID_INDEX, deltaX, deltaY, x0, y0, cv::Scalar::all(SEA_LABEL)); 
            } else if (grid.at<Vec3b>(r,c)(BG_GRID_INDEX) > 0) {
                cv::Vec2f com = bgComs.at<cv::Vec2f>(r,c);
                drawSingleMarker(img, grid, com, r, c, BG_GRID_INDEX, deltaX, deltaY, x0, y0, cv::Scalar::all(BG_LABEL));
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
                else{
                    for(int nr = -1; nr <=1; nr++){
                        for(int nc = -1; nc <=1; nc++){
                            if(inHollowMat(r+nr, c+nc, img.size())){
                                if(grid.at<uchar>(r+nr,c+nc) > 0){
                                    double x01 = x0 + nc*(deltaX/2);
                                    double y01 = y0 + nr*(deltaY/2);
                                    cv::circle(img, cv::Point2f(x01, y01),1, cv::Scalar(255,255,255), -1, 8, 0 );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void fillBg(cv::Mat& bg,cv::Mat& sea,const cv::Mat& boats, cv::Mat& laplacian, bool addBg){
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
    if(addBg)
        cv::bitwise_or(adder, bg, bg);
    //else
    //    cv::bitwise_or(adder, sea, sea);
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

cv::Mat getLaplacianMask(cv::Mat& image, uint cels_x, uint cels_y, double thresh){
    cv::Mat gray, laplacian;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(gray,gray, cv::Size(5,5), 0);
    
    cv::Laplacian(gray, laplacian, CV_32FC1);
    cv::normalize(laplacian, laplacian, cv::NORM_MINMAX);
    cv::threshold(laplacian,laplacian, thresh, 1., cv::THRESH_BINARY);
    laplacian *= 255;
    laplacian.convertTo(laplacian, CV_8UC1);
    cv::resize(laplacian, laplacian, cv::Size(cels_x, cels_y), cv::INTER_MAX);
    return laplacian;
}

void morphMask(cv::Mat& mask, int maskSize){
    cv::Size largeSize = cv::Size(mask.cols*2, mask.rows*2);
    cv::Size origSize = mask.size();

    cv::resize(mask,mask, largeSize, cv::INTER_NEAREST);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(maskSize,maskSize)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(maskSize,maskSize)));
    cv::resize(mask, mask, origSize, cv::INTER_NEAREST);
}

float meanVec(std::vector<float>& vec){
    float acc = 0.;
    for(const auto& el: vec)
        acc += el;
    return acc / vec.size();
}

float varVec(std::vector<float>& vec, float mean){
    float acc = 0;
    for(const auto& el: vec)
        acc += ((el - mean)*(el-mean));
    return acc / vec.size();
}

bool isNoisySmallBoat(cv::Mat& boatsGrid, double varThreshold){
    SiftMasked smasked;
    std::vector<cv::Rect> initialBoxes;
    std::vector<float> xs, ys;
    smasked.binaryToBBoxes(boatsGrid,initialBoxes, true);
    bool noisy = false;
    for(const auto& rect: initialBoxes){
        xs.push_back(rect.x);
        ys.push_back(rect.y);
    }
    float meanX = meanVec(xs);
    float meanY = meanVec(ys);
    float varX = varVec(xs, meanX);
    float varY = varVec(ys, meanY);
    float varCoeffX = sqrtf(varX) / boatsGrid.cols;
    float varCoeffY = sqrtf(varY) / boatsGrid.rows;
    //std::cout<<"varcoeffX "<<varCoeffX<<" varcoeffY "<<varCoeffY;
    // If the detected boats are small and distributed over 33% of both directions then 
    // it is most likely that the detected boats are noise. It is not possible
    // to distinguish an image with many sparse small boats to an image with just
    // noise detections. 
    if (varCoeffX > varThreshold || varCoeffY > varThreshold){
        // if the variance along one direction results to be significantly
        // bigger than the variance along the other, then it is likey that we are 
        // experiencing a convoy of small boats, since we assume that noise would be
        // more or less evenly distributed along both directions.

        float condNumber = 0.;
        // both are over the threshold so division by zero is not a concern
        if(varCoeffX >= varCoeffY){
            condNumber = varCoeffX / varCoeffY;
        } else {
            condNumber = varCoeffY / varCoeffX;
        }
        //std::cout<<" cond num "<<condNumber<<std::endl;
        // We want the largest variance to be at least 1.5 times the smallest one to have
        // a predominant detection direction
        if(condNumber > 1.5f)
            return false;
        else
            return true;
    }
    return false;
}

bool largeBoatFound(cv::Mat& boatsGrid, double threshold){
    SiftMasked smasked;
    std::vector<cv::Rect> initialBoxes;
    smasked.binaryToBBoxes(boatsGrid,initialBoxes, true);
    for(const auto& rect: initialBoxes){
        if(rect.area() >= threshold)
            return true;
    }
    return false;
}

void removeFlatVarianceBoats(cv::Mat& boatsGrid, cv::Mat& laplacian){
    for(int r = 0; r < boatsGrid.rows; r++){
        for(int c = 0; c < boatsGrid.rows; c++){
            if(laplacian.at<uchar>(r,c) == 0){
                boatsGrid.at<uchar>(r,c) = 0;
            }
        }
    }
}

void drawBlobPyramidBoats(cv::Mat& image, cv::Mat& markersMask, int layers){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(11,11),0);
    double med = median(gray);
	double lower = 0.67*med;
	double upper = 1.33*med;
	cv::Canny(gray, gray, lower, upper);
    //cv::imshow("canny", gray);
    /*std::cout<<"mmask type "<<type2str(markersMask.type())<<std::endl;
    cv::imshow("mmask", markersMask);
    cv::Mat blurred = image.clone();
    cv::GaussianBlur(blurred, blurred, cv::Size(5,5),0);
    for(int i = 0; i < layers; i++){
        cv::Mat blobbed;
        std::vector<cv::KeyPoint> blobs, darkBlobs;
        //cv::GaussianBlur(blurred, blurred, cv::Size(21,21),0);
        cv::imshow("blurred",blurred);
        cv::SimpleBlobDetector::Params params = cv::SimpleBlobDetector::Params();
        params.filterByColor = true;
        params.blobColor = 255;
        params.filterByCircularity = false;
        params.minCircularity = 0.3;
        params.maxCircularity = 1.0;
        params.filterByArea = true;
        params.filterByConvexity = false;
        params.minConvexity = 0.87;
        params.filterByInertia = false;
        params.maxInertiaRatio = 0.9;
        // detect light
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
        detector->detect(blurred, blobs);
        //detect dark
        params.blobColor = 0;
        detector = cv::SimpleBlobDetector::create(params);
        // detect
        detector->detect(blurred, darkBlobs);
        int k = 2*i;
        if(k == 0)
            k = 1;
        std::vector<cv::KeyPoint> validBlobs;
        for(const auto& kp: blobs){
            if(markersMask.at<cv::Vec3b>((int)kp.pt.y*k, (int)kp.pt.x*k) == cv::Vec3b(0,255,0)){
                validBlobs.push_back(kp);
            }
        }
        for(const auto& kp: darkBlobs){
            if(markersMask.at<cv::Vec3b>((int)kp.pt.y*k, (int)kp.pt.x*k) == cv::Vec3b(0,255,0)){
                validBlobs.push_back(kp);
            }
        }

        cv::drawKeypoints(blurred, validBlobs, blobbed, cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //cv::drawKeypoints(blobbed, darkBlobs, blobbed, cv::Scalar(0,0,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("blobbed",blobbed);
        cv::waitKey(0);
        cv::pyrDown(blurred, blurred);
    }*/


}


void SegmentationInfo::performSegmentation(bool showResults, bool addBg, uint maxDim, double minNormVariance) {

    cv::Mat markersMask = cv::Mat::zeros(image.size(), CV_8U);
    cv::Mat denseMarkers = image.clone();

    unsigned int cels_x = maxDim;
    unsigned int cels_y = maxDim;

    if(image.rows >= image.cols){
        cels_x = (maxDim * image.cols)/(image.rows);
        if (cels_x == 0)
            cels_x = 1;
    } else {
        cels_y = (maxDim * image.rows)/(image.cols);
        if (cels_y == 0)
            cels_y = 1;
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

    
    bool bigBoatFlag = largeBoatFound(chs[BOAT_GRID_INDEX], 9.);
    //cv::Mat boatLaplacian = getLaplacianMask(image, cels_x, cels_y, 0.1);
    //removeFlatVarianceBoats(chs[BOAT_GRID_INDEX], boatLaplacian);



    if(bigBoatFlag){
        //std::cout<<"big boat"<<std::endl;
        morphMask(chs[BOAT_GRID_INDEX],5);
        cv::morphologyEx(chs[BOAT_GRID_INDEX], chs[BOAT_GRID_INDEX], cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    } else {
        //std::cout<<"small boat"<<std::endl;
        bool isNoisy = isNoisySmallBoat(chs[BOAT_GRID_INDEX],0.2);
        if(isNoisy){
            // noise, erase hard
            morphMask(chs[BOAT_GRID_INDEX],21);
        }
        //morphMask(chs[BOAT_GRID_INDEX],1);
    }
    morphMask(chs[SEA_GRID_INDEX],5);
    morphMask(chs[BG_GRID_INDEX],5);


    cv::Mat laplacian = getLaplacianMask(image, cels_x, cels_y, minNormVariance);
    if(addBg){
        fillBg(chs[BG_GRID_INDEX], chs[SEA_GRID_INDEX], chs[BOAT_GRID_INDEX], laplacian, addBg);
        cv::morphologyEx(chs[SEA_GRID_INDEX], chs[SEA_GRID_INDEX], cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    }
    
    
    drawGridOnMat(denseMarkers, chs[BG_GRID_INDEX],bgComs, delta_x, delta_y, cv::Scalar(255,0,0));
    drawGridOnMat(denseMarkers, chs[SEA_GRID_INDEX],seaComs, delta_x, delta_y, cv::Scalar(0,0,255));
    drawGridOnMat(denseMarkers, chs[BOAT_GRID_INDEX],boatsComs, delta_x, delta_y, cv::Scalar(0,255,0));

    cv::Mat mergedGrid;
    cv::merge(chs, 3, mergedGrid);

    // BG has lowest priority, drawn first
    drawMarkersFromGrid(markersMask, mergedGrid,bgComs, seaComs, boatsComs, delta_x, delta_y);
    /*// sea drawn next
    drawMarkersFromGrid(markersMask, chs[SEA_GRID_INDEX], seaComs, delta_x, delta_y, cv::Scalar::all(SEA_LABEL));
    // boats have highest priority and so gat drawn last
    drawMarkersFromGrid(markersMask, chs[BOAT_GRID_INDEX],boatsComs,  delta_x, delta_y, cv::Scalar::all(BOAT_LABEL));*/

    if(showResults){
        cv::imshow("dense markers", denseMarkers);
        //cv::imwrite(imageName.substr(0,imageName.size()-4) + "_dense_markers.png", denseMarkers);
    }

	
    markersMask.convertTo(markersMask, CV_32S);
    cv::Mat sharp;
    sharpen(image, sharp, 1);

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
        //cv::imwrite(imageName.substr(0,imageName.size()-4) + "_watershed_transform.png", wshed);
        //drawBlobPyramidBoats(image, segmentationResult, 5);
    }
}



// >>>>>
// >>>>> METRICS COMPUTATION
// >>>>>

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

std::vector<cv::Point2i> getRectCorners(cv::Rect& r){
    cv::Point2i uple = cv::Point2i(r.x, r.y); 
    cv::Point2i upri = cv::Point2i(r.x + r.width, r.y); 
    cv::Point2i bole = cv::Point2i(r.x , r.y + r.height);
    cv::Point2i bori = cv::Point2i(r.x + r.width, r.y + r.height);
    std::vector<cv::Point2i> corners;
    corners.push_back(uple);
    corners.push_back(upri);
    corners.push_back(bole);
    corners.push_back(bori);
    return corners;
}

bool rectanglesIntersect(cv::Rect& r1, cv::Rect& r2){
    auto corners = getRectCorners(r2);
    for(const auto&c: corners){
        if(r1.contains(c))
            return true;
    }
    corners = getRectCorners(r1);
    for(const auto&c: corners){
        if(r2.contains(c))
            return true;
    }
    return false;
}

cv::Rect findUnionRect(cv::Rect& r1, cv::Rect& r2){
    auto corners1 = getRectCorners(r1);
    auto corners2 = getRectCorners(r2);
    corners1.insert(corners1.end(), corners2.begin(), corners2.end());
    int minx = corners1[0].x, miny = corners1[0].y, maxx = corners1[0].x, maxy = corners1[0].y;
    for(const auto& c: corners1){
        if(c.x < minx)
            minx = c.x;
        else if (c.x > maxx)
            maxx = c.x;

        if(c.y < miny)
            miny = c.y;
        else if (c.y > maxy)
            maxy = c.y;
    }
    return cv::Rect(minx, miny, maxx - minx, maxy - miny);
}

void mergeOverlappingRectangles(std::vector<cv::Rect>& rectangles, double threshold){
    int prevSize = rectangles.size();
    int newSize = prevSize;
    do{
        if(rectangles.size() <= 1){
            break;
        }

        size_t bestFirstIndex = 0;
        size_t bestSecondIndex = 0;
        double bestValue = 1e15;

        for(size_t firstIndex = 0; firstIndex < rectangles.size(); firstIndex++){
            cv::Rect& firstBox = rectangles[firstIndex];
            for(size_t secondIndex = firstIndex + 1; secondIndex < rectangles.size(); secondIndex++){
                cv::Rect& secondBox = rectangles[secondIndex];
                if(rectanglesIntersect(firstBox, secondBox)){
                    //auto unionRect = findUnionRect(firstBox, secondBox);
                    auto unionRect = firstBox | secondBox;
                    int areaSum = firstBox.area() + secondBox.area() - (firstBox & secondBox).area();
                    double metric = (double) unionRect.area() / (double)areaSum;
                    if(metric < bestValue){
                        bestFirstIndex = firstIndex;
                        bestSecondIndex = secondIndex;
                        bestValue = metric;
                    } 
                }
            }
        }
        if(bestValue < threshold){
            cv::Rect newRect = rectangles[bestFirstIndex] | rectangles[bestSecondIndex];
            rectangles.erase(rectangles.begin() + bestSecondIndex);
            rectangles.erase(rectangles.begin() + bestFirstIndex);
            rectangles.push_back(newRect);

        } else {
            break;
        }
        prevSize = newSize;
        newSize = rectangles.size();
    } while (prevSize != newSize);
}

void SegmentationInfo::findBBoxes(bool showBoxes, double minPercArea, double maxOverlapMetric){
    SiftMasked smasked = SiftMasked();

    // extract boat-labeled pixels and perform erosion/dilation
    cv::Mat boatsSegments = getBoatsMaskErodedDilated(segmentationResult);
    // compute bounding boxes on the result
    smasked.binaryToBBoxes(boatsSegments, estBboxes, true);

    //std::cout<<"Size before "<<estBboxes.size()<<std::endl;

    mergeOverlappingRectangles(estBboxes, maxOverlapMetric);
    
    //std::cout<<"Size after "<<estBboxes.size()<<std::endl;

    uint prevSize = estBboxes.size();
    uint newSize = prevSize;
    
    
    // filter bboxes with area <= 2% of the mean area
    do{
        filterBoundingBoxesByArea(estBboxes, minPercArea);
        prevSize = newSize;
        newSize = estBboxes.size();
    } while(prevSize != newSize);



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
        //cv::imwrite(imageName.substr(0,imageName.size()-4) + "_bboxes.png", bboxes_img);
    }

    
}

std::vector<double> SegmentationInfo::computeIOU(bool showBoxes, double minPercArea, double maxOverlapMetric, uint& falsePos, uint& falseNeg){
    std::vector<double> ious;
    
    findBBoxes(showBoxes, minPercArea, maxOverlapMetric);

    // precompute target bboxes masks
    auto targetBBoxesMasks = masksForBoxes(trueBboxes, image.size());
    // precompute estimated bboxes
    auto estBBoxesMasks = masksForBoxes(estBboxes, image.size());

    while(estBBoxesMasks.size() > 0){
        if(targetBBoxesMasks.size() == 0){
            //ious.push_back(0.);
            //estBBoxesMasks.pop_back();
            break;
        }

        size_t bestEstIndex = 0;
        size_t bestTargetIndex = 0;
        double bestIou = -1.;

        for(size_t estIndex = 0; estIndex < estBBoxesMasks.size(); estIndex++){
            cv::Mat& estBBox = estBBoxesMasks[estIndex];
            for(size_t targetIndex = 0; targetIndex < targetBBoxesMasks.size(); targetIndex++){
                cv::Mat& targetBBox = targetBBoxesMasks[targetIndex];
                int intersectionArea = cv::countNonZero(estBBox.mul(targetBBox));
                int unionArea = cv::countNonZero(estBBox + targetBBox);
                double iou = (double)intersectionArea / (double) unionArea;
                if(iou > bestIou){
                    bestEstIndex = estIndex;
                    bestTargetIndex = targetIndex;
                    bestIou = iou;
                }
            }
        }
        if(bestIou <= 0)
            break;
        estBBoxesMasks.erase(estBBoxesMasks.begin() + bestEstIndex);
        targetBBoxesMasks.erase(targetBBoxesMasks.begin() + bestTargetIndex);
        ious.push_back(bestIou);
    }
    //std::cout<<"Found "<<estBBoxesMasks.size()<<" false positives"<<std::endl;
    //std::cout<<"Missed "<<targetBBoxesMasks.size()<<" boats"<<std::endl;
    falsePos += estBBoxesMasks.size();
    falseNeg += targetBBoxesMasks.size();
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
