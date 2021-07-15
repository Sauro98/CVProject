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

bool maskFilenamesMatch(const cv::String& mask, const cv::String& base, bool boat)
{
	if (boat){
		return base.substr(base.size()-(4+5)) == "_maskb.png" or 
		(
			mask.size() == (base.size()+ 6) and // 6 = "_maskb".size()
			mask.substr(0,mask.size()-(4+6)) == base.substr(0,base.size()-4)
		);
	} else {
		return base.substr(base.size()-(4+5)) == "_mask.png" or 
		(
			mask.size() == (base.size()+ 5) and // 5 = "_mask".size()
			mask.substr(0,mask.size()-(4+5)) == base.substr(0,base.size()-4)
		);
	}
}

void drawROIs(cv::Mat& img, std::vector<cv::Rect>& ROIs)
{
	for(const auto& ROI: ROIs)
	{
		cv::rectangle(img,ROI,cv::Scalar(0,0,255));
	}
}

void bgr_to_gray_HE(cv::Mat& image, cv::Mat& out)
{
	// Created by Anna Zuccante on 20/06/2021.
	cv::Mat gray_img;
	cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray_img, out);
};

void sharpen(cv::Mat& input, cv::Mat& output, int laplacianWeight)
{
	// by Ivano
    cv::Mat imgLaplacian;
    //cv::Laplacian(input, imgLaplacian, CV_32F);
    cv::GaussianBlur(input, output, cv::Size(5,5),0);
    cv::Laplacian(output, imgLaplacian, CV_32F);
    output.convertTo(output, CV_32F);
    output = output - laplacianWeight * imgLaplacian;
    // convert back to 8bits gray scale
    output.convertTo(output, CV_8U);
}

double median(cv::Mat& img)
{
	// https://gist.github.com/heisters/9cd68181397fbd35031b
	double m = (img.rows*img.cols) / 2;
        int bin = 0;
        double med = -1.0;

        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::Mat hist;
        cv::calcHist( &img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

        for ( int i = 0; i < histSize && med < 0.0; ++i )
        {
            bin += cvRound( hist.at< float >( i ) );
            if ( bin > m && med < 0.0 )
                med = i;
        }

        return med;
}

typedef struct {
	cv::Mat* mask;
	cv::Mat* img;
	cv::String* name;
	unsigned int brushSize = 20;
	unsigned int count = 0;
	cv::Point prevPt;
	bool boat;
} SeaCallbackData;

void updateSeaImage(SeaCallbackData& data)
{
	showMask(*data.name, *data.img, *data.mask, data.boat);
}

static void selectSeaCallback( int event, int x, int y, int flags, void* userdata)
{
	SeaCallbackData* data = (SeaCallbackData*)userdata;
	const double brush_size = data->brushSize;
	
    if( x < 0 || x >= data->img->cols || y < 0 || y >= data->img->rows )
        return;
	
	if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) ) 
	{
		cv::Point pt(x, y);
		if (data->prevPt.x == -1 && data->prevPt.y == -1) {
			data->prevPt = pt;
		} 
		cv::line(*data->mask, data->prevPt, pt, cv::Scalar(255,255,255), data->brushSize,8,0);
		data->prevPt=pt;
		
		//cv::circle(*data->mask, pt, data->brushSize, cv::Scalar(255,255,255),-1,8,0);
		data->count += 1;
		if(data->count>10)
		{
			updateSeaImage(*data);
			data->count = 0;
		}
	}
	if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_RBUTTON) ) 
	{
		cv::Point pt(x, y);
		if (data->prevPt.x == -1 && data->prevPt.y == -1) {
			data->prevPt = pt;
		} 
		cv::line(*data->mask, data->prevPt, pt, cv::Scalar(0,0,0), data->brushSize,8,0);
		data->prevPt=pt;
		data->count += 1;
		if(data->count>10)
		{
			updateSeaImage(*data);
			data->count = 0;
		}
	}

	
	if( event == cv::EVENT_LBUTTONUP )
	{
		updateSeaImage(*data);
		data->prevPt = cv::Point(-1,-1);
	} else if( event == cv::EVENT_RBUTTONUP )
	{
		updateSeaImage(*data);
		data->prevPt = cv::Point(-1,-1);
	}else if( event == cv::EVENT_LBUTTONDOWN ){
        data->prevPt = cv::Point(x,y);
	} else if( event == cv::EVENT_RBUTTONDOWN ){
        data->prevPt = cv::Point(x,y);
	}else if (!(flags & cv::EVENT_FLAG_LBUTTON)) {
		updateSeaImage(*data);
	}
}

unsigned int selectSea(cv::String name, cv::Mat& img, cv::Mat& mask, unsigned int brushSize, bool boat)
{
	SeaCallbackData data;
	data.img = &img;
	data.mask = &mask;
	data.name = &name;
	data.prevPt = cv::Point(-1,-1);
	data.brushSize = brushSize;
	data.boat = boat;
	
	updateSeaImage(data);
	cv::setMouseCallback(name, selectSeaCallback, &data);
	while(true)
	{
		int pressed = cv::waitKey(0);
		if(pressed == 'q')
		{
			data.brushSize *= 2;
			data.brushSize = data.brushSize>100?100:data.brushSize;
		}
		else if(pressed == 'a')
		{
			data.brushSize /= 2;
			data.brushSize = data.brushSize<5?5:data.brushSize;
		}
		else break;
	}
	return data.brushSize;
}

void selectKeypointsRectangle(cv::String name, cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& selected)
{
	cv::Mat out;
	bool loop = true;
	
	// Loop until the user presses enter (or space) without selecting any ROIs
	// (or until they press c as indicated in opencv's docs)
	while(loop)
	{
		cv::drawKeypoints(img, keypoints, out, cv::Scalar(0,0,255));
		cv::drawKeypoints(out, selected, out, cv::Scalar(0,255,0));
		cv::Rect ROI = cv::selectROI(name, out, false);
		// stop looping if no ROI is selected
		if(ROI.x == 0 and ROI.y == 0 and ROI.width == 0 and ROI.height == 0)
		{
			loop = false;
		}
		// otherwise select the keypoints inside the ROI
		// and loop again to select the next ROI
		else
		{
			for(const auto& kp: keypoints)
			{
				if(ROI.contains(kp.pt))
				{
					selected.push_back(kp);
				}
			}
			keypoints.erase( std::remove_if
			(
				keypoints.begin(), keypoints.end(),
				[ROI](const cv::KeyPoint& kp)
				{
					return ROI.contains(kp.pt);
				}
			), keypoints.end());
		}
	}
}

void findAllKeypoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, bool shouldSharpen)
{
	cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
	cv::Mat equalized;
	bgr_to_gray_HE(img, equalized);
	if(shouldSharpen)
	{
		sharpen(equalized, equalized);
	}
    sift->detect(equalized, keypoints);
}

void createMask(cv::Mat& img, std::vector<cv::KeyPoint>& background, std::vector<cv::KeyPoint>& foreground, cv::Mat& mask)
{
	bgr_to_gray_HE(img, mask);
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
	
	// Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(mask.size(), CV_32SC1);
    
	// Draw the foreground markers
	for(const auto& kp: foreground)
	{
		cv::circle(markers, kp.pt, 3, cv::Scalar(255), -1);
	}
	// Draw the background marker
	for(const auto& kp: background)
	{
		cv::circle(markers, kp.pt, 3, cv::Scalar(1), -1);
	}
	
	cv::cvtColor(mask,mask,cv::COLOR_GRAY2BGR);
	cv::watershed(mask, markers);
	markers.convertTo(mask, CV_8U);
}

void showMask(cv::String name, cv::Mat& img, cv::Mat& mask, bool boat)
{
	cv::Mat foreground;
	cv::Mat background;
	if (mask.channels() == 1) {
		cv::cvtColor(mask,foreground,cv::COLOR_GRAY2BGR);
	} else {
		foreground = mask.clone();
	}

	cv::Mat channels[3];
	cv::split(foreground, channels);
	if(boat) {
		channels[0] *= 0;
		channels[2] *= 0;
	} else {	
		channels[0] *= 0;
		channels[1] *= 0;
	}
	cv::merge(&channels[0], 3, foreground);
	

	foreground.convertTo(foreground, CV_32FC3, 0.85/255.0);
	img.convertTo(background, CV_32FC3, 0.85/255.0);
	cv::multiply(foreground, foreground, foreground);
	cv::multiply(cv::Scalar::all(1.0)-foreground, background, background);
	cv::add(foreground, background, background);
	imshow(name,background);
}

void saveMask(cv::String baseName, cv::Mat& mask, bool boat)
{
	if(boat)
		cv::imwrite(baseName+"_maskb.png", mask);
	else
		cv::imwrite(baseName+"_mask.png", mask);
}
