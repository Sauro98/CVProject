#include <iostream>
#include <vector>
#include "ANN.hpp"
#include "SegmentationHelper.hpp"
#include "DatasetHelper.hpp"

void categoryToOneHot(const std::vector<unsigned int>& cat, std::vector<std::vector<double>>& out, unsigned int n)
{
	for(unsigned int i=0; i<cat.size(); ++i)
	{
		std::vector<double> tmp(3,0);
		tmp[cat[i]-1] = 1;
		out.push_back(tmp);
	}
}

void rescaleVector(std::vector<double>& in)
{
	for(unsigned int i=0; i<in.size(); ++i)
	{
		in[i] = in[i]/255.d;
	}
}

void rescaleVectors(std::vector<std::vector<double>>& in)
{
	for(unsigned int i=0; i<in.size(); ++i)
	{
		rescaleVector(in[i]);
	}
}

unsigned int segCallback(std::vector<double>& input, void* usrData)
{
	rescaleVector(input);
	Ann* ann = (Ann*)usrData;
	return 1 + ann->useCategorical(input);
}

double vectorAvg(std::vector<double>& v){
    double avg = 0.;
    for(const auto& d: v)
        avg += d;
    return avg / v.size();
}

void computeShowMetrics(std::vector<SegmentationInfo>& infos, bool displayImages, bool detailed, Ann* ann){
    std::vector<double> allIous;
    std::vector<double> allPixAcc;
    std::cout<<std::endl;
    for(size_t i = 0; i < infos.size(); i++) {
        auto& imageInfo = infos[i];
        if(!detailed){
            std::cout<<"Image ("<<i+1<<"/"<<infos.size()<<")"<<std::endl;
        }
        imageInfo.computeKeypoints(true,segCallback,ann);
        if(displayImages){
            imageInfo.showLabeledKps();
        }
        imageInfo.performSegmentation(displayImages);
        auto ious = imageInfo.computeIOU(displayImages);
        if(detailed){
            std::cout<<imageInfo.getName()<<std::endl;
            for(const auto& iou: ious)
                std::cout<<"iou: "<<iou<<std::endl;
        }
        
        allIous.insert(allIous.end(), ious.begin(), ious.end());
        double pixAcc = imageInfo.computePixelAccuracy();
        allPixAcc.push_back(pixAcc);
        if(detailed){
            std::cout<<"Pixel accuracy: "<<pixAcc*100.<<"%"<<std::endl;
        }
        if(displayImages)
            cv::waitKey(0);
    }

    if(allIous.size() > 0){    
        std::cout<<std::endl;
        std::sort(allIous.begin(), allIous.end());
        double avgIou = vectorAvg(allIous);
        std::cout<<" - Iou (average, min, max) = ("<<avgIou<<", "<<allIous[0]<<", "<<allIous[allIous.size() - 1]<<")"<<std::endl;
    }

    if(allPixAcc.size() > 0){
        std::cout<<std::endl;
        std::sort(allPixAcc.begin(), allPixAcc.end());
        double avgPixacc = vectorAvg(allPixAcc);
        std::cout<<" - Pixel accuracy (average, min, max) = ("<<avgPixacc<<", "<<allPixAcc[0]<<", "<<allPixAcc[allPixAcc.size() - 1]<<")"<<std::endl;
    }
}

void testSeg(Ann* ann, bool display = false, bool detailed = false)
{
	cv::String input_directory = "../TestSet";
    cv::String images_ext = "*g";

    SegmentationHelper sHelper = SegmentationHelper(input_directory, images_ext);
    auto segmentationInfos = sHelper.loadInfos(false);
    computeShowMetrics(segmentationInfos, display, detailed, ann);
}

int main(int argc, char** argv)
{
	std::vector<unsigned int> topology;
	topology.push_back(128);
	topology.push_back(80);
	topology.push_back(3);
	Ann ann = Ann(topology);
	ann.randomize(0);
	
	std::vector<std::vector<double>> inputs, tInputs, outputs, tOutputs;
    std::vector<unsigned int> outCat, tOutCat;
    std::cout<<"Loading dataset...\n";
    loadDataset("TrainingSet.bin", inputs, outCat, inputs, outCat, inputs, outCat, 10000, 0, 0); //647800
    loadDataset("TestSet.bin", tInputs, tOutCat, tInputs, tOutCat, tInputs, tOutCat, 10000, 0, 0); //262506
	categoryToOneHot(outCat,outputs,3);
	categoryToOneHot(tOutCat,tOutputs,3);
	rescaleVectors(inputs);
	rescaleVectors(tInputs);
	std::cout<<"Dataset loaded.\n";
	std::cout<<inputs.size()<<" training samples, "<<tInputs.size()<<" test samples.\n";
	
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	//testSeg(&ann);
	
	std::cout<<"\nTraining....\n\n";
	ann.train(
	
	inputs,outputs,
	//tInputs,tOutputs,
	
	tInputs,tOutputs,
	0.001, 10, 1, -1, true, true, 0.3, 0.1);
	
	ann.save("../ann");
	
	std::cout<<"Error: "<<ann.getError(inputs,outputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(inputs,outputs)*100<<"%\n";
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	testSeg(&ann,true);
	
	return 0;
}
