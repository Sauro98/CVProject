#include "DatasetHelper.hpp"

unsigned int targetFromEnc(double* buffer){
	unsigned int classEnc = 0;
	if((int)buffer[0])
		classEnc = BG_TARGET;
	else if ((int)buffer[1])
		classEnc = BOAT_TARGET;
	else if ((int)buffer[2])
		classEnc = SEA_TARGET;
	else{
		std::cout<<"Warning: unrecognized encoding "<<buffer[0]<<" "<<buffer[1]<<" "<<buffer[2]<<std::endl;
    }
	return classEnc;
}

void saveDataset(
					const std::string& name,
					std::vector<std::vector<double>>& descriptors
				)
{   
    
	auto rng = std::default_random_engine(42);
	std::shuffle(std::begin(descriptors), std::end(descriptors), rng);
	
	std::ofstream output;
	output.open(name, std::ofstream::binary);
	std::ostream_iterator<char> outIt(output);
	for(unsigned int i=0; i<descriptors.size(); ++i)
	{
		const char* byte_s = (char*)&descriptors[i][0];
		const char* byte_e = (char*)&descriptors[i].back() + sizeof(double);
		std::copy(byte_s, byte_e, outIt);
	}
	output.close();
	output.close();
}

void appendDescriptors(std::vector<std::vector<double>>& vect, const cv::Mat& descriptors, char oneHotEnc, bool addEnc){
    std::vector<double> line;
	for(unsigned int r=0; r<descriptors.rows; ++r)
	{
		line.clear();
		for(unsigned int c=0; c<descriptors.cols; ++c)
		{
			line.push_back(descriptors.at<float>(r,c));
		}

        if(addEnc)
        {
            int ch1 = ((oneHotEnc >> 2) & 0x01);
            int ch2 = ((oneHotEnc >> 1) & 0x01);
            int ch3 = ((oneHotEnc) & 0x01);
            
            // the first three digits of the one hot encoding
            line.push_back((double) ch1);
            line.push_back((double) ch2);
            line.push_back((double) ch3);
        }
        vect.push_back(line);
    }
}

void loadDataset(
					const std::string& name,
					std::vector<std::vector<double>>& inputs,
					std::vector<unsigned int>& outputs,
					std::vector<std::vector<double>>& vInputs,
					std::vector<unsigned int>& vOutputs,
					std::vector<std::vector<double>>& tInputs,
					std::vector<unsigned int>& tOutputs,
					
					unsigned int inSize,
					unsigned int vSize,
					unsigned int tSize
				)
{
	std::ifstream input;
	input.open(name,std::ifstream::binary);
	std::vector<double> in(128,0);
	double buffer[3] = {0, 0, 0};
	while(!input.eof() && inputs.size()<inSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 3*sizeof(double));
		inputs.push_back(in);
		outputs.push_back(targetFromEnc(buffer));
	}
	while(!input.eof() && vInputs.size()<vSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 3*sizeof(double));
		vInputs.push_back(in);
		vOutputs.push_back(targetFromEnc(buffer));
	}
	while(!input.eof() && tInputs.size()<tSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 3*sizeof(double));
		tInputs.push_back(in);
		tOutputs.push_back(targetFromEnc(buffer));
	}

	if (vInputs.size() < vSize || tInputs.size() < tSize)
		std::cout<<"Warning: not enough samples in the data file"<<std::endl;
}
