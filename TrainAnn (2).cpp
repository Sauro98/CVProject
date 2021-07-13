#include <iostream>
#include <vector>
#include <fstream>
#include "ANN.hpp"

void loadDataset(
					const std::string& name,
					std::vector<std::vector<double>>& inputs,
					std::vector<std::vector<double>>& outputs,
					std::vector<std::vector<double>>& vInputs,
					std::vector<std::vector<double>>& vOutputs,
					std::vector<std::vector<double>>& tInputs,
					std::vector<std::vector<double>>& tOutputs
				)
{
	std::ifstream input;
	input.open(name,std::ifstream::binary);
	std::vector<double> in(128,0);
	std::vector<double> out(2,0);
	while(!input.eof() and inputs.size()<150000)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&out[0], 2*sizeof(double));
		for(unsigned int i=0; i<in.size(); ++i)
		{
			in[i] = in[i]/200.d;
		}
		inputs.push_back(in);
		outputs.push_back(out);
	}
	
	/*
	std::ifstream input;
	input.open(name);
	double x;
	unsigned int count = 0;
	std::vector<double> in;
	std::vector<double> out;
	
	std::cout<<0;
	while(input>>x)
	{
		if(count>=128)
		{
			out.push_back(x);
		}
		else
		{
			in.push_back(x/200.d);
		}
		++count;
		if(count==130)
		{
			count = 0;
			inputs.push_back(in);
			outputs.push_back(out);
			in.clear();
			out.clear();
			std::cout<<"\r"<<inputs.size();
			if(inputs.size()>=150000)
			{
				break;
			}
		}
	}
	std::cout<<"\n";
	*/
	input.close();
	
	for(unsigned int i=0; i<50000; ++i)
	{
		vInputs.push_back(inputs.back());
		vOutputs.push_back(outputs.back());
		inputs.pop_back();
		outputs.pop_back();
	}
	for(unsigned int i=0; i<50000; ++i)
	{
		tInputs.push_back(inputs.back());
		tOutputs.push_back(outputs.back());
		inputs.pop_back();
		outputs.pop_back();
	}
}

/*
int foo(int argc, char** argv)
{	
	mnist_loader trainSet("../MNIST/train-images.idx3-ubyte",
						"../MNIST/train-labels.idx1-ubyte");
	mnist_loader testSet("../MNIST/t10k-images.idx3-ubyte",
						"../MNIST/t10k-labels.idx1-ubyte");
	
	const unsigned int trainSize = 5*(trainSet.size())/6.0;
	const unsigned int validSize = trainSet.size() - trainSize;
	const unsigned int testSize = 100;//testSet.size();
	std::cout<<trainSize<<" "<<validSize<<" "<<testSize<<"\n";
	
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> outputs;
	
	std::vector<std::vector<double>> vInputs;
	std::vector<std::vector<double>> vOutputs;
	
	std::vector<std::vector<double>> tInputs;
	std::vector<std::vector<double>> tOutputs;
	
	for(int i=0; i<trainSize; ++i)
	{
		outputs.push_back(toCategorical(trainSet.labels(i)));
		inputs.push_back(trainSet.images(i));
	}
	for(int i=trainSize; i<(trainSize+validSize); ++i)
	{
		vOutputs.push_back(toCategorical(trainSet.labels(i)));
		vInputs.push_back(trainSet.images(i));
	}
	for(int i=0; i<testSize; ++i)
	{
		tOutputs.push_back(toCategorical(testSet.labels(i)));
		tInputs.push_back(testSet.images(i));
	}
	
	std::vector<unsigned int> topology;
	topology.push_back(trainSet.rows()*trainSet.cols());
	topology.push_back(20);
	topology.push_back(10);
	Ann ann = Ann(topology);
	ann.randomize(0);
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	std::cout<<"\nTraining....\n\n";
	ann.train(inputs,outputs, 
	//inputs,outputs,
	vInputs,vOutputs,
	0.001, 100, 1, -1, true);
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	return 0;
}

*/

int main(int argc, char** argv)
{
	std::vector<unsigned int> topology;
	topology.push_back(128);
	topology.push_back(80);
	topology.push_back(2);
	Ann ann = Ann(topology);
	ann.randomize(0);
	
	//ann.show();
	
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> outputs;
	
	std::vector<std::vector<double>> vInputs;
	std::vector<std::vector<double>> vOutputs;
	
	std::vector<std::vector<double>> tInputs;
	std::vector<std::vector<double>> tOutputs;
	
	std::cout<<"Loading dataset...\n";
	loadDataset("dataset.bin", inputs, outputs, vInputs, vOutputs, tInputs, tOutputs);
	std::cout<<"Dataset loaded.\n";
	std::cout<<inputs.size()<<" training samples, "<<vInputs.size()<<" validation samples, "<<tInputs.size()<<" test samples.\n";
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	std::cout<<"\nTraining....\n\n";
	ann.train(
	inputs,outputs,
	vInputs,vOutputs,
	0.001, 5000, 4, -1, true, true, 0.3, 0.1);
	
	std::cout<<"Error: "<<ann.getError(inputs,outputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(inputs,outputs)*100<<"%\n";
	
	std::cout<<"Error: "<<ann.getError(tInputs,tOutputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(tInputs,tOutputs)*100<<"%\n";
	
	return 0;
}