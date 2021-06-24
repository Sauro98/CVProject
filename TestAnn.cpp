#include <iostream>
#include <vector>
#include "ANN.hpp"

int main(int argc, char** argv)
{
	std::vector<unsigned int> topology;
	topology.push_back(2);
	topology.push_back(5);
	topology.push_back(5);
	topology.push_back(2);
	Ann ann = Ann(topology);
	
	ann.show();
	
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> outputs;
	
	for(int i=0; i<2; ++i)
	{
		for(int j=0; j<2; ++j)
		{
			std::vector<double> tmp;
			tmp.push_back(i);
			tmp.push_back(j);
			inputs.push_back(tmp);
			tmp.clear();
			tmp.push_back(1-(i==j));
			tmp.push_back((i==j));
			outputs.push_back(tmp);
			ann.use(inputs[inputs.size()-1],tmp);
			std::cout<<i<<" "<<j<<": "<<tmp[0]<<", "<<tmp[1]<<" ("<<outputs[2*i+j][0]<<", "<<outputs[2*i+j][1]<<")\n";
		}
	}
	
	std::cout<<"Error: "<<ann.getError(inputs,outputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(inputs,outputs)*100<<"%\n";
	
	std::cout<<"\nTraining....\n\n";
	ann.train(inputs,outputs,inputs,outputs,
				0.001, 10000, 20, -1, false);
	
	for(int i=0; i<2; ++i)
	{
		for(int j=0; j<2; ++j)
		{
			std::vector<double> tmp(2,0);
			ann.use(inputs[2*i+j],tmp);
			std::cout<<i<<" "<<j<<": "<<tmp[0]<<", "<<tmp[1]<<" ("<<outputs[2*i+j][0]<<", "<<outputs[2*i+j][1]<<")\n";
		}
	}
	std::cout<<"Error: "<<ann.getError(inputs,outputs)<<"\n";
	std::cout<<"Categorical: "<<ann.getCategoricalError(inputs,outputs)*100<<"%\n";
	
	ann.show();
	
//	bool saved = ann.save("../test_ann");
//	std::cout<<"Saved: "<<saved<<"\n";
//	Ann ann2 = Ann("../test_ann");
//	std::cout<<"Reloaded ANN:\n";
//	ann2.show();
	return 0;
}
