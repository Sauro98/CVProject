#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

class DecisionTree
{
public:
	DecisionTree(unsigned int in): inSize(in), root(true,0) {}
	
	unsigned int predict(const std::vector<double>& in)
	{
		return root.predict(in);
	}
	
	void train(
				const std::vector<std::vector<double>>& inputs,
				const std::vector<unsigned int>& outputs,
				const std::vector<std::vector<double>>& vInputs,
				const std::vector<unsigned int>& vOutputs,
				double minError = 1, unsigned int maxNodes = 1000
			)
	{
		double error;
		tree.reserve(maxNodes);
		do
		{
			root.resetPenalty();
			for(unsigned int i=0; i<inputs.size(); ++i)
			{
				const unsigned int res = root.predict(inputs[i]);
				if(res!=outputs[i])
				{
					root.getLast()->addPenalty();
				}
			}
			
			double penalty = 0;
			DecisionNode* node = root.getArgMaxPenalty(penalty);
			//std::cout<<" "<<penalty<<"\n";
			
			std::vector<double> avg_0(inSize,0);
			std::vector<double> avg_1(inSize,0);
			unsigned int count_0 = 0;
			unsigned int count_1 = 0;

			for(unsigned int i=0; i<inputs.size(); ++i)
			{
				root.predict(inputs[i]);
				if(root.getLast()==node)
				{
					if(outputs[i]==0)
					{
						++count_0;
						for(unsigned int j=0; j<inputs[i].size(); ++j)
						{
							avg_0[j] += inputs[i][j];
						}
					}
					else
					{
						++count_1;
						for(unsigned int j=0; j<inputs[i].size(); ++j)
						{
							avg_1[j] += inputs[i][j];
						}
					}
				}
			}
			count_0 = (count_0>0)?count_0:1;
			count_1 = (count_1>0)?count_1:1;
			for(unsigned int i=0; i<avg_0.size(); ++i)
			{
				avg_0[i] /= count_0;
				avg_1[i] /= count_1;
			}
			
			double maxDiff = 0;
			unsigned int maxIndex = 0;
			for(unsigned int i=0; i<avg_0.size(); ++i)
			{
				const double diff = std::abs(avg_0[i]-avg_1[i]);
				if(diff>maxDiff)
				{
					maxDiff = diff;
					maxIndex = i;
				}
			}

			const double thresh = (avg_0[maxIndex]+avg_1[maxIndex])/2.d;
			unsigned int highID = (avg_0[maxIndex]>thresh)?0:1;
			
			tree.push_back(DecisionNode(true,1-highID));
			tree.push_back(DecisionNode(true,highID));
			
			node->split(thresh,maxIndex, &tree[tree.size()-1],&tree[tree.size()-2]);
			
			error = getError(inputs,outputs);
			std::cout<<error<<"\n";
			/*
			std::cout<<"===================================\n";
			show();
			std::cout<<"===================================\n";
			*/
		}
		while(error>minError and tree.size()<maxNodes);
	}
	
	double getError(const std::vector<std::vector<double>>& inputs, const std::vector<unsigned int>& outputs)
	{
		double error = 0;
		for(unsigned int i=0; i<inputs.size(); ++i)
		{
			if(predict(inputs[i])!=outputs[i])
			{
				++error;
			}
		}
		return error/inputs.size()*100;
	}
	
	void show()
	{
		root.show("");
	}

private:
	
	class DecisionNode
	{
	public:
		DecisionNode(bool isF, unsigned int p, double th=0, unsigned int i=0, DecisionNode* nH = nullptr, DecisionNode* nL = nullptr):
		isFinal(isF), pred(p), threshold(th), index(i), childHigh(nH), childLow(nL), last(nullptr), penalty(0) {}
		
		unsigned int predict(const std::vector<double>& in)
		{
			if(isFinal)
			{
				return pred;
			}
			if(in[index]>threshold)
			{
				last = childHigh;
				return childHigh->predict(in);
			}
			last = childLow;
			return childLow->predict(in);
		}
		
		void resetPenalty()
		{
			penalty = 0;
			if(not isFinal)
			{
				childHigh->resetPenalty();
				childLow->resetPenalty();
			}
		}
		
		void addPenalty()
		{
			++penalty;
		}
		
		DecisionNode* getArgMaxPenalty(double& curMax)
		{
			if(isFinal)
			{
				curMax = penalty;
				return this;
			}
			
			double high;
			double low;
			DecisionNode* nH = childHigh->getArgMaxPenalty(high);
			DecisionNode* nL = childLow->getArgMaxPenalty(low);
			
			if(high>low and high>penalty)
			{
				curMax = high;
				return nH;
			}
			if(low>penalty)
			{
				curMax = low;
				return nL;
			}
			curMax = penalty;
			return this;
			
		}
		
		DecisionNode* getLast()
		{
			if(isFinal) return this;
			return last->getLast();
		}
		
		void split(double thresh, unsigned int i, DecisionNode* high, DecisionNode* low)
		{
			isFinal = false;
			threshold = thresh;
			index = i;
			childHigh = high;
			childLow = low;
		}
		
		void show(std::string s)
		{
			if(isFinal)
			{
				std::cout<<s<<"final "<<pred<<"\n";
			}
			else
			{
				std::cout<<s<<"t "<<threshold<<" i "<<index<<"\n";
				childHigh->show(s+" ");
				childLow->show(s+" ");
			}
		}
		
	private:
		bool isFinal;
		double threshold;
		unsigned int index;
		unsigned int pred;
		DecisionNode* childHigh;
		DecisionNode* childLow;
		DecisionNode* last;
		double penalty;
	};
	
	unsigned int inSize;
	DecisionNode root;
	std::vector<DecisionNode> tree;
};

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
	double buffer[2] = {0,0};
	
	while(!input.eof() and inputs.size()<inSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 2*sizeof(double));
		inputs.push_back(in);
		outputs.push_back(buffer[0]?0:1);
	}
	while(!input.eof() and vInputs.size()<vSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 2*sizeof(double));
		vInputs.push_back(in);
		vOutputs.push_back(buffer[0]?0:1);
	}
	while(!input.eof() and tInputs.size()<tSize)
	{
		input.read((char*)&in[0], 128*sizeof(double));
		input.read((char*)&buffer[0], 2*sizeof(double));
		tInputs.push_back(in);
		tOutputs.push_back(buffer[0]?0:1);
	}
}

int main(int argc, char** argv)
{	
	std::vector<std::vector<double>> inputs;
	std::vector<unsigned int> outputs;
	
	std::vector<std::vector<double>> vInputs;
	std::vector<unsigned int> vOutputs;
	
	std::vector<std::vector<double>> tInputs;
	std::vector<unsigned int> tOutputs;
	
	std::cout<<"Loading dataset...\n";
	loadDataset("dataset.bin", inputs, outputs, vInputs, vOutputs, tInputs, tOutputs, 20000, 50000, 50000);
	std::cout<<"Dataset loaded.\n";
	std::cout<<inputs.size()<<" training samples, "<<vInputs.size()<<" validation samples, "<<tInputs.size()<<" test samples.\n";
	
	DecisionTree tree(128);
	tree.train(inputs,outputs,vInputs,vOutputs,5,10000);
	
	return 0;
}
