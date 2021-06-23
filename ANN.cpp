#include <iostream>
#include <fstream>
#include <random>

#include "ANN.hpp" //includes vector, string

# define ReLU(x) (x>0?x:0)
# define GradReLU(x) (x>0?1:0)
# define Sigmoid(x) ( (1+ x/(1 + (x>0?x:-x)))*0.5 )

Ann::Ann(std::string fileName)
{
    std::ifstream file;
    file.open(fileName+".ann");
    if(!file.is_open())
    {
        throw std::runtime_error("Error loading file...");
    }
    else
    {
		unsigned int NoL;
		file >> NoL;
		NpL = std::vector<unsigned int>(NoL, 0);
		for(unsigned int i=0; i<NoL; ++i)
		{
			file >> NpL[i];
		}
		
		for(unsigned int layer=0; layer<(NoL-1); ++layer)
		{
			WpL.push_back(std::vector<std::vector<double>>());
			for(unsigned int i=0; i<(NpL[layer]+1); ++i)
			{
				WpL[layer].push_back(std::vector<double>(NpL[layer+1], 0.0));
				for(unsigned int j=0; j<NpL[layer+1]; ++j)
				{
					file >> WpL[layer][i][j];
				}
			}
		}
	
		for(unsigned int layer=0; layer<NpL.size(); ++layer)
		{
			results.push_back(std::vector<double>(NpL[layer]+1,1));
		}
		results[results.size()-1].pop_back();
		
		file.close();
    }
}

Ann::Ann(const std::vector<unsigned int> &NPL)
{
	NpL = NPL;
	const unsigned int NoL = NpL.size();
	
	WpL.clear();
	for(unsigned int layer=0; layer<(NoL-1); ++layer)
	{
		WpL.push_back(std::vector<std::vector<double>>());
		for(unsigned int i=0; i<(NpL[layer]+1); ++i)
		{
			WpL[layer].push_back(std::vector<double>(NpL[layer+1], 0.0));
		}
	}
	
	results.clear();
	for(unsigned int layer=0; layer<NpL.size(); ++layer)
	{
		results.push_back(std::vector<double>(NpL[layer]+1,1));
	}
	results[results.size()-1].pop_back();
	
	randomize(42);
}

void Ann::randomize(unsigned int seed)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-2.0,2.0);
    generator.seed(seed);
	for(unsigned int layer=0; layer<WpL.size(); ++layer)
	{
		for(unsigned int i=0; i<WpL[layer].size(); ++i)
		{
			for(unsigned int j=0; j<WpL[layer][i].size(); ++j)
			{
				WpL[layer][i][j] = distribution(generator);
			}
		}
	}
}

bool Ann::save(std::string fileName)
{	
	std::ofstream file;
    file.open(fileName+".ann");
    if(!file.is_open())
    {
        return false;
    }
	
	file << NpL.size() << " ";
	for(unsigned int i=0; i<NpL.size(); ++i)
	{
		file << NpL[i] << " ";
	}
	
	for(unsigned int layer=0; layer<WpL.size(); ++layer)
	{
		for(unsigned int i=0; i<WpL[layer].size(); ++i)
		{
			for(unsigned int j=0; j<WpL[layer][i].size(); ++j)
			{
				file << WpL[layer][i][j] << " ";
			}
		}
	}
	
	file.close();
	return true;
}

void Ann::show()
{
	std::cout   <<"\n"
				<<"==================== ANN DATA ==================== \n"
				<<"Number of layers: "<<NpL.size()<<"\n";

	for(unsigned int layer=0; layer<NpL.size(); ++layer)
	{
		std::cout<<"Number of nodes in layer "<<layer<<": "<<NpL[layer]<<"\n";
	}
	
	for(unsigned int layer=0; layer<WpL.size(); ++layer)
	{
		std::cout<<"\n"<<"Weigths in layer "<<layer<<":\n";
		for(unsigned int i=0; i<WpL[layer].size(); ++i)
		{
			std::cout<<"Start node "<<i<<": ";
			for(unsigned int j=0; j<WpL[layer][i].size(); ++j)
			{
				std::cout<<WpL[layer][i][j]<<" ("<<j<<")"<<" ";
			}
			std::cout<<"\n";
		}
	}
	
	std::cout<<"\n";
	for(unsigned int layer=0; layer<results.size(); ++layer)
	{
		std::cout<<"Last input of layer "<<layer<<": ";
		for(unsigned int i=0; i<results[layer].size(); ++i)
		{
			std::cout<<results[layer][i]<<" ("<<i<<")"<<" ";
		}
		std::cout<<"\n";
	}
	
	if(DpL.size()>0)
	{
		std::cout<<"\n";
		for(unsigned int layer=0; layer<DpL.size(); ++layer)
		{
			std::cout<<"\n"<<"Last update of weigths in layer "<<layer<<":\n";
			for(unsigned int i=0; i<DpL[layer].size(); ++i)
			{
				std::cout<<"Start node "<<i<<": ";
				for(unsigned int j=0; j<DpL[layer][i].size(); ++j)
				{
					std::cout<<DpL[layer][i][j]<<" ("<<j<<")"<<" ";
				}
				std::cout<<"\n";
			}
		}
	}
	
	if(grad.size()>0)
	{
		std::cout<<"\n";
		for(unsigned int layer=0; layer<grad.size(); ++layer)
		{
			std::cout<<"Last gradient of layer "<<layer<<": ";
			for(unsigned int i=0; i<grad[layer].size(); ++i)
			{
				std::cout<<grad[layer][i]<<" ("<<i<<")"<<" ";
			}
			std::cout<<"\n";
		}
	}

	std::cout<<"================================================== \n"<<std::endl;
}

unsigned int Ann::train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs,
						const std::vector<std::vector<double>> &vInputs, const std::vector<std::vector<double>> &vOutputs,
						bool verbose, double minError, unsigned int iterations,
						double N, double B)
{
	unsigned int mainIt = 0;
	std::vector<double> buffer = std::vector<double>(NpL[NpL.size()-1], 0.0);
	
	grad.clear();
	grad.push_back(std::vector<double>());
	for(unsigned int i=1; i<results.size(); ++i)
	{
		grad.push_back(std::vector<double>(results[i].size(),0.0));
	}
	
	DpL.clear();
	for(unsigned int layer=0; layer<WpL.size(); ++layer)
	{
		DpL.push_back(std::vector<std::vector<double>>());
		for(unsigned int i=0; i<WpL[layer].size(); ++i)
		{
			DpL[layer].push_back(std::vector<double>(WpL[layer][i].size(),0.0));
		}
	}
	
    for(; mainIt<iterations; ++mainIt)
    {
        for(unsigned int sample=0; sample<inputs.size(); ++sample)
        {
			// forward pass
			use(inputs[sample],buffer);
			
            // backpropagation
            for(unsigned int i=0; i<NpL[NpL.size()-1]; ++i)
            {
                grad[grad.size()-1][i] = (buffer[i] - outputs[sample][i]) * buffer[i]*(1-buffer[i]);
				//std::cout<<grad[grad.size()-1][i]<<" "<<buffer[i] - outputs[sample][i]<<" "<<buffer[i]<<" ";
            }//std::cout<<" grad\n";
			
            for(unsigned int layer=NpL.size()-2; layer>0; --layer)
            {
				for(unsigned int i=0; i<grad[layer].size(); ++i)
				{
					grad[layer][i] = GradReLU(results[layer][i]);
					if(grad[layer][i]>0)
					{
						grad[layer][i] = 0;
						for(unsigned int j=0; j<grad[layer+1].size(); ++j)
						{
							grad[layer][i] += grad[layer+1][j]*WpL[layer][i][j];
						}
					}
				}
            }
			
			// gradient descent
			for(unsigned int layer=0; layer<WpL.size(); ++layer)
			{
				for(unsigned int i=0; i<WpL[layer].size(); ++i)
				{
					for(unsigned int j=0; j<WpL[layer][i].size(); ++j)
					{
						const double delta = DpL[layer][i][j] - (grad[layer+1][j]*results[layer][i]);
						//std::cout<<delta<<" "<<DpL[layer][i][j]<<" "<<grad[layer+1][j]<<" "<<results[layer][i]<<"\n";
                        WpL[layer][i][j] += N*delta;
                        DpL[layer][i][j] = B*delta;
					}
				}
			}
		}
		
		if(mainIt%10==0)
        {
			const double error = getError(vInputs,vOutputs);
			if(verbose)
			{
				std::cout<<mainIt<<": "<<error<<"\n";
			}
            if(error<minError)
            {
                break;
            }
        }
    }
	
	if(verbose)
	{
		std::cout<<"\n";
	}

    return mainIt;
}

double Ann::getError(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs)
{
    double total = 0;
	std::vector<double> buffer = std::vector<double>(NpL[NpL.size()-1], 0.0);
    
	for(unsigned int i=0; i<inputs.size(); ++i)
    {
		use(inputs[i],buffer);
		for(unsigned int node=0; node<buffer.size(); ++node)
		{
			const double error = buffer[node]-outputs[i][node];
			total += error*error;
		}
    }
    
	return 0.5*total/inputs.size();
}

void Ann::use(const std::vector<double> &inputs, std::vector<double> &outputs)
{
    for(unsigned int i=0; i<inputs.size(); ++i)
    {
        results[0][i] = inputs[i];
    }
	
	// Could be optimized by looping in a more cache-friendly order
    for(unsigned int layer=1; layer<(results.size()-1); ++layer)
	{
		for(unsigned int to=0; to<(results[layer].size()-1); ++to)
		{
			results[layer][to] = 0;
			for(unsigned int from=0; from<results[layer-1].size(); ++from)
			{
				results[layer][to] += results[layer-1][from]*WpL[layer-1][from][to];
			}
			results[layer][to] = ReLU(results[layer][to]);
		}
	}
	
	// Could be optimized by looping in a more cache-friendly order
	const unsigned int layer = NpL.size()-1;
	for(unsigned int to=0; to<results[layer].size(); ++to)
	{
		results[layer][to] = 0;
		for(unsigned int from=0; from<results[layer-1].size(); ++from)
		{
			results[layer][to] += results[layer-1][from]*WpL[layer-1][from][to];
		}
		results[layer][to] = Sigmoid(results[layer][to]);
	}

    for(unsigned int i=0; i<outputs.size(); ++i)
    {
        outputs[i] = results[results.size()-1][i];
    }
}
