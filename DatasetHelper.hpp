#ifndef __DATASET_HELPER_HPP__
#define __DATASET_HELPER_HPP__


#include <iostream>
#include <fstream>
#include <random>
#include <iterator>
#include <opencv2/core.hpp>

#define BOAT_TARGET 1
#define SEA_TARGET 2
#define BG_TARGET 3


#define BG_1H_ENC 0x4 // 100 in binary
#define BOATS_1H_ENC  0x2 // 010 in binary
#define SEA_1H_ENC 0x1 // 001 in binary

void saveDataset(
					const std::string& name,
					std::vector<std::vector<double>>& descriptors
				);

void appendDescriptors(std::vector<std::vector<double>>& vect, const cv::Mat& descriptors, char oneHotEnc);

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
				);

#endif