#pragma once
#ifndef FUNCTION_H
#define FUNCTION_H

#include "iostream"
#include "stdlib.h"
#include "stdio.h"
#include <vector>
#include <string>
#include "opencv.hpp"
#include "highgui.hpp"

using namespace std;

struct package
{
	vector<vector<float>> data;
	int howMany = 0;
};

struct cell
{
	vector<float> feature;
	int label;
};


void SaveCodeBook(vector<vector<float>>data);
bool RestoreFromDisk(vector<vector<float>>& codebook);


cv::Mat CreateTrainData(bool blur, int N, int low, int high, bool debug);
vector<vector<float>> BuildCodeBook(cv::Mat data, int size, bool debug);
cell ComputeFeatureForOneImage(string path, int label, bool blur, vector<vector<float>> codeBook, int flag, bool debug);
vector<cell> GenerateCell(vector<vector<float>> codebook, bool blur, int N, int low, int high, bool debug);
void GatherTrainAccuracy(vector<cell> ref, vector<cell> unknown, int flag);
void GatherTestAccuracy(vector<cell> ref, vector<cell> unknown, int flag);



#endif // !FUNCTION_H

