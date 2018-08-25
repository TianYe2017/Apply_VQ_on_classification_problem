#include "function.h"

vector<vector<float>> ProcessSingleImage(string path, bool blur, bool debug)
{
	cv::Mat img = cv::imread(path);
	cv::cvtColor(img,img, CV_RGB2GRAY);
	if (blur) 
	{
		cv::blur(img, img, cv::Size(3, 3));
	}
	img.convertTo(img, CV_32F);
	for (int row = 0; row < img.rows; row++) 
	{
		for (int col = 0; col < img.cols; col++) 
		{
			img.at<float>(row, col) /= 255.0f;
		}
	}
	vector<vector<float>> cluster;
	vector<float> tmp;
	for (int row = 0; row < img.rows - 1; row++) 
	{
		for (int col = 0; col < img.cols - 1; col++)
		{
			tmp.clear();
			tmp.push_back(img.at<float>(row, col));
			tmp.push_back(img.at<float>(row, col + 1));
			tmp.push_back(img.at<float>(row + 1, col));
			tmp.push_back(img.at<float>(row + 1, col + 1));
			float min = 2.0f;
			for (int k = 0; k < 4; k++) 
			{
				if (tmp[k] < min) 
				{
					min = tmp[k];
				}
			}
			for (int k = 0; k < 4; k++)
			{
				tmp[k] -= min;
			}
			cluster.push_back(tmp);
		}
	}
	if (debug) 
	{
		cout << "path: " << path << endl;
		cout << "rows: " << img.rows << " " << "cols: " << img.cols << " " << "SizeOfCluster: " << cluster.size() << " " << cluster[0].size() << endl;
		cout << "" << endl;
	}
	return cluster;
}

cv::Mat CreateTrainData(bool blur, int N, int low, int high, bool debug) 
{
	cv::Mat ans;
	vector<vector<float>> cluster;
	for (int s = 1; s <= N; s++)
	{
		for (int i = low; i <= high; i++)
		{
			string path = "./att_faces/s";
			path += to_string(s);
			path += "/";
			path += to_string(i);
			path += ".pgm";
			vector<vector<float>> buffer = ProcessSingleImage(path, blur, debug);
			for (int i = 0; i < buffer.size(); i++) 
			{
				cluster.push_back(buffer[i]);
			}
		}
	}
	int width = cluster[0].size();
	int height = cluster.size();
	ans = cv::Mat(cv::Size(width, height), CV_32F);
	for (int row = 0; row < height; row++) 
	{
		for (int col = 0; col < width; col++) 
		{
			ans.at<float>(row, col) = cluster[row][col];
			//cout << ans.at<float>(row, col) <<endl;
		}
	}
	return ans;
}

vector<vector<float>> BuildCodeBook(cv::Mat data,int size, bool debug) 
{
	vector<vector<float>> codeBook;
	cv::Mat centers, labels;
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 0.1);
	cv::kmeans(data, size, labels, criteria, 3, 2, centers);
	vector<float>tmp;
	for (int i = 0; i < size; i++)
	{
		tmp.clear();
		for (int j = 0; j < 4; j++)
		{
			tmp.push_back(centers.at<float>(i, j));
		}
		codeBook.push_back(tmp);
	}
	if (debug) 
	{
		for (int i = 0; i < size; i++) 
		{
			for (int j = 0; j < 4; j++) 
			{
				cout << codeBook[i][j] << " ";
			}
			cout << "" << endl;
		}
		
	}
	cout << "sizeOfCodeBook: " << size << endl;
	return codeBook;
}

float L1Distance(vector<float> a, vector<float> b) 
{
	float sum = 0.0f;
	for (int i = 0; i < a.size(); i++) 
	{
		sum += abs(a[i] - b[i]);
	}
	return sum;
}

float L2Distance(vector<float> a, vector<float> b)
{
	float sum = 0.0f;
	for (int i = 0; i < a.size(); i++)
	{
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	sum = sqrt(sum);
	return sum;
}


void SaveCodeBook(vector<vector<float>>data) 
{
	int rows = data.size();
	int cols = data[0].size();
	int total = rows * cols + 2;
	unsigned char* buffer = (unsigned char*)malloc(sizeof(unsigned char) * total * 4);
	unsigned char tmp[4];
	memcpy(tmp, &rows, 4);
	buffer[0] = tmp[0];
	buffer[1] = tmp[1];
	buffer[2] = tmp[2];
	buffer[3] = tmp[3];
	memcpy(tmp, &cols, 4);
	buffer[4] = tmp[0];
	buffer[5] = tmp[1];
	buffer[6] = tmp[2];
	buffer[7] = tmp[3];

	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			float w = data[i][j];
			memcpy(tmp, &w, 4);
			buffer[8 + (i * cols + j) * 4] = tmp[0];
			buffer[8 + (i * cols + j) * 4 + 1] = tmp[1];
			buffer[8 + (i * cols + j) * 4 + 2] = tmp[2];
			buffer[8 + (i * cols + j) * 4 + 3] = tmp[3];
		}
	}
	FILE* p = fopen("./codebook/codebook.dat","wb");
	fwrite(buffer,sizeof(unsigned char),total*4,p);
	fclose(p);
}

bool RestoreFromDisk(vector<vector<float>>& codebook) 
{
	FILE* p = fopen("./codebook/codebook.dat", "rb");
	if (p == NULL) 
	{
		return false;
	}
	unsigned char tmp[4];
	fread(tmp, sizeof(unsigned char), 4, p);
	int rows, cols;
	memcpy(&rows, tmp, 4);
	fread(tmp, sizeof(unsigned char), 4, p);
	memcpy(&cols, tmp, 4);
	for (int i = 0; i < rows; i++) 
	{
		vector<float> buffer;
		for (int j = 0; j < cols; j++) 
		{
			float f;
			fread(tmp, sizeof(unsigned char), 4, p);
			memcpy(&f, tmp, 4);
			buffer.push_back(f);
		}
		codebook.push_back(buffer);
	}
	fclose(p);
	return true;
}



cell ComputeFeatureForOneImage(string path, int label, bool blur, vector<vector<float>> codeBook, int flag, bool debug) 
{
	cell c;
	c.label = label;
	vector<float> feature;
	for (int i = 0; i < codeBook.size(); i++) 
	{
		feature.push_back(0.0f);
	}
	vector<vector<float>> cluster = ProcessSingleImage(path, blur, debug);
	
	for (int i = 0; i < cluster.size(); i++) 
	{	
		float min_dis = 99999.0f;
		int index = -1;
		float dis;
		for (int j = 0; j < codeBook.size(); j++) 
		{
			if (flag == 0) 
			{
				dis = L2Distance(codeBook[j],cluster[i]);
			}
			else 
			{
				dis = L1Distance(codeBook[j], cluster[i]);
			}
			if (dis < min_dis) 
			{
				min_dis = dis;
				index = j;
			}
		}
		feature[index] += 1.0f;
	}
	for (int i = 0; i < codeBook.size(); i++) 
	{
		feature[i] /= (float)cluster.size();
	}
	c.feature = feature;
	if (debug) 
	{
		cout << "label: " << c.label << " " << "feature: " << endl;
		for (int i = 0; i < codeBook.size(); i++) 
		{
			cout << c.feature[i] << " ";
		}
		cout << "" << endl;
	}
	return c;
}

vector<cell> GenerateCell(vector<vector<float>> codebook,bool blur, int N, int low, int high, bool debug) 
{
	vector<cell> cluster;
	for (int s = 1; s <= N; s++)
	{
		for (int i = low; i <= high; i++)
		{
			string path = "./att_faces/s";
			path += to_string(s);
			path += "/";
			path += to_string(i);
			path += ".pgm";
			cluster.push_back(ComputeFeatureForOneImage(path, s, blur, codebook, 0, debug));
		}
	}
	return cluster;
}

void GatherTestAccuracy(vector<cell> ref, vector<cell> unknown, int flag)
{
	float accuracy;
	float numOfPos = 0.0f;
	for (int i = 0; i < unknown.size(); i++)
	{
		int real_label = unknown[i].label;
		int label = ref[0].label;
		float minDis;
		float dis;
		if (flag == 0)
		{
			minDis = L2Distance(ref[0].feature, unknown[i].feature);
		}
		else
		{
			minDis = L1Distance(ref[0].feature, unknown[i].feature);
		}
		for (int j = 1; j < ref.size(); j++)
		{
			if (flag == 0)
			{
				dis = L2Distance(ref[j].feature, unknown[i].feature);
			}
			else
			{
				dis = L1Distance(ref[j].feature, unknown[i].feature);
			}
			if (dis < minDis)
			{
				minDis = dis;
				label = ref[j].label;
			}
		}
		cout << "real label: " << real_label << "  " << "predicted label: " << label << "  " << "distance: " << minDis << endl;
		if (real_label == label)
		{
			numOfPos += 1.0f;
		}
	}
	accuracy = numOfPos / ((float)unknown.size());
	cout << "Total number of test cases:(test) " << unknown.size() << endl;
	cout << "# correct: " << (int)numOfPos << "  " << "test accuracy: " << accuracy << endl;
}

void GatherTrainAccuracy(vector<cell> ref, vector<cell> unknown, int flag)
{
	float accuracy;
	float numOfPos = 0.0f;
	for (int i = 0; i < unknown.size(); i++)
	{
		int real_label = unknown[i].label;
		int label = ref[0].label;
		float minDis;
		float dis;
		if (flag == 0)
		{
			minDis = L2Distance(ref[0].feature, unknown[i].feature);
		}
		else
		{
			minDis = L1Distance(ref[0].feature, unknown[i].feature);
		}
		for (int j = 1; j < ref.size(); j++)
		{
			if (flag == 0)
			{
				dis = L2Distance(ref[j].feature, unknown[i].feature);
			}
			else
			{
				dis = L1Distance(ref[j].feature, unknown[i].feature);
			}
			if (dis < minDis)
			{
				minDis = dis;
				label = ref[j].label;
			}
		}
		cout << "real label: " << real_label << "  " << "predicted label: " << label << "  " << "distance: " << minDis << endl;
		if (real_label == label)
		{
			numOfPos += 1.0f;
		}
	}
	accuracy = numOfPos / ((float)unknown.size());
	cout << "Total number of test cases:(train) " << unknown.size() << endl;
	cout << "# correct: " << (int)numOfPos << "  " << "train accuracy: " << accuracy << endl;
}






