#include "function.h"


using namespace std;


void main(void) 
{
	vector<vector<float>> codeBook;
	if (!RestoreFromDisk(codeBook)) 
	{
		cout << "generate train data..." << endl;
		cv::Mat trainData = CreateTrainData(true, 40, 1, 7, true);
		cout << "generate codeBook..." << endl;
		cout << "this gonna take a while...extremely long :( " << endl;
		codeBook = BuildCodeBook(trainData, 174, true);
		cout << "build histogram..." << endl;
		SaveCodeBook(codeBook);
	}
	cout << "generate feature and label... " << endl;
	vector<cell> trainCluster = GenerateCell(codeBook, true, 40, 1, 7, true);
	vector<cell> testCluster = GenerateCell(codeBook, true, 40, 8, 10, true);
	cout << "gather train accuracy..." << endl;
	GatherTrainAccuracy(trainCluster, trainCluster, 0);
	GatherTestAccuracy(trainCluster, testCluster, 0);
	while (1) 
	{
		cv::waitKey(50);
	}

}