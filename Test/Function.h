#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace std;

class Diagnostic {
public:
	Diagnostic();
	Diagnostic(int k_, int nd_);
	~Diagnostic();

	void reset() {
		matrixA.clear();
		vectorB.clear();
		sA.clear();
		sB = 0;
		skoA.clear();
		skoB = 0;
		mf.clear();
		mfComponent.clear();
		votingComponentAfter.clear();
		votingComponentBefore.clear();
		votingParameter.clear();
		votingGlobal.clear();
	}

	void loadData(string path, vector<double> noise, double sko);
	void printData();
	void calculateS();
	void calculateSKO();
	void membershipFunction();
	int predict();
private:
	vector<vector<double>> matrixA;
	vector<double> vectorB;
	
	vector<double> sA;
	double sB = 0;
	double skoB = 0;
	vector<double> skoA;

	vector<double> mf;
	vector<vector<double>> mfComponent;
	vector<vector<int>> votingParameter;
	vector<vector<int>> votingComponentBefore;
	vector<vector<int>> votingComponentAfter;
	vector<int> votingGlobal;

	const double BT = 0.95; // коэффициент для порога бинаризации
	const int NP = 24;		// количество параметров Харалика
	int k;					// номер заболевания на входе
	int nd;					// количество болезней, имеющихсь в базе данных
	int method = 2;			// 1 or dif
	bool testing = false;   // использовать шум ли
	char nameFeatures[4][20] = { "Contrast: ", "Correlation: ", "Energy: ", "Homogeneity: " };
	char nameComponent[6][5] = { "R ", "G ", "B ", "RG ", "RB ", "GB " };
};