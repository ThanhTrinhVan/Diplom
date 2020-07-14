// ProgramDiagnostic.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Function.h"
#include <random>

int k = 2;				// номер заболевания на входе 9 - unknown
int kb_in = pow(2, k);  // входной код заболевания (нужен для определения правильности диагностики)
int m = 1;			// количество статистических испытаний. 
double S = 0.05;		// СКО случайных отклонений от эталонных значений method 1 SKO = 0.05, method 2 SKO = 0.04

int main()
{
	Diagnostic* dig = new Diagnostic(k, 8);
	int nCorrect = 0;
	for (int c = 0; c < m; c++) {
		dig->reset();
		vector<double> noise;
		random_device rd;
		mt19937 gen(rd());
		normal_distribution<double> d(0.0, 1.0);
		for (int i = 0; i < 24; i++)
			noise.push_back(d(gen));
		dig->loadData("data.csv", noise, S);
		dig->calculateS();
		dig->calculateSKO();
		dig->membershipFunction();
		dig->printData();
		int k_out = dig->predict();
		if (k_out == k)
			nCorrect++;
	}
	cout << "Accuracy: " << (double)(nCorrect*100.0 / m) << endl;
	system("pause");
	return 0;
}