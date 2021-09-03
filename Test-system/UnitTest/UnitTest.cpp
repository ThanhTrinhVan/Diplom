#include "pch.h"
#include "CppUnitTest.h"
#include "../NormImage/Preprocess.h"
#include "../NormImage/Preprocess.cpp"
#include "../NormImage/GLCM.h"
#include "../NormImage/GLCM.cpp"
#include "../NormImage/Classifier.h"
#include "../NormImage/Classifier.cpp"
#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace cv;

namespace UnitTest
{
	TEST_CLASS(WatershedSegmenter_Test)
	{
	public:
		
		TEST_METHOD(setMarkers)
		{
			WatershedSegmenter segmenter;
			Mat markers = Mat::zeros(Size(10, 10), CV_8U);
			Assert::AreEqual(segmenter.setMarkers(markers), false);
			markers = Mat(Size(10, 10), CV_8U, Scalar(-1));
			//top rectangle
			markers(Rect(0, 0, 10, 5)) = Scalar::all(1);
			//bottom rectangle
			markers(Rect(0, 10 - 5, 10, 5)) = Scalar::all(1);
			//left rectangle
			markers(Rect(0, 0, 5, 10)) = Scalar::all(1);
			//right rectangle
			markers(Rect(10 - 5, 0, 5, 10)) = Scalar::all(1);
			//centre rectangle
			markers(Rect((10 / 2) - (5 / 2), (10 / 2) - (5 / 2), 5, 5)) = Scalar::all(2);
			markers.convertTo(markers, CV_BGR2GRAY);
			Assert::AreEqual(segmenter.setMarkers(markers), true);
		}
		TEST_METHOD(process)
		{
			bool pass = true;
			WatershedSegmenter segmenter;
			Mat test = Mat::zeros(Size(0, 0), CV_8U);
			Mat res = segmenter.process(test);
			bitwise_xor(res, test, res);
			if (countNonZero(res) > 0)
				pass = false;
			Assert::AreEqual(pass, true);
		}
	};
	TEST_CLASS(Preprocess_Test)
	{
	public:

		TEST_METHOD(Processing_Test)
		{
			Mat res = imread("I:/GradEs/Testing/unittest.png");
			Preprocess* pr = new Preprocess();
			Mat img = imread("I:/GradEs/300_Photos/Brown_rust/Screen Shot 2018-10-07 at 19.28.02.png");
			pr->updateImg(img);
			img = *(pr->processing());
			Assert::AreEqual(img.rows, 300);
			Assert::AreEqual(img.cols, 100);
			cvtColor(img, img, COLOR_BGR2GRAY);
			cvtColor(res, res, COLOR_BGR2GRAY);
			bitwise_xor(img, res, img);
			bool passed = true;
			if (countNonZero(img) > 0)
				passed = false;
			delete pr;
			Assert::AreEqual(passed, true);
		}
		TEST_METHOD(UpdateImg_Test)
		{
			Preprocess* pr = new Preprocess();
			Mat img = imread("I:/GradEs/300_Photos/Septoria/Septoria-tritici (1).jpg");
			pr->updateImg(img);
			Mat src = *(pr->getSourceImg());
			cvtColor(img, img, COLOR_BGR2GRAY);
			cvtColor(src, src, COLOR_BGR2GRAY);
			bitwise_xor(img, src, img);
			bool passed = true;
			if (countNonZero(img) > 0)
				passed = false;
			delete pr;
			Assert::AreEqual(passed, true);
		}
		TEST_METHOD(GetAvgPixel_Test)
		{
			Mat M(2, 2, CV_8UC1, Scalar(10));
			Assert::AreEqual(getAvgPixel(M), 10.0);
		}
	};
	TEST_CLASS(GLCM_Test)
	{
	public:
		TEST_METHOD(Reset_Test)
		{
			GLCM* glcm = new GLCM();
			Assert::AreEqual(glcm->getGLevel(), 8);
			glcm = new GLCM(GRAY_8);
			Mat M(2, 2, CV_8UC1, Scalar(10));
			glcm->calculateMatrix(M, 2, 0);
			glcm->normalizeMatrix();
			glcm->calculateFeatures();
			glcm->reset();
			double** m = glcm->getMatrix();
			double*  f = glcm->getFeatures_Haralick();
			bool passed = true;
			for(int i=0; i<6; i++)
				if (f[i] != 0) {
					passed = false;
					break;
				}
			Assert::AreEqual(passed, true);
			for(int i=0; i<8; i++)
				for(int j=0; j<8; j++)
					if (m[i][j] != 0) {
						passed = false;
						break;
					}
			Assert::AreEqual(passed, true);
			delete glcm;
		}
		TEST_METHOD(GetImgByChannel_Test)
		{
			Mat img = imread("I:/GradEs/Testing/unittest.png");
			Mat bgr[3];
			split(img, bgr);
			Mat rg = bgr[2] - bgr[1];
			Mat rb = bgr[2] - bgr[0];
			Mat gb = bgr[1] - bgr[0];
			GLCM* glcm = new GLCM();
			Assert::AreEqual(isSame(bgr[0], glcm->getImgByChannel(img, CHANNEL_B)), true);
			Assert::AreEqual(isSame(bgr[1], glcm->getImgByChannel(img, CHANNEL_G)), true);
			Assert::AreEqual(isSame(bgr[2], glcm->getImgByChannel(img, CHANNEL_R)), true);
			Assert::AreEqual(isSame(rg, glcm->getImgByChannel(img, CHANNEL_RG)), true);
			Assert::AreEqual(isSame(rb, glcm->getImgByChannel(img, CHANNEL_RB)), true);
			Assert::AreEqual(isSame(gb, glcm->getImgByChannel(img, CHANNEL_GB)), true);
			delete glcm;
		}
		TEST_METHOD(CalculateMatrix_Test)
		{
			Mat img = imread("I:/GradEs/Testing/unittest.png");
			GLCM* glcm = new GLCM(GRAY_8);
			glcm->reset();
			img = glcm->getImgByChannel(img, CHANNEL_GB);
			glcm->calculateMatrix(img, 2, 0);
			double** m = glcm->getMatrix();
			glcm->printMatrix();
			double res[8][8] = { { 5395, 705, 0, 0, 0, 0, 0, 0},
				{ 817, 14884, 1564, 0, 0, 0, 0, 0},
				{ 0, 1639, 4796, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0} };
			double dif = 0;
			for (int i = 0; i < 8; i++)
				for (int j = 0; j < 8; j++)
					dif += abs(m[i][j] - res[i][j]);
			delete glcm;
			Assert::AreEqual(dif, 0.0);
		}
		TEST_METHOD(NormalizeMatrix_Test)
		{
			Mat img = imread("I:/GradEs/Testing/unittest.png");
			GLCM* glcm = new GLCM();
			glcm->reset();
			img = glcm->getImgByChannel(img, CHANNEL_GB);
			glcm->calculateMatrix(img, 2, 0);
			glcm->normalizeMatrix();
			double** m = glcm->getMatrix();
			glcm->printMatrix();
			double res[8][8] = { { 0.18104, 0.0236577, 0, 0, 0, 0, 0, 0},
				{ 0.0274161, 0.499463, 0.0524832, 0, 0, 0, 0, 0},
				{ 0, 0.055, 0.16094, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0},
				{ 0, 0, 0, 0, 0, 0, 0, 0} };
			double dif = 0;
			for (int i = 0; i < 8; i++)
				for (int j = 0; j < 8; j++)
					dif += abs(m[i][j] - res[i][j]);
			delete glcm;
			Assert::AreEqual((dif < 0.00001), true);
		}
		TEST_METHOD(CalculateFeatures_Test)
		{
			Mat img = imread("I:/GradEs/Testing/unittest.png");
			std::ofstream outFile;
			outFile.open("I:/GradEs/Testing/unittest.csv");
			GLCM* glcm = new GLCM();
			glcm->reset();
			img = glcm->getImgByChannel(img, CHANNEL_GB);
			glcm->calculateMatrix(img, 2, 0);
			glcm->normalizeMatrix();
			glcm->calculateFeatures();
			double* f = glcm->getFeatures_Haralick();
			glcm->printFeatures_Haralick(outFile);
			double res[6] = { 0.158557, 0.81182, 0.315231, 0.920721, 0.630388 , 0.920721 };
			double dif = 0;
			for (int i = 0; i < 6; i++)
					dif += abs(f[i] - res[i]);
			delete glcm;
			Assert::AreEqual((dif < 0.00001), true);
		}
	private:
		bool isSame(Mat img1, Mat img2) 
		{
			bitwise_xor(img1, img2, img1);
			if (countNonZero(img1) > 0)
				return false;
			return true;
		}
	};
	TEST_CLASS(Classifier_Test)
	{
	public:

		TEST_METHOD(Reset)
		{
			Classifier* dig = new Classifier();
			dig->reset();
			Assert::AreEqual(dig->getS_B(), 0.0);
			Assert::AreEqual(dig->getSKO_B(), 0.0);
			delete dig;
		}
		TEST_METHOD(LoadData)
		{
			int k = 2;
			Classifier* dig = new Classifier(k, 8);
			dig->reset();
			vector<double> noise;
			random_device rd;
			mt19937 gen(rd());
			normal_distribution<double> d(0.0, 1.0);
			for (int i = 0; i < 24; i++)
				noise.push_back(d(gen));
			dig->loadData("I:/GradEs/Program/System/NormImage/data.csv", noise, 0.05);
			double vb[24] = { 0.00337584, 0.96374 ,0.278344 ,0.92207, 0.00352966 ,0.968015,
				0.252569 ,0.919038, 0.00291303, 0.951104, 0.323066, 0.931105, 0.000238734,
				1, 0.910848, 0.994151,0.0013376, 0.949747,0.459469,0.967246, 0.00126435,
				0.954003 , 0.44933, 0.969033 };
			double dif = 0;
			vector<double> v_b = dig->getVectorB();
			for (int i = 0; i < 24; i++)
				dif += abs(vb[i] - v_b[i]);
			Assert::AreEqual((dif < 0.0001), true);
			delete dig;
		}
		TEST_METHOD(CalculateS)
		{
			int k = 2;
			Classifier* dig = new Classifier(k, 8);
			dig->reset();
			vector<double> noise;
			random_device rd;
			mt19937 gen(rd());
			normal_distribution<double> d(0.0, 1.0);
			for (int i = 0; i < 24; i++)
				noise.push_back(d(gen));
			dig->loadData("I:/GradEs/Program/System/NormImage/data.csv", noise, 0.05);
			dig->calculateS();
			vector<double> s_a = dig->getVectorS_A();
			double sa[8] = { 0.513068, 0.595236, 0.558783, 0.576086, 0.548891, 0.58094, 0.560587, 0.535506 };
			double dif = 0;
			for (int i = 0; i < 8; i++)
				dif += abs(sa[i] - s_a[i]);
			Assert::AreEqual((dif < 0.00001), true);
			Assert::AreEqual((abs(dig->getS_B() - 0.590647) < 0.00001), true);
			delete dig;
		}
		TEST_METHOD(CalculateSKO)
		{
			int k = 2;
			Classifier* dig = new Classifier(k, 8);
			dig->reset();
			vector<double> noise;
			random_device rd;
			mt19937 gen(rd());
			normal_distribution<double> d(0.0, 1.0);
			for (int i = 0; i < 24; i++)
				noise.push_back(d(gen));
			dig->loadData("I:/GradEs/Program/System/NormImage/data.csv", noise, 0.05);
			dig->calculateS();
			dig->calculateSKO();
			vector<double> s_a = dig->getVectorSKO_A();
			double sa[8] = { 3.60155, 4.10371, 4.21151, 4.2702, 3.836, 4.27975, 4.27114, 3.91418 };
			double dif = 0;
			for (int i = 0; i < 8; i++)
				dif += abs(sa[i] - s_a[i]);
			Assert::AreEqual((dif < 0.001), true);
			Assert::AreEqual((abs(dig->getSKO_B() - 4.1229) < 0.001), true);
			delete dig;
		}
		TEST_METHOD(Predict)
		{
			int k = 2;
			Classifier* dig = new Classifier(k, 8);
			dig->reset();
			vector<double> noise;
			random_device rd;
			mt19937 gen(rd());
			normal_distribution<double> d(0.0, 1.0);
			for (int i = 0; i < 24; i++)
				noise.push_back(d(gen));
			dig->loadData("I:/GradEs/Program/System/NormImage/data.csv", noise, 0.05);
			dig->calculateS();
			dig->calculateSKO();
			dig->membershipFunction();
			dig->printData();
			Assert::AreEqual(dig->predict(), k);
			delete dig;
		}
	};
}
