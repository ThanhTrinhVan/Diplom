#pragma once
#include <opencv2/opencv.hpp> // библиотека opencv
#include <vector>
#include <numeric> // содержит функцию вычисления среднее значение списка accumulate
#include <math.h>

using namespace cv;
typedef Point_<double> PointCenter; // для сохранения центр изображения в виде double

class Preprocess // нормализация изображения
{
public:
	Preprocess();
	Preprocess(Mat &src_img); // конструктор
	~Preprocess();

	Mat* processing(); // предварительная обработка изображения
	void updateImage(Mat &new_img); // обновления src

private:
	double getThreshValue(Mat* img); // среднее значение яркости изображения img
	Mat* rotateSrc(Mat* img, double &angle, PointCenter &center); // вращать img на угол angle вокруг center
	Mat* cropping(Mat * img, Mat* InfoArea, PointCenter &center); // обрезать информационную часть с размером 300x100 из img

	Mat* src; // исходное изображение
	Mat* dst; // результат нормализации
	
	const Scalar color_red = Scalar(0, 0, 255);
	const Scalar color_blue = Scalar(255, 0, 0);
};

