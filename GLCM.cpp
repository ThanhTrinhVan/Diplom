#include "GLCM.h"



GLCM::GLCM()
{
	G_level = 8;
	direct = DIRECTION_0;
}

GLCM::GLCM(GrayLevel level_, Direction direct_)
{
	switch (level_)
	{
	case GRAY_4:
		G_level = 4;
		break;
	case GRAY_8:
		G_level = 8;
		break;
	case GRAY_16:
		G_level = 16;
		break;
	default:
		G_level = 8;
		break;
	}
	direct = direct_;
}

GLCM::~GLCM()
{
	if (matrix) {
		for (i = 0; i < G_level; i++)
			delete matrix[i];
		delete matrix;
	}
	if (features)
		delete features;
}

void GLCM::calculateMatrix(Mat & img)
{
	if (matrix) {
		for (i = 0; i < G_level; i++)
			delete matrix[i];
		delete matrix;
	}
	
	matrix = new double *[G_level];
	for (i = 0; i < G_level; i++) {
		matrix[i] = new double[G_level];
	}

	sumValue = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			matrix[i][j] = countValue(img, i, j);
			sumValue += matrix[i][j];
		}
	}
}

void GLCM::calculateFeatures()
{
	if (features)
		delete features;
	features = new double[6];

	// contrast
	features[0] = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			features[0] += pow((i - j), 2)*matrix[i][j];
		}
	}
	//features[0] /= pow(G_level - 1, 2);
	// correlation
	features[1] = 0;
	double* mu_u = new double[G_level + 1];
	double* si_u = new double[G_level + 1];
	double* mu_v = new double[G_level + 1];
	double* si_v = new double[G_level + 1];

	mu_u[G_level] = 0;
	for (i = 0; i < G_level; i++) {
		mu_u[i] = 0.0;
		for (j = 0; j < G_level; j++) {
			mu_u[i] += i * matrix[i][j];
		}
		mu_u[G_level] += mu_u[i];
	}

	mu_v[G_level] = 0;
	for (j = 0; j < G_level; j++) {
		mu_v[j] = 0;
		for (i = 0; i < G_level; i++) {
			mu_v[j] += j * matrix[i][j];
		}
		mu_v[G_level] += mu_v[j];
	}

	si_u[G_level] = 0;
	for (i = 0; i < G_level; i++) {
		si_u[i] = 0;
		for (j = 0; j < G_level; j++) {
			si_u[i] += matrix[i][j] * pow((i - mu_u[G_level]), 2);
		}
		//si_u[i] = sqrt(si_u[i]);
		si_u[G_level] += si_u[i];
	}

	si_v[G_level] = 0;
	for (j = 0; j < G_level; j++) {
		si_v[j] = 0;
		for (i = 0; i < G_level; i++) {
			si_v[j] += matrix[i][j] * pow((j - mu_v[G_level]), 2);
		}
		//si_v[j] = sqrt(si_v[j]);
		si_v[G_level] += si_v[j];
	}

	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			if (matrix[i][j] != 0)
				features[1] += matrix[i][j] * (i - mu_u[G_level])*(j - mu_v[G_level]) / sqrt(si_u[G_level] * si_v[G_level]);
		}
	}
	//features[1] = (features[1] + 1) / 2;
	// energy
	features[2] = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			features[2] += pow(matrix[i][j], 2);
		}
	}
	// homogeneity
	features[3] = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			features[3] += matrix[i][j] / (1 + abs(i - j));
		}
	}
	// entropy
	features[4] = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			if (matrix[i][j] != 0)
				features[4] += matrix[i][j] * log10(matrix[i][j]);
		}
	}
	features[4] = -features[4];
	// Inverse Difference Moment
	features[5] = 0;
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			features[5] += matrix[i][j] / (1 + pow(i - j, 2));
		}
	}
	delete mu_u, mu_v, si_u, si_v;
}

void GLCM::normalizeMatrix()
{
	for (i = 0; i < G_level; i++) {
		for (j = 0; j < G_level; j++) {
			matrix[i][j] /= sumValue;
		}
	}
}

Mat GLCM::getImgByChannel(Mat & img, ChannelRGB cn)
{
	// Component RG, RB, GB
	Mat bgr_2[3];
	for (i = 0; i < 3; i++) {
		// Component R, G, B
		Mat bgr[3];
		split(img, bgr);
		bgr[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1); // set 1 chanel to 0
		cv::merge(bgr, 3, bgr_2[i]); // merging 2 channels
	}
	/*
	imshow("R+G", bgr_2[0]);
	imshow("R+B", bgr_2[1]);
	imshow("G+B", bgr_2[2]);
	*/
	// Component R, G, B
	Mat bgr[3];
	split(img, bgr);
	/*
	imshow("Blue", bgr[0]);
	imshow("Green", bgr[1]);
	imshow("Red", bgr[2]);
	*/
	switch (cn)
	{
	case CHANNEL_R:
		return bgr[2];
		break;
	case CHANNEL_G:
		return bgr[1];
		break;
	case CHANNEL_B:
		return bgr[0];
		break;
	case CHANNEL_RG:
		cvtColor(bgr_2[0], bgr_2[0], CV_RGB2GRAY);
		return bgr_2[0];
		break;
	case CHANNEL_RB:
		cvtColor(bgr_2[1], bgr_2[1], CV_RGB2GRAY);
		return bgr_2[1];
		break;
	case CHANNEL_GB:
		cvtColor(bgr_2[2], bgr_2[2], CV_RGB2GRAY);
		return bgr_2[2];
		break;
	default:
		return bgr[0];
		break;
	}
}

void GLCM::update(GrayLevel level_, Direction direct_)
{
	switch (level_)
	{
	case GRAY_4:
		G_level = 4;
		break;
	case GRAY_8:
		G_level = 8;
		break;
	case GRAY_16:
		G_level = 16;
		break;
	default:
		G_level = 8;
		break;
	}
	direct = direct_;
}

double ** GLCM::getMatrix()
{
	return matrix;
}

void GLCM::printMatrix()
{
	std::cout << "  -GLCM matrix: \n";
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			std::cout << "     " << matrix[i][j] << "     ";
		}
		std::cout << std::endl;
	}
}

double * GLCM::getFeatures_Haralick()
{
	return features;
}

void GLCM::printFeatures_Haralick(std::ofstream &outFile)
{
	std::cout << "  -Haralick's features: \n";
	for (int i = 0; i < 6; i++) {
		std::cout << NameFeatures[i] << features[i] << std::endl;
		outFile << features[i] << ",";
	}
}

double GLCM::countValue(Mat & img, int id_u, int id_v)
{
	int coefficient = 32;
	switch (G_level)
	{
	case 4:
		coefficient = 64;
		break;
	case 8:
		coefficient = 32;
		break;
	case 16:
		coefficient = 16;
		break;
	default:
		break;
	}
	double res = 0;
	for (iu = 0; iu < img.rows; iu++) {
		for (jv = 0; jv < img.cols; jv++) {
			int pixel = (int)img.at<uchar>(iu, jv);
			pixel /= coefficient;
			switch (direct)
			{
			case DIRECTION_0:
				if (pixel == id_u && (jv + 1) < img.cols) {
					if (id_v == ((int)img.at<uchar>(iu, jv + 1) / coefficient)) {
						res += 1;
					}
				}
				break;
			case DIRECTION_45:
				if (pixel == id_u && (jv + 1) < img.cols && (iu - 1) >= 0) {
					if (id_v == ((int)img.at<uchar>(iu - 1, jv + 1) / coefficient)) {
						res += 1;
					}
				}
				break;
			case DIRECTION_90:
				if (pixel == id_u && (iu - 1) >= 0) {
					if (id_v == ((int)img.at<uchar>(iu - 1, jv) / coefficient)) {
						res += 1;
					}
				}
				break;
			case DIRECTION_135:
				if (pixel == id_u && (jv - 1) >= 0 && (iu - 1) >= 0) {
					if (id_v == ((int)img.at<uchar>(iu - 1, jv - 1) / coefficient)) {
						res += 1;
					}
				}
				break;
			default:
				break;
			}
		}
	}
	return res;
}
