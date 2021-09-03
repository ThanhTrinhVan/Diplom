#include "GLCM.h"

GLCM::GLCM()
{
	gLevel = 8;
	matrix = new double *[gLevel];
	for (int i = 0; i < gLevel; i++) {
		matrix[i] = new double[gLevel];
	}
	features = new double[6];
}

GLCM::GLCM(GrayLevel gl)
{
	switch (gl)
	{
	case GRAY_4:
		gLevel = 4;
		break;
	case GRAY_8:
		gLevel = 8;
		break;
	case GRAY_16:
		gLevel = 16;
		break;
	default:
		gLevel = 8;
		break;
	}
	matrix = new double *[gLevel];
	for (int i = 0; i < gLevel; i++) {
		matrix[i] = new double[gLevel];
	}
	features = new double[6];
}

GLCM::~GLCM()
{
	if (matrix) {
		for (int i = 0; i < gLevel; i++)
			delete matrix[i];
		delete matrix;
	}
	if (features)
		delete features;
}

Mat GLCM::getImgByChannel(Mat & img, ChannelRGB cn)
{
	Mat bgr[3];
	split(img, bgr);

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
		return (bgr[2] - bgr[1]);
		break;
	case CHANNEL_RB:
		return (bgr[2] - bgr[0]);
		break;
	case CHANNEL_GB:
		return (bgr[1] - bgr[0]);
		break;
	default:
		std::cout << "Incorrect type of channel! \n";
		exit(1);
	}
}

void GLCM::calculateMatrix(Mat & img, int delta_row, int delta_col)
{
	sumValue = 0;
	int coefficient = 32;
	switch (gLevel)
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

	int imatrix, jmatrix;
	int newi, newj;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			newi = i + delta_row;
			newj = j + delta_col;
			if (newi < img.rows && newi >= 0 && newj < img.cols && newj >= 0) {
				imatrix = (int)img.at<uchar>(i, j) / coefficient;
				jmatrix = (int)img.at<uchar>(newi, newj) / coefficient;
				matrix[imatrix][jmatrix] += 1.0;
				sumValue += 1.0;
			}
		}
	}
}

void GLCM::normalizeMatrix()
{
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			matrix[i][j] /= sumValue;
		}
	}
}

void GLCM::calculateFeatures()
{
	// contrast
	features[0] = 0;
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			features[0] += pow((i - j), 2)*matrix[i][j];
		}
	}
	//features[0] /= pow(gLevel - 1, 2);
	// correlation
	features[1] = 0;
	double* mu_u = new double[gLevel + 1];
	double* si_u = new double[gLevel + 1];
	double* mu_v = new double[gLevel + 1];
	double* si_v = new double[gLevel + 1];

	mu_u[gLevel] = 0;
	for (int i = 0; i < gLevel; i++) {
		mu_u[i] = 0.0;
		for (int j = 0; j < gLevel; j++) {
			mu_u[i] += i * matrix[i][j];
		}
		mu_u[gLevel] += mu_u[i];
	}

	mu_v[gLevel] = 0;
	for (int j = 0; j < gLevel; j++) {
		mu_v[j] = 0;
		for (int i = 0; i < gLevel; i++) {
			mu_v[j] += j * matrix[i][j];
		}
		mu_v[gLevel] += mu_v[j];
	}

	si_u[gLevel] = 0;
	for (int i = 0; i < gLevel; i++) {
		si_u[i] = 0;
		for (int j = 0; j < gLevel; j++) {
			si_u[i] += matrix[i][j] * pow((i - mu_u[gLevel]), 2);
		}
		//si_u[i] = sqrt(si_u[i]);
		si_u[gLevel] += si_u[i];
	}

	si_v[gLevel] = 0;
	for (int j = 0; j < gLevel; j++) {
		si_v[j] = 0;
		for (int i = 0; i < gLevel; i++) {
			si_v[j] += matrix[i][j] * pow((j - mu_v[gLevel]), 2);
		}
		//si_v[j] = sqrt(si_v[j]);
		si_v[gLevel] += si_v[j];
	}

	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			if (matrix[i][j] != 0)
				features[1] += matrix[i][j] * (i - mu_u[gLevel])*(j - mu_v[gLevel]) / sqrt(si_u[gLevel] * si_v[gLevel]);
		}
	}
	//features[1] = (features[1] + 1) / 2;
	// energy
	features[2] = 0;
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			features[2] += pow(matrix[i][j], 2);
		}
	}
	// homogeneity
	features[3] = 0;
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			features[3] += matrix[i][j] / (1 + abs(i - j));
		}
	}
	// entropy
	features[4] = 0;
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			if (matrix[i][j] != 0)
				features[4] += matrix[i][j] * log10(matrix[i][j]);
		}
	}
	features[4] = -features[4];
	// Inverse Difference Moment
	features[5] = 0;
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			features[5] += matrix[i][j] / (1 + pow(i - j, 2));
		}
	}

	delete mu_u, mu_v, si_u, si_v;
}

double ** GLCM::getMatrix()
{
	return matrix;
}

void GLCM::printMatrix()
{
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			std::cout << matrix[i][j] << "     ";
		}
		std::cout << std::endl;
	}
	std::cout << sumValue << std::endl;
}

double * GLCM::getFeatures_Haralick()
{
	return features;
}

void GLCM::printFeatures_Haralick(std::ofstream &outFile)
{
	std::cout << "  -Haralick's features: \n";
	for (int i = 0; i < 6; i++) {
		std::cout << nameFeatures[i] << features[i] << std::endl;
		outFile << features[i] << ",";
	}
}

void GLCM::reset()
{
	for (int i = 0; i < gLevel; i++) {
		for (int j = 0; j < gLevel; j++) {
			matrix[i][j] = 0;
		}
	}
	for (int i = 0; i < 6; i++) {
		features[i] = 0;
	}
	sumValue = 0;
}