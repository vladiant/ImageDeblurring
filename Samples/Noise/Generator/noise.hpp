/*
 * Image Noise ver 1.0
 *
 * This library is to be used to add specific noise to
 * images in the OpenCv framework.
 *
 * Implementation is only for float numbers,
 * so the images must be rescaled
 *
 * Also it work only with one image plane,
 * so color images must be treated plane by plane
 *
 * Noise generated:
 * - Salt and Pepper
 * - Gaussian
 * - Poisson
 * - Speckle
 *
 * Implementation of the noise functions is the same
 * as in the Matlab function imnoise with minor changes.
 * Read the description of the functions
 * for more information.
 *
 * Created by Vladislav Antonov
 * October 2011
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
 * Gaussian distribution, taken from GSL
 * sigma i the standard deviation
 * mean is zero - add value if required
 */
double gaussian(const double sigma);

/*
 * This function generates
 * Possion distribution,
 * via the Knuth's algorithm
 */
double poisson(const double lambda);

/*
 * Generates salt and pepper noise
 * pr - part of the image to be affected
 */

void SPnoise(cv::Mat &imgc, float pr = 0.05);

/*
 * Adds Gaussian noise with:
 * mean - mean value
 * stdev - standard deviation
 */
void Gnoise(cv::Mat &imgc, float mean = 0, float stdev = 0.01);

/*
 * Adds zero mean Gaussian noise with
 * variance, defined for each pixel by:
 * imgm
 * and mean=0
 */
void LVnoise(cv::Mat &imgc, cv::Mat &imgm);

/*
 * Adds zero mean Gaussian noise with
 * zero mean and variance, defined
 * for each pixel by a function of its intensity.
 *
 * This function should be defined
 * in the code of the program.
 */
void LV1noise(cv::Mat &imgc, float (*ptLV1f)(float));

/*
 * Adds Poisson noise with
 * scaling of pixel values:
 * scl - should be defined
 * to suit user needs
 */
void Pnoise(cv::Mat &imgc, float scl = 255);

/*
 * Generates multiplicative uniform noise
 * with mean=0 and variance:
 * stdev
 * which is applied to image as follows:
 * I=I+I*noise
 */
void SPEnoise(cv::Mat &imgc, float stdev = 0.04);

/*
 * Adds uniform noise, with:
 * v - intensity
 * pr - part of the image to be affected
 */

void Unoise(cv::Mat &imgc, float pr, float v = 1.0);
