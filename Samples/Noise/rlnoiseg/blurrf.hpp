/*
 * Blurring Functions ver 1.0
 *
 * This library is to be used
 * for creating PSF to blurr and
 * to deblurr images.
 *
 * Implementation is only for float numbers,
 * so the images must be rescaled.
 *
 * Also it work only with one image plane,
 * so color images must be treated plane by plane
 *
 * Implemented blurring
 * - Moffat
 * - Gaussian
 * - Defocus  (rough and antialiased)
 * - Motion   (rough and antialiased)
 *
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
 * PSF function - Moffat
 * center is assumed to be (0,0)
 * s1 - FWHM @ x
 * s2 - FWHM @ y
 * r - orientation
 * bet - (atmosferic) scattering coefficient
 */

float psfMoffat(int x, int y, float r, float s1, float s2, float bet);

/*
 * PSF function - Gauss
 * center is assumed to be (0,0)
 * s1 - FWHM @ x
 * s2 - FWHM @ y
 * r - orientation
 */

float psfGauss(int x, int y, float r, float s1, float s2);

/*
 * PSF function - Defocus
 * center is assumed to be (0,0)
 * r - defocusing radius
 */

float psfDefoc(int x, int y, float r);

/*
 * Procedure to calculate the maximal
 * span of the kernel, used for
 * the initialization of the kernel
 * for one variable
 */

int kspan(float (*ptLV1f)(int, int, float), float a1);

/*
 * Procedure to calculate the maximal
 * span of the kernel, used for
 * the initialization of the kernel
 * for three variables
 */

int kspan(float (*ptLV1f)(int, int, float, float, float), float a1, float a2,
          float a3);

/*
 * Procedure to calculate the maximal
 * span of the kernel, used for
 * the initialization of the kernel
 * for four variables
 */

int kspan(float (*ptLV1f)(int, int, float, float, float, float), float a1,
          float a2, float a3, float a4);

/*
 * Sets kernel for a function with one variable only
 */

void kset(cv::Mat& kernel, float (*ptLV1f)(int, int, float), float a1);

/*
 * Sets kernel for a function with three variables
 */

void kset(cv::Mat& kernel, float (*ptLV1f)(int, int, float, float, float),
          float a1, float a2, float a3);

/*
 * Sets kernel for a function with four variables
 */

void kset(cv::Mat& kernel,
          float (*ptLV1f)(int, int, float, float, float, float), float a1,
          float a2, float a3, float a4);

/*
 * This is procedure for setting of the defocus kernel
 * which seems to work better than
 * the previous one
 */

void ksetDefocus(cv::Mat& kernel, int r);

/*
 * Reminder!
 * To draw a complicated curve do this point by point
 * which correspond to moves for a fixed time steps.
 * Then take the average of all the kernels
 */

/*
 * This is procedure for setting of the move kernel
 * (x-x0,y-y0) defines the vector of movement
 * move was assumed with constant speed.
 */

void ksetMoveXY(cv::Mat& kernel, int x, int y, int x0 = 0, int y0 = 0);

/*
 * This is procedure for setting of the move kernel
 * (x-x0,y-y0) defines the vector of movement
 * move was assumed with constant speed.
 * Here the algorithm of Xiaolin Wu
 * is applied to give antialiased line
 * Better accuracy is expected.
 */

void ksetMoveXYxw(cv::Mat& kernel, float x, float y, float x0 = 0.0,
                  float y0 = 0.0);

/*
 * This is the best function
 * for creating the motion blurr
 * kernel
 */

void ksetMoveXYb(cv::Mat& kernel, float x, float y, float x0 = 0.0,
                 float y0 = 0.0);

/*
 * This is the continuation function
 * for creating the motion blurr
 * kernel; first point is not drawn,
 * but it is taken from the previous kernel;
 * the kernels are summed, then normalized
 */

void ksetMoveXYbcont(cv::Mat& kernel, float x, float y, float x0 = 0.0,
                     float y0 = 0.0);
