/*
 * Fourier DeBlurring ver 1.0
 *
 * This library is to be used with the DFT approach
 * to deblurr and denoise images.
 *
 * Implementation is only for float numbers,
 * so the images must be rescaled
 *
 * Also it work only with one image plane,
 * so color images must be treated plane by plane
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
 * This procedure does convolution
 * of two images via the DFT
 * Note: Periodic Border Conditions!
 */

void FCFilter(cv::Mat& imga1, cv::Mat& imgb1, cv::Mat& imgc1);

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix.
 * To solve the division by zero problem
 * and to apply Wiener filter
 * a variable of gamma is introduced:
 * it can be zero for pure inversion;
 * it can be equal to NSR for filtering;
 */
void FMatInv(cv::Mat& imgw, float gamma = 1e-7);

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * Note: Periodic Border Conditions!
 */

void FDFilter(cv::Mat& imga1, cv::Mat& imgb1, cv::Mat& imgc,
              float gamma = 1e-7);

/*
 * This procedure sets the blurr image
 * which is DFT transformed for blurring
 * using given kernel
 * Note that the blurr image should be the same size
 * as the iage to be blurred by DFT
 */

void Blurrset(cv::Mat& imga, cv::Mat& krnl);

/*
 * This procedure applies mirror border around the image
 * with sizes:
 * px - horizontal
 * py - vertical
 */
void MirrorBorder(cv::Mat& src, cv::Mat& ibmat, int px, int py);

/*
 * This function gives axes to the elements
 * of the packed Fourier transformed matrix.
 * It was found in internet address
 * http://mildew.ee.engr.uky.edu/~weisu/OpenCV/OpenCV_archives_16852-19727.htm
 * and proposed by Vadim Pisarevsky
 */

void FGet2D(const cv::Mat& Y, int k, int i, float* re, float* im);

/*
 * Modification of the previous function
 * to set the values of Fourier matrix
 */

void FSet2D(cv::Mat& Y, int k, int i, float* re, float* im);

/*
 * This function gradually
 * decreases intensity at
 * the image borders to zero
 * It is used to prevent "ringing"
 */

void EdgeTaper(cv::Mat& imga, int px, int py);
