/*
 * vanCittertTvDeblurrer.cpp
 *
 *  Created on: Jan 24, 2015
 *      Author: vantonov
 */

#include "vanCittertTvDeblurrer.h"

#include <string.h>

// TODO: For debug and tuning only
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define USE_GRADIENTS

namespace Test {
namespace Deblurring {

vanCittertTvDeblurrer::vanCittertTvDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mIrlsWeights(NULL),
      mIrlsWeightsX(NULL),
      mIrlsWeightsY(NULL),
      mOldDeblurredImage(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

vanCittertTvDeblurrer::vanCittertTvDeblurrer(int imageWidth, int imageHeight,
                                             void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mIrlsWeights(NULL),
      mIrlsWeightsX(NULL),
      mIrlsWeightsY(NULL),
      mOldDeblurredImage(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

vanCittertTvDeblurrer::~vanCittertTvDeblurrer() { deinit(); }

void vanCittertTvDeblurrer::operator()(
    const uint8_t* inputImageData, int inputImagePpln,
    const SparseBlurKernel& currentBlurKernel, uint8_t* outputImageData,
    int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(currentBlurKernel);

  // TODO: Prepare IRLS
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);
  memcpy(mOldDeblurredImage, mCurrentDeblurredImage, imageBlockSize);
  for (int i = 0; i < mImageWidth * mImageHeight; i++) {
    mIrlsWeights[i] = 1;
    mIrlsWeightsX[i] = 1;
    mIrlsWeightsY[i] = 1;
  }
  //	calculateIrlsWeights(mCurrentDeblurredImage, mIrlsWeights);
  //	calculateIrlsWeightsX(mCurrentDeblurredImage, mIrlsWeightsX);
  //	calculateIrlsWeightsY(mCurrentDeblurredImage, mIrlsWeightsY);

  double normIrls = 1;
  do {
    doIterations(currentBlurKernel);

    calculateIrlsWeights(mCurrentDeblurredImage, mIrlsWeights);
    calculateIrlsWeightsX(mCurrentDeblurredImage, mIrlsWeightsX);
    calculateIrlsWeightsY(mCurrentDeblurredImage, mIrlsWeightsY);

    normIrls = calculateL2NormOfDifference(mCurrentDeblurredImage,
                                           mOldDeblurredImage) /
               (mImageWidth * mImageHeight);

    memcpy(mOldDeblurredImage, mCurrentDeblurredImage, imageBlockSize);

    // TODO: Debug print
    cv::imshow("Current Iteration Weights",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeights));
    cv::imshow("Current Iteration Weights X",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeightsX));
    cv::imshow("Current Iteration Weights Y",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeightsY));
    cv::waitKey(10);
    std::cout << "norm IRLS: " << normIrls << std::endl;

    // TODO: Set as constant!
  } while (normIrls > 1e-5);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void vanCittertTvDeblurrer::operator()(const uint8_t* inputImageData,
                                       int inputImagePpln,
                                       const point_value_t* kernelData,
                                       int kernelWidth, int kernelHeight,
                                       int kernelPpln, uint8_t* outputImageData,
                                       int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  // TODO: Prepare IRLS
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);
  memcpy(mOldDeblurredImage, mCurrentDeblurredImage, imageBlockSize);
  for (int i = 0; i < mImageWidth * mImageHeight; i++) {
    mIrlsWeights[i] = 1;
    mIrlsWeightsX[i] = 1;
    mIrlsWeightsY[i] = 1;
  }
  //	calculateIrlsWeights(mCurrentDeblurredImage, mIrlsWeights);
  //	calculateIrlsWeightsY(mCurrentDeblurredImage, mIrlsWeightsY);

  double normIrls = 1;
  do {
    doIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

    calculateIrlsWeights(mCurrentDeblurredImage, mIrlsWeights);
    calculateIrlsWeightsX(mCurrentDeblurredImage, mIrlsWeightsX);
    calculateIrlsWeightsY(mCurrentDeblurredImage, mIrlsWeightsY);

    normIrls = calculateL2NormOfDifference(mCurrentDeblurredImage,
                                           mOldDeblurredImage) /
               (mImageWidth * mImageHeight);

    memcpy(mOldDeblurredImage, mCurrentDeblurredImage, imageBlockSize);

    // TODO: Debug print
    cv::imshow("Current Iteration Weights",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeights));
    cv::imshow("Current Iteration Weights X",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeightsX));
    cv::imshow("Current Iteration Weights Y",
               cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mIrlsWeightsY));
    cv::waitKey(10);
    std::cout << "norm IRLS: " << normIrls << std::endl;

    // TODO: Set as constant!
  } while (normIrls > 1e-5);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

size_t vanCittertTvDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize =
      10 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void vanCittertTvDeblurrer::init(int imageWidth, int imageHeight,
                                 void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mCurrentDeblurredImage = &pDataBuffer[0];
    mResidualImage = &pDataBuffer[imageBlockSize];
    mReblurredImage = &pDataBuffer[2 * imageBlockSize];
    mBlurredImage = &pDataBuffer[3 * imageBlockSize];

    mRegularizationImage = &pDataBuffer[4 * imageBlockSize];
    mRegularizationImageTransposed = &pDataBuffer[5 * imageBlockSize];

    mIrlsWeights = &pDataBuffer[6 * imageBlockSize];
    mIrlsWeightsX = &pDataBuffer[7 * imageBlockSize];
    mIrlsWeightsY = &pDataBuffer[8 * imageBlockSize];
    mOldDeblurredImage = &pDataBuffer[9 * imageBlockSize];
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mReblurredImage = new point_value_t[imageBlockSize];
    mBlurredImage = new point_value_t[imageBlockSize];

    mRegularizationImage = new point_value_t[imageBlockSize];
    mRegularizationImageTransposed = new point_value_t[imageBlockSize];

    mIrlsWeights = new point_value_t[imageBlockSize];
    mIrlsWeightsX = new point_value_t[imageBlockSize];
    mIrlsWeightsY = new point_value_t[imageBlockSize];
    mOldDeblurredImage = new point_value_t[imageBlockSize];
  }
}

void vanCittertTvDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mReblurredImage;
    delete[] mBlurredImage;

    delete[] mRegularizationImage;
    delete[] mRegularizationImageTransposed;

    delete[] mIrlsWeights;
    delete[] mIrlsWeightsX;
    delete[] mIrlsWeightsY;
    delete[] mOldDeblurredImage;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mReblurredImage = NULL;
  mBlurredImage = NULL;

  mRegularizationImage = NULL;
  mRegularizationImageTransposed = NULL;

  mIrlsWeights = NULL;
  mIrlsWeightsX = NULL;
  mIrlsWeightsY = NULL;
  mOldDeblurredImage = NULL;

  mRegularizationWeight = 0;
}

void vanCittertTvDeblurrer::doIterations(const point_value_t* kernelData,
                                         int kernelWidth, int kernelHeight,
                                         int kernelPpln) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mReblurredImage, false);

    subtractImages(mBlurredImage, mReblurredImage, mResidualImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    blur(mResidualImage, kernelData, kernelWidth, kernelHeight, kernelPpln,
         mReblurredImage, true);

    // TODO: regularization
#ifndef USE_GRADIENTS
    calculateRegularization(mCurrentDeblurredImage, mRegularizationImage,
                            false);
    calculateRegularization(mRegularizationImage,
                            mRegularizationImageTransposed, true);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);
#else
    calculateRegularizationX(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationX(mRegularizationImage,
                             mRegularizationImageTransposed, true);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeights,
                   mRegularizationImageTransposed);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeightsX,
                   mRegularizationImageTransposed);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);

    calculateRegularizationY(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationY(mRegularizationImage,
                             mRegularizationImageTransposed, true);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeights,
                   mRegularizationImageTransposed);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeightsY,
                   mRegularizationImageTransposed);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);
#endif

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    // TODO: Add as debug print!
    //		calculateIrlsWeights(mCurrentDeblurredImage, mIrlsWeights);
    //		cv::imshow("Current Iteration Weights",
    //				cv::Mat(mImageHeight, mImageWidth, CV_32FC1,
    // mIrlsWeights));
    cv::imshow(
        "Current Iteration mCurrentDeblurredImage",
        cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mCurrentDeblurredImage));
    cv::waitKey(10);
    std::cout << " Iteration: " << mCurrentIteration
              << " Norm: " << mCurrentNorm << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void vanCittertTvDeblurrer::doIterations(
    const SparseBlurKernel& currentBlurKernel) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, currentBlurKernel, mReblurredImage, false);

    subtractImages(mBlurredImage, mReblurredImage, mResidualImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    blur(mResidualImage, currentBlurKernel, mReblurredImage, true);

    // TODO: regularization
#ifndef USE_GRADIENTS
    calculateRegularization(mCurrentDeblurredImage, mRegularizationImage,
                            false);
    calculateRegularization(mRegularizationImage,
                            mRegularizationImageTransposed, true);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);
#else
    calculateRegularizationX(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationX(mRegularizationImage,
                             mRegularizationImageTransposed, true);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeights,
                   mRegularizationImageTransposed);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeightsX,
                   mRegularizationImageTransposed);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);

    calculateRegularizationY(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationY(mRegularizationImage,
                             mRegularizationImageTransposed, true);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeights,
                   mRegularizationImageTransposed);
    multiplyImages(mRegularizationImageTransposed, mIrlsWeightsY,
                   mRegularizationImageTransposed);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);
#endif

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    // TODO: Add as debug print!
    cv::imshow(
        "Current Iteration mCurrentDeblurredImage",
        cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mCurrentDeblurredImage));
    cv::waitKey(10);
    std::cout << " Iteration: " << mCurrentIteration
              << " Norm: " << mCurrentNorm << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void vanCittertTvDeblurrer::prepareIterations(
    [[maybe_unused]] const point_value_t* kernelData,
    [[maybe_unused]] int kernelWidth, [[maybe_unused]] int kernelHeight,
    [[maybe_unused]] int kernelPpln) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  memcpy(mCurrentDeblurredImage, mBlurredImage, imageBlockSize);
  memset(mResidualImage, 0, imageBlockSize);
  memset(mReblurredImage, 0, imageBlockSize);
}

void vanCittertTvDeblurrer::prepareIterations(
    [[maybe_unused]] const SparseBlurKernel& currentBlurKernel) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  memcpy(mCurrentDeblurredImage, mBlurredImage, imageBlockSize);
  memset(mResidualImage, 0, imageBlockSize);
  memset(mReblurredImage, 0, imageBlockSize);
}

void vanCittertTvDeblurrer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  //	inputImage.copyTo(outputImage); // TM

  cv::Laplacian(inputImage, outputImage, CV_32FC1);  // Laplace
}

void vanCittertTvDeblurrer::calculateRegularizationX(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel = false) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  float kernel[] = {-3.0, 0, 3.0, -10.0, 0, 10.0, -3.0, 0, 3.0};

  cv::Mat blurKernel(3, 3, CV_32FC1, kernel);

  if (transposeKernel) {
    cv::Mat flippedBlurKernel;
    cv::flip(blurKernel, flippedBlurKernel, -1);
    cv::filter2D(inputImage, outputImage, CV_32FC1, flippedBlurKernel);
  } else {
    cv::filter2D(inputImage, outputImage, CV_32FC1, blurKernel);
  }

  return;
}

void vanCittertTvDeblurrer::calculateRegularizationY(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel = false) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  float kernel[] = {-3.0, -10.0, -3.0, 0, 0, 0, 3.0, 10.0, 3.0};

  cv::Mat blurKernel(3, 3, CV_32FC1, kernel);

  if (transposeKernel) {
    cv::Mat flippedBlurKernel;
    cv::flip(blurKernel, flippedBlurKernel, -1);
    cv::filter2D(inputImage, outputImage, CV_32FC1, flippedBlurKernel);
  } else {
    cv::filter2D(inputImage, outputImage, CV_32FC1, blurKernel);
  }

  return;
}

void vanCittertTvDeblurrer::calculateIrlsWeights(
    const point_value_t* inputImageData, point_value_t* outputImageData) {
  // Adapted TV
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  cv::Mat derivativeX(mImageHeight, mImageWidth, CV_32FC1);
  cv::Mat derivativeY(mImageHeight, mImageWidth, CV_32FC1);

  float kernelX[] = {-3.0, 0, 3.0, -10.0, 0, 10.0, -3.0, 0, 3.0};
  float kernelY[] = {-3.0, -10.0, -3.0, 0, 0, 0, 3.0, 10.0, 3.0};
  cv::Mat blurKernelX(3, 3, CV_32FC1, kernelX);
  cv::Mat blurKernelY(3, 3, CV_32FC1, kernelY);

  cv::filter2D(inputImage, derivativeX, CV_32FC1, blurKernelX);
  cv::filter2D(inputImage, derivativeY, CV_32FC1, blurKernelY);

  cv::multiply(derivativeX, derivativeX, derivativeX);
  cv::multiply(derivativeY, derivativeY, derivativeY);
  cv::add(derivativeX, derivativeY, outputImage);

  // Adapted TV
  //	cv::pow(outputImage, 0.5, outputImage);

  // Hyper Laplacian
  //	cv::pow(outputImage, 0.8, outputImage);

  // Ordinary TV
  outputImage = cv::Scalar(1.0);
}

void vanCittertTvDeblurrer::calculateIrlsWeightsX(
    const point_value_t* inputImageData, point_value_t* outputImageData) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  cv::Mat derivativeX(mImageHeight, mImageWidth, CV_32FC1);
  float kernelX[] = {-3.0, 0, 3.0, -10.0, 0, 10.0, -3.0, 0, 3.0};
  cv::Mat blurKernelX(3, 3, CV_32FC1, kernelX);

  cv::filter2D(inputImage, derivativeX, CV_32FC1, blurKernelX);

  // adapted TV, Hyper Laplacian
  cv::multiply(derivativeX, derivativeX, outputImage);
  cv::sqrt(outputImage, outputImage);
  cv::divide(1.0, outputImage, outputImage);

  // Ordinary TV
  cv::sqrt(outputImage, outputImage);

  //	derivativeX = cv::abs(derivativeX);
  //	cv::multiply(outputImage, derivativeX, outputImage);

  // TODO: Define truncation value
  cv::threshold(outputImage, outputImage, 1.0, 1.0, cv::THRESH_TRUNC);
}

void vanCittertTvDeblurrer::calculateIrlsWeightsY(
    const point_value_t* inputImageData, point_value_t* outputImageData) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  cv::Mat derivativeY(mImageHeight, mImageWidth, CV_32FC1);
  float kernelY[] = {-3.0, -10.0, -3.0, 0, 0, 0, 3.0, 10.0, 3.0};
  cv::Mat blurKernelY(3, 3, CV_32FC1, kernelY);

  // adapted TV, Hyper Laplacian
  cv::filter2D(inputImage, derivativeY, CV_32FC1, blurKernelY);
  cv::multiply(derivativeY, derivativeY, outputImage);
  cv::divide(1.0, outputImage, outputImage);

  // Ordinary TV
  cv::sqrt(outputImage, outputImage);

  //	derivativeY = cv::abs(derivativeY);
  //	cv::multiply(outputImage, derivativeY, outputImage);

  // TODO: Define truncation value
  cv::threshold(outputImage, outputImage, 1.0, 1.0, cv::THRESH_TRUNC);
}

void vanCittertTvDeblurrer::doRegularization() {}

}  // namespace Deblurring
}  // namespace Test
