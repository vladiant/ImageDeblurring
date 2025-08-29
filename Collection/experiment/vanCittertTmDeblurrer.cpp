/*
 * vanCittertTmDeblurrer.cpp
 *
 *  Created on: Jan 24, 2015
 *      Author: vantonov
 */

#include "vanCittertTmDeblurrer.h"

#include <string.h>

// TODO: For debug and tuning only
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define USE_GRADIENTS

namespace Test {
namespace Deblurring {

vanCittertTmDeblurrer::vanCittertTmDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

vanCittertTmDeblurrer::vanCittertTmDeblurrer(int imageWidth, int imageHeight,
                                             void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

vanCittertTmDeblurrer::~vanCittertTmDeblurrer() { deinit(); }

void vanCittertTmDeblurrer::operator()(
    const uint8_t* inputImageData, int inputImagePpln,
    const SparseBlurKernel& currentBlurKernel, uint8_t* outputImageData,
    int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(currentBlurKernel);

  doIterations(currentBlurKernel);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void vanCittertTmDeblurrer::operator()(const uint8_t* inputImageData,
                                       int inputImagePpln,
                                       const point_value_t* kernelData,
                                       int kernelWidth, int kernelHeight,
                                       int kernelPpln, uint8_t* outputImageData,
                                       int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  doIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

size_t vanCittertTmDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 6 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void vanCittertTmDeblurrer::init(int imageWidth, int imageHeight,
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
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mReblurredImage = new point_value_t[imageBlockSize];
    mBlurredImage = new point_value_t[imageBlockSize];

    mRegularizationImage = new point_value_t[imageBlockSize];
    mRegularizationImageTransposed = new point_value_t[imageBlockSize];
  }
}

void vanCittertTmDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mReblurredImage;
    delete[] mBlurredImage;

    delete[] mRegularizationImage;
    delete[] mRegularizationImageTransposed;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mReblurredImage = NULL;
  mBlurredImage = NULL;

  mRegularizationImage = NULL;
  mRegularizationImageTransposed = NULL;

  mRegularizationWeight = 0;
}

void vanCittertTmDeblurrer::doIterations(const point_value_t* kernelData,
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
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);

    calculateRegularizationY(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationY(mRegularizationImage,
                             mRegularizationImageTransposed, true);
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

void vanCittertTmDeblurrer::doIterations(
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
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);

    calculateRegularizationY(mCurrentDeblurredImage, mRegularizationImage,
                             false);
    calculateRegularizationY(mRegularizationImage,
                             mRegularizationImageTransposed, true);
    addWeightedImages(mReblurredImage, 1.0, mRegularizationImageTransposed,
                      -mRegularizationWeight, 0.0, mReblurredImage);
#endif

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

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

void vanCittertTmDeblurrer::prepareIterations(
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

void vanCittertTmDeblurrer::prepareIterations(
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

void vanCittertTmDeblurrer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  //	inputImage.copyTo(outputImage); // TM

  cv::Laplacian(inputImage, outputImage, CV_32FC1);  // Laplace
}

void vanCittertTmDeblurrer::calculateRegularizationX(
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

void vanCittertTmDeblurrer::calculateRegularizationY(
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

void vanCittertTmDeblurrer::doRegularization() {}

}  // namespace Deblurring
}  // namespace Test
