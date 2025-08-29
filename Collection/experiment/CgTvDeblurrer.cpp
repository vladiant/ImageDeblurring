/*
 * CgTvDeblurrer.cpp
 *
 *  Created on: Jan 26, 2015
 *      Author: vantonov
 */

#include "CgTvDeblurrer.h"

#include <string.h>

// TODO: For debug and tuning only
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define USE_GRADIENTS

namespace Test {
namespace Deblurring {

CgTvDeblurrer::CgTvDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mIrlsWeights(NULL),
      mIrlsWeightsX(NULL),
      mIrlsWeightsY(NULL),
      mOldDeblurredImage(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

CgTvDeblurrer::CgTvDeblurrer(int imageWidth, int imageHeight,
                             void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL),
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

CgTvDeblurrer::~CgTvDeblurrer() { deinit(); }

void CgTvDeblurrer::operator()(const uint8_t* inputImageData,
                               int inputImagePpln,
                               const SparseBlurKernel& currentBlurKernel,
                               uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

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
  } while (normIrls > 5e-6);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void CgTvDeblurrer::operator()(const uint8_t* inputImageData,
                               int inputImagePpln,
                               const point_value_t* kernelData, int kernelWidth,
                               int kernelHeight, int kernelPpln,
                               uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

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
  //	calculateIrlsWeightsX(mCurrentDeblurredImage, mIrlsWeightsX);
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
  } while (normIrls > 5e-6);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

size_t CgTvDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize =
      11 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void CgTvDeblurrer::init(int imageWidth, int imageHeight,
                         void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mCurrentDeblurredImage = &pDataBuffer[0];
    mResidualImage = &pDataBuffer[imageBlockSize];
    mPreconditionedImage = &pDataBuffer[2 * imageBlockSize];
    mBlurredPreconditionedImage = &pDataBuffer[3 * imageBlockSize];
    mDifferenceResidualImage = &pDataBuffer[4 * imageBlockSize];

    mRegularizationImage = &pDataBuffer[5 * imageBlockSize];
    mRegularizationImageTransposed = &pDataBuffer[6 * imageBlockSize];

    mIrlsWeights = &pDataBuffer[7 * imageBlockSize];
    mIrlsWeightsX = &pDataBuffer[8 * imageBlockSize];
    mIrlsWeightsY = &pDataBuffer[9 * imageBlockSize];
    mOldDeblurredImage = &pDataBuffer[10 * imageBlockSize];
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mPreconditionedImage = new point_value_t[imageBlockSize];
    mBlurredPreconditionedImage = new point_value_t[imageBlockSize];
    mDifferenceResidualImage = new point_value_t[imageBlockSize];

    mRegularizationImage = new point_value_t[imageBlockSize];
    mRegularizationImageTransposed = new point_value_t[imageBlockSize];

    mIrlsWeights = new point_value_t[imageBlockSize];
    mIrlsWeightsX = new point_value_t[imageBlockSize];
    mIrlsWeightsY = new point_value_t[imageBlockSize];
    mOldDeblurredImage = new point_value_t[imageBlockSize];
  }
}

void CgTvDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mPreconditionedImage;
    delete[] mBlurredPreconditionedImage;
    delete[] mDifferenceResidualImage;

    delete[] mRegularizationImage;
    delete[] mRegularizationImageTransposed;

    delete[] mIrlsWeights;
    delete[] mIrlsWeightsX;
    delete[] mIrlsWeightsY;
    delete[] mOldDeblurredImage;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mPreconditionedImage = NULL;
  mBlurredPreconditionedImage = NULL;
  mDifferenceResidualImage = NULL;

  mRegularizationImage = NULL;
  mRegularizationImageTransposed = NULL;

  mIrlsWeights = NULL;
  mIrlsWeightsX = NULL;
  mIrlsWeightsY = NULL;
  mOldDeblurredImage = NULL;

  mRegularizationWeight = 0;
}

void CgTvDeblurrer::doIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mDifferenceResidualImage, false);
    blur(mDifferenceResidualImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mBlurredPreconditionedImage, true);

    // TODO: regularization
#ifndef USE_GRADIENTS
    calculateRegularization(mPreconditionedImage,
                            mRegularizationImageTransposed, false);
    calculateRegularization(mRegularizationImageTransposed,
                            mRegularizationImage, true);
    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);
#else
    calculateRegularizationX(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationX(mRegularizationImageTransposed,
                             mRegularizationImage, true);

    multiplyImages(mRegularizationImage, mIrlsWeights, mRegularizationImage);
    multiplyImages(mRegularizationImage, mIrlsWeightsX, mRegularizationImage);

    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);

    calculateRegularizationY(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationY(mRegularizationImageTransposed,
                             mRegularizationImage, true);

    multiplyImages(mRegularizationImage, mIrlsWeights, mRegularizationImage);
    multiplyImages(mRegularizationImage, mIrlsWeightsY, mRegularizationImage);

    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);

#endif

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    // TODO: Add as debug print!
    cv::imshow(
        "Current Iteration mCurrentDeblurredImage",
        cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mCurrentDeblurredImage));
    cv::waitKey(10);
    std::cout << " Iteration: " << mCurrentIteration
              << " Norm: " << mCurrentNorm
              << " preconditionWeight: " << preconditionWeight
              << " updateWeight: " << updateWeight << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void CgTvDeblurrer::doIterations(const SparseBlurKernel& currentBlurKernel) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, currentBlurKernel, mDifferenceResidualImage,
         false);
    blur(mDifferenceResidualImage, currentBlurKernel,
         mBlurredPreconditionedImage, true);

    // TODO: regularization
#ifndef USE_GRADIENTS
    calculateRegularization(mPreconditionedImage,
                            mRegularizationImageTransposed, false);
    calculateRegularization(mRegularizationImageTransposed,
                            mRegularizationImage, true);
    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);
#else
    calculateRegularizationX(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationX(mRegularizationImageTransposed,
                             mRegularizationImage, true);

    multiplyImages(mRegularizationImage, mIrlsWeights,
                   mRegularizationImageTransposed);
    multiplyImages(mRegularizationImage, mIrlsWeightsX,
                   mRegularizationImageTransposed);

    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);

    calculateRegularizationY(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationY(mRegularizationImageTransposed,
                             mRegularizationImage, true);

    multiplyImages(mRegularizationImage, mIrlsWeights, mRegularizationImage);
    multiplyImages(mRegularizationImage, mIrlsWeightsY,
                   mRegularizationImageTransposed);

    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);
#endif

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    // TODO: Add as debug print!
    cv::imshow(
        "Current Iteration mCurrentDeblurredImage",
        cv::Mat(mImageHeight, mImageWidth, CV_32FC1, mCurrentDeblurredImage));
    cv::waitKey(10);
    std::cout << " Iteration: " << mCurrentIteration
              << " Norm: " << mCurrentNorm
              << " preconditionWeight: " << preconditionWeight
              << " updateWeight: " << updateWeight << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void CgTvDeblurrer::prepareIterations(const point_value_t* kernelData,
                                      int kernelWidth, int kernelHeight,
                                      int kernelPpln) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mBlurredPreconditionedImage, false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mResidualImage, true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  blur(mPreconditionedImage, kernelData, kernelWidth, kernelHeight, kernelPpln,
       mDifferenceResidualImage, false);
  blur(mDifferenceResidualImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mBlurredPreconditionedImage, true);

  // TODO: regularization
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgTvDeblurrer::prepareIterations(
    const SparseBlurKernel& currentBlurKernel) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, currentBlurKernel, mBlurredPreconditionedImage,
       false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, currentBlurKernel, mResidualImage, true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgTvDeblurrer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  inputImage.copyTo(outputImage);  // TM

  //	cv::Laplacian(inputImage, outputImage, CV_32FC1);
  //	cv::Laplacian(inputImage, outputImage, CV_32FC1); // Laplace
}

void CgTvDeblurrer::calculateRegularizationX(
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

void CgTvDeblurrer::calculateRegularizationY(
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

void CgTvDeblurrer::calculateIrlsWeights(const point_value_t* inputImageData,
                                         point_value_t* outputImageData) {
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
  cv::pow(outputImage, 0.5, outputImage);

  // Hyper Laplacian
  //	cv::pow(outputImage, 0.8, outputImage);

  // Ordinary TV
  outputImage = cv::Scalar(1.0);

  //	cv::threshold(outputImage, outputImage, 1.0, 1.0, cv::THRESH_TRUNC);
}

void CgTvDeblurrer::calculateIrlsWeightsX(const point_value_t* inputImageData,
                                          point_value_t* outputImageData) {
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
  //	cv::sqrt(outputImage, outputImage);

  //	derivativeX = cv::abs(derivativeX);
  //	cv::multiply(outputImage, derivativeX, outputImage);
}

void CgTvDeblurrer::calculateIrlsWeightsY(const point_value_t* inputImageData,
                                          point_value_t* outputImageData) {
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
  //	cv::sqrt(outputImage, outputImage);

  //	derivativeY = cv::abs(derivativeY);
  //	cv::multiply(outputImage, derivativeY, outputImage);
}

void CgTvDeblurrer::doRegularization() {}

}  // namespace Deblurring
}  // namespace Test
