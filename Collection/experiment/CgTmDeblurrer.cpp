/*
 * CgTmDeblurrer.cpp
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#include "CgTmDeblurrer.h"

#include <string.h>

// TODO: For debug and tuning only
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #define USE_GRADIENTS

namespace Test {
namespace Deblurring {

CgTmDeblurrer::CgTmDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

CgTmDeblurrer::CgTmDeblurrer(int imageWidth, int imageHeight,
                             void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

CgTmDeblurrer::~CgTmDeblurrer() { deinit(); }

void CgTmDeblurrer::operator()(const uint8_t* inputImageData,
                               int inputImagePpln,
                               const SparseBlurKernel& currentBlurKernel,
                               uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

  prepareIterations(currentBlurKernel);

  doIterations(currentBlurKernel);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void CgTmDeblurrer::operator()(const uint8_t* inputImageData,
                               int inputImagePpln,
                               const point_value_t* kernelData, int kernelWidth,
                               int kernelHeight, int kernelPpln,
                               uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

  prepareIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  doIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

size_t CgTmDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 7 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void CgTmDeblurrer::init(int imageWidth, int imageHeight,
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
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mPreconditionedImage = new point_value_t[imageBlockSize];
    mBlurredPreconditionedImage = new point_value_t[imageBlockSize];
    mDifferenceResidualImage = new point_value_t[imageBlockSize];

    mRegularizationImage = new point_value_t[imageBlockSize];
    mRegularizationImageTransposed = new point_value_t[imageBlockSize];
  }
}

void CgTmDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mPreconditionedImage;
    delete[] mBlurredPreconditionedImage;
    delete[] mDifferenceResidualImage;

    delete[] mRegularizationImage;
    delete[] mRegularizationImageTransposed;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mPreconditionedImage = NULL;
  mBlurredPreconditionedImage = NULL;
  mDifferenceResidualImage = NULL;

  mRegularizationImage = NULL;
  mRegularizationImageTransposed = NULL;

  mRegularizationWeight = 0;
}

void CgTmDeblurrer::doIterations(const point_value_t* kernelData,
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
    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);

    calculateRegularizationY(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationY(mRegularizationImageTransposed,
                             mRegularizationImage, true);
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

void CgTmDeblurrer::doIterations(const SparseBlurKernel& currentBlurKernel) {
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
    addWeightedImages(mBlurredPreconditionedImage, 1.0, mRegularizationImage,
                      mRegularizationWeight, 0.0, mBlurredPreconditionedImage);

    calculateRegularizationY(mPreconditionedImage,
                             mRegularizationImageTransposed, false);
    calculateRegularizationY(mRegularizationImageTransposed,
                             mRegularizationImage, true);
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

void CgTmDeblurrer::prepareIterations(const point_value_t* kernelData,
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
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgTmDeblurrer::prepareIterations(
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

void CgTmDeblurrer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1,
                     const_cast<point_value_t*>(inputImageData));
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, outputImageData);

  inputImage.copyTo(outputImage);  // TM

  //	cv::Laplacian(inputImage, outputImage, CV_32FC1);
  //	cv::Laplacian(inputImage, outputImage, CV_32FC1); // Laplace
}

void CgTmDeblurrer::calculateRegularizationX(
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

void CgTmDeblurrer::calculateRegularizationY(
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

void CgTmDeblurrer::doRegularization() {}

}  // namespace Deblurring
}  // namespace Test
