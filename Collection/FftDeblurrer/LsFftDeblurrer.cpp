/*
 * LsFftDeblurrer.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#include "LsFftDeblurrer.h"

namespace Test {
namespace Deblurring {

LsFftDeblurrer::LsFftDeblurrer(int imageWidth, int imageHeight)
    : FftDeblurrer(imageWidth, imageHeight),
      mRegularizer(NULL),
      mWorkImage(NULL),
      mKernelImage(NULL),
      mInvertedKernelImage(NULL),
      mDeblurredImage(NULL) {
  init(imageWidth, imageHeight, NULL);
}

LsFftDeblurrer::LsFftDeblurrer(int imageWidth, int imageHeight,
                               void* pExternalMemory)
    : FftDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mRegularizer(NULL),
      mWorkImage(NULL),
      mKernelImage(NULL),
      mInvertedKernelImage(NULL),
      mDeblurredImage(NULL) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

LsFftDeblurrer::~LsFftDeblurrer() { deinit(); }

size_t LsFftDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageWidth);
  imageHeight =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 8 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void LsFftDeblurrer::init([[maybe_unused]] int imageWidth,
                          [[maybe_unused]] int imageHeight,
                          void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mWorkImage = &pDataBuffer[0 * bufferSize];
    mKernelImage = &pDataBuffer[1 * bufferSize];
    mInvertedKernelImage = &pDataBuffer[2 * bufferSize];
    mDeblurredImage = &pDataBuffer[3 * bufferSize];
  } else {
    mWorkImage = new point_value_t[bufferSize];
    mKernelImage = new point_value_t[bufferSize];
    mInvertedKernelImage = new point_value_t[bufferSize];
    mDeblurredImage = new point_value_t[bufferSize];
  }
}

void LsFftDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mWorkImage;
    delete[] mKernelImage;
    delete[] mInvertedKernelImage;
    delete[] mDeblurredImage;
  }

  mRegularizer = NULL;

  mWorkImage = NULL;
  mKernelImage = NULL;
  mInvertedKernelImage = NULL;
  mDeblurredImage = NULL;
}

void LsFftDeblurrer::operator()(const uint8_t* inputImageData,
                                int inputImagePpln,
                                const SparseBlurKernel& currentBlurKernel,
                                uint8_t* outputImageData, int outputImagePpln) {
  convertFromInputToFft(inputImageData, mImageWidth, mImageHeight,
                        inputImagePpln, mWorkImage, mFftImageWidth * 2,
                        mFftImageHeight);

  prepareBordersOfFftImage(mWorkImage, mImageWidth, mImageHeight,
                           mFftImageWidth * 2);

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  prepareKernelFftImage(currentBlurKernel, mKernelImage, 2 * mFftImageWidth,
                        mFftImageHeight);

  calculateFft2D(mKernelImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  float beta = 1.0;
  float maxBeta = 1.0;
  float betaMultStep = 1.0;

  if (mRegularizer != NULL) {
    beta = mRegularizer->getMinimalHqpmWeight();
    betaMultStep = mRegularizer->getMultiplyStepHqpmWeight();
    maxBeta = mRegularizer->getMaximalHqpmWeight();
  }

  updateHqpmWeights(beta);

  do {
    calculateDeblurKernel(beta);

    multiplyFftImages(mWorkImage, mInvertedKernelImage, mFftImageWidth,
                      mFftImageHeight, mDeblurredImage);

    if (mRegularizer != NULL && !mRegularizer->isHqpmUsed()) {
      break;
    }

    applyHqpmWeights(beta);

    updateHqpmWeights(beta);

    beta *= betaMultStep;

  } while (beta < maxBeta);

  calculateFft2D(mDeblurredImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mDeblurredImage, mImageWidth, mImageHeight,
                     mFftImageWidth * 2, outputImageData, outputImagePpln);

  return;
}  // void LsFftDeblurrer::operator()( ...

void LsFftDeblurrer::operator()(const uint8_t* inputImageData,
                                int inputImagePpln,
                                const point_value_t* kernelData,
                                int kernelWidth, int kernelHeight,
                                int kernelPpln, uint8_t* outputImageData,
                                int outputImagePpln) {
  convertFromInputToFft(inputImageData, mImageWidth, mImageHeight,
                        inputImagePpln, mWorkImage, mFftImageWidth * 2,
                        mFftImageHeight);

  prepareBordersOfFftImage(mWorkImage, mImageWidth, mImageHeight,
                           mFftImageWidth * 2);

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  prepareKernelFftImage(kernelData, kernelWidth, kernelHeight, kernelPpln,
                        mKernelImage, 2 * mFftImageWidth, mFftImageHeight);

  calculateFft2D(mKernelImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  float beta = 1.0;
  float maxBeta = 1.0;
  float betaMultStep = 1.0;

  if (mRegularizer != NULL) {
    beta = mRegularizer->getMinimalHqpmWeight();
    betaMultStep = mRegularizer->getMultiplyStepHqpmWeight();
    maxBeta = mRegularizer->getMaximalHqpmWeight();
  }

  updateHqpmWeights(beta);

  do {
    calculateDeblurKernel(beta);

    multiplyFftImages(mWorkImage, mInvertedKernelImage, mFftImageWidth,
                      mFftImageHeight, mDeblurredImage);

    if (mRegularizer != NULL && !mRegularizer->isHqpmUsed()) {
      break;
    }

    applyHqpmWeights(beta);

    updateHqpmWeights(beta);

    beta *= betaMultStep;

  } while (beta < maxBeta);

  calculateFft2D(mDeblurredImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mDeblurredImage, mImageWidth, mImageHeight,
                     mFftImageWidth * 2, outputImageData, outputImagePpln);

  return;
}  // void LsFftDeblurrer::operator()( ...

void LsFftDeblurrer::updateHqpmWeights(float beta) {
  if (mRegularizer != NULL && mRegularizer->isHqpmUsed()) {
    point_value_t* hpmWeightsFftImage = mRegularizer->getHqpmWeightsFftImage();
    point_value_t* hpmWeightsFftImageX =
        mRegularizer->getHqpmWeightsFftImageX();
    point_value_t* hpmWeightsFftImageY =
        mRegularizer->getHqpmWeightsFftImageY();

    if (hpmWeightsFftImage != NULL) {
      mRegularizer->calculateHqpmWeights(mWorkImage, beta);
    }

    if (hpmWeightsFftImageX != NULL && hpmWeightsFftImageY != NULL) {
      mRegularizer->calculateHqpmWeightsX(mWorkImage, beta);
      mRegularizer->calculateHqpmWeightsY(mWorkImage, beta);
    }
  }
}

void LsFftDeblurrer::calculateDeblurKernel(float beta) {
  if (mRegularizer != NULL) {
    point_value_t* regularizationImage =
        mRegularizer->getRegularizationFftImage();
    point_value_t* regularizationImageX =
        mRegularizer->getRegularizationFftImageX();
    point_value_t* regularizationImageY =
        mRegularizer->getRegularizationFftImageY();

    float regularizationWeight = mRegularizer->getRegularizationWeight();

    if (mRegularizer != NULL && mRegularizer->isHqpmUsed()) {
      regularizationWeight *= beta;
    }

    if (regularizationImage != NULL) {
      invertFftMatrixRegularized(mKernelImage, regularizationImage,
                                 mFftImageWidth, mFftImageHeight,
                                 regularizationWeight, mInvertedKernelImage);

    } else if (regularizationImageX != NULL && regularizationImageY != NULL) {
      invertFftMatrixRegularized(mKernelImage, regularizationImageX,
                                 regularizationImageY, mFftImageWidth,
                                 mFftImageHeight, regularizationWeight,
                                 mInvertedKernelImage);

    } else {
      invertFftMatrix(mKernelImage, mFftImageWidth, mFftImageHeight,
                      regularizationWeight, mInvertedKernelImage);
    }

  } else {
    invertFftMatrix(mKernelImage, mFftImageWidth, mFftImageHeight,
                    mInvertedKernelImage);
  }

  return;
}  // void LsFftDeblurrer::calculateDeblurKernel( ...

void LsFftDeblurrer::applyHqpmWeights(float beta) {
  if (mRegularizer != NULL && mRegularizer->isHqpmUsed()) {
    point_value_t* hpmWeightsFftImage = mRegularizer->getHqpmWeightsFftImage();
    point_value_t* hpmWeightsFftImageX =
        mRegularizer->getHqpmWeightsFftImageX();
    point_value_t* hpmWeightsFftImageY =
        mRegularizer->getHqpmWeightsFftImageY();

    point_value_t* regularizationImage =
        mRegularizer->getRegularizationFftImage();
    point_value_t* regularizationImageX =
        mRegularizer->getRegularizationFftImageX();
    point_value_t* regularizationImageY =
        mRegularizer->getRegularizationFftImageY();

    float regularizationWeight =
        (mRegularizer->getRegularizationWeight() * beta);

    if (hpmWeightsFftImage != NULL) {
      mRegularizer->addFftHqpmImage(mDeblurredImage, mKernelImage,
                                    regularizationImage, hpmWeightsFftImage,
                                    regularizationWeight, mDeblurredImage);
    }

    if (hpmWeightsFftImageX != NULL && hpmWeightsFftImageY != NULL) {
      mRegularizer->addFftHqpmImage(mDeblurredImage, mKernelImage,
                                    regularizationImageX, regularizationImageY,
                                    hpmWeightsFftImageX, hpmWeightsFftImageY,
                                    regularizationWeight, mDeblurredImage);
    }
  }
}

}  // namespace Deblurring
}  // namespace Test
