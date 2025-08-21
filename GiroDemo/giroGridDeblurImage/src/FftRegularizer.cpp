/*
 * FftRegularizer.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#include "FftRegularizer.h"

#include <stdlib.h>
#include <string.h>

#include "FftDeblurrer.h"

namespace Test {
namespace Deblurring {

FftRegularizer::FftRegularizer(int imageWidth, int imageHeight)
    : isExternalMemoryUsed(false),
      mFftImageWidth(0),
      mFftImageHeight(0),
      mRegularizationWeight(0),
      mRegularizationFftImage(NULL),
      mRegularizationFftImageX(NULL),
      mRegularizationFftImageY(NULL),
      mMinimalHqpmWeight(HQPM_WEIGHT_MINIMAL),
      mMaximalHqpmWeight(HQPM_WEIGHT_MAXIMAL),
      mMultiplyStepHqpmWeight(HQPM_WEIGHT_MULTIPLY_STEP),
      mHqpmWeightsFftImage(NULL),
      mHqpmWeightsFftImageX(NULL),
      mHqpmWeightsFftImageY(NULL),
      mBufferReal(NULL),
      mBufferImaginary(NULL) {
  initalize(imageWidth, imageHeight, NULL);
}

FftRegularizer::FftRegularizer(int imageWidth, int imageHeight,
                               void* pExternalMemory)
    : isExternalMemoryUsed(true),
      mFftImageWidth(0),
      mFftImageHeight(0),
      mRegularizationWeight(0),
      mRegularizationFftImage(NULL),
      mRegularizationFftImageX(NULL),
      mRegularizationFftImageY(NULL),
      mMinimalHqpmWeight(HQPM_WEIGHT_MINIMAL),
      mMaximalHqpmWeight(HQPM_WEIGHT_MAXIMAL),
      mMultiplyStepHqpmWeight(HQPM_WEIGHT_MULTIPLY_STEP),
      mHqpmWeightsFftImage(NULL),
      mHqpmWeightsFftImageX(NULL),
      mHqpmWeightsFftImageY(NULL),
      mBufferReal(NULL),
      mBufferImaginary(NULL) {
  initalize(imageWidth, imageHeight, pExternalMemory);
}

FftRegularizer::~FftRegularizer() { deinitalize(); }

size_t FftRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int bufferSize = imageWidth > imageHeight ? imageWidth : imageHeight;

  int requiredMemorySize = 2 * bufferSize * sizeof(point_value_t);

  requiredMemorySize += ImageDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void FftRegularizer::initalize(int imageWidth, int imageHeight,
                               void* pExternalMemory) {
  mFftImageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  mFftImageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int bufferSize =
      mFftImageWidth > mFftImageHeight ? mFftImageWidth : mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mBufferReal = &pDataBuffer[0];
    mBufferImaginary = &pDataBuffer[bufferSize];
  } else {
    mBufferReal = new point_value_t[bufferSize];
    mBufferImaginary = new point_value_t[bufferSize];
  }
}

void FftRegularizer::deinitalize() {
  if (!isExternalMemoryUsed) {
    delete[] mBufferReal;
    delete[] mBufferImaginary;
  }

  mFftImageWidth = 0;
  mFftImageHeight = 0;

  mRegularizationWeight = 0;
  mRegularizationFftImage = NULL;
  mRegularizationFftImageX = NULL;
  mRegularizationFftImageY = NULL;

  mHqpmWeightsFftImage = NULL;
  mHqpmWeightsFftImageX = NULL;
  mHqpmWeightsFftImageY = NULL;

  mBufferReal = NULL;
  mBufferImaginary = NULL;
}

bool FftRegularizer::calculateFft2D(point_value_t* c, int nx, int ny, int dir) {
  return FftDeblurrer::calculateFft2D(c, nx, ny, dir, mBufferReal,
                                      mBufferImaginary);
}

void FftRegularizer::calculateIdentityFft2D(point_value_t* fftImage) {
  memset(fftImage, 0,
         2 * mFftImageWidth * mFftImageHeight * sizeof(point_value_t));

  fftImage[0] = 1.0;

  calculateFft2D(fftImage, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void FftRegularizer::calculateLaplaceFft2D(point_value_t* fftImage) {
  memset(fftImage, 0,
         2 * mFftImageWidth * mFftImageHeight * sizeof(point_value_t));

  fftImage[0] = 4.0;
  fftImage[1] = -1.0;
  fftImage[2 * (mFftImageWidth - 1)] = -1.0;
  fftImage[2 * mFftImageWidth] = 1.0;
  fftImage[2 * mFftImageWidth * (mFftImageHeight - 1)] = -1.0;

  calculateFft2D(fftImage, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void FftRegularizer::calculateGradientXFft2D(point_value_t* fftImage) {
  memset(fftImage, 0,
         2 * mFftImageWidth * mFftImageHeight * sizeof(point_value_t));

  fftImage[2] = 3.0;
  fftImage[2 + 2 * mFftImageWidth] = 10.0;
  fftImage[2 + 2 * mFftImageWidth * (mFftImageHeight - 1)] = 3.0;

  fftImage[2 * (mFftImageWidth - 1)] = -3.0;
  fftImage[2 * (mFftImageWidth - 1) + 2 * mFftImageWidth] = -10.0;
  fftImage[2 * (mFftImageWidth - 1) +
           2 * mFftImageWidth * (mFftImageHeight - 1)] = -3.0;

  calculateFft2D(fftImage, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void FftRegularizer::calculateGradientYFft2D(point_value_t* fftImage) {
  memset(fftImage, 0,
         2 * mFftImageWidth * mFftImageHeight * sizeof(point_value_t));

  fftImage[2 * (mFftImageWidth - 1) + 2 * mFftImageWidth] = 3.0;
  fftImage[0 + 2 * mFftImageWidth] = 10.0;
  fftImage[2 + 2 * mFftImageWidth] = 3.0;

  fftImage[2 * (mFftImageWidth - 1) +
           2 * mFftImageWidth * (mFftImageHeight - 1)] = -3.0;
  fftImage[0 + 2 * mFftImageWidth * (mFftImageHeight - 1)] = -10.0;
  fftImage[2 + 2 * mFftImageWidth * (mFftImageHeight - 1)] = -3.0;

  calculateFft2D(fftImage, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void FftRegularizer::addFftHqpmImage(
    const point_value_t* inputFftImageData,
    const point_value_t* kernelFftImageData,
    const point_value_t* regularizerFftImageData,
    const point_value_t* weightFftImageData, float regularizationWeight,
    point_value_t* outputFftImageData) {
  for (int row = 0; row < mFftImageHeight; row++) {
    for (int col = 0; col < mFftImageWidth; col++) {
      point_value_t real =
          kernelFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginary =
          kernelFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realReg =
          regularizerFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryReg =
          regularizerFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realW =
          weightFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryW =
          weightFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t sum = real * real + imaginary * imaginary +
                          regularizationWeight *
                              (realReg * realReg + imaginaryReg * imaginaryReg);

      outputFftImageData[2 * (col + row * mFftImageWidth) + 0] =
          inputFftImageData[2 * (col + row * mFftImageWidth) + 0] +
          regularizationWeight * (realW * realReg + imaginaryW * imaginaryReg) /
              sum;
      outputFftImageData[2 * (col + row * mFftImageWidth) + 1] =
          inputFftImageData[2 * (col + row * mFftImageWidth) + 1] +
          regularizationWeight * (imaginaryW * realReg - realW * imaginaryReg) /
              sum;
    }
  }

  return;
}

void FftRegularizer::addFftHqpmImage(
    const point_value_t* inputFftImageData,
    const point_value_t* kernelFftImageData,
    const point_value_t* firstRegularizerFftImageData,
    const point_value_t* secondRegularizerFftImageData,
    const point_value_t* firstWeightFftImageData,
    const point_value_t* secondWeightFftImageData, float regularizationWeight,
    point_value_t* outputFftImageData) {
  for (int row = 0; row < mFftImageHeight; row++) {
    for (int col = 0; col < mFftImageWidth; col++) {
      point_value_t real =
          kernelFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginary =
          kernelFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realReg1 =
          firstRegularizerFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryReg1 =
          firstRegularizerFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realReg2 =
          secondRegularizerFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryReg2 =
          secondRegularizerFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realW1 =
          firstWeightFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryW1 =
          firstWeightFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t realW2 =
          secondWeightFftImageData[2 * (col + row * mFftImageWidth) + 0];
      point_value_t imaginaryW2 =
          secondWeightFftImageData[2 * (col + row * mFftImageWidth) + 1];

      point_value_t sum =
          real * real + imaginary * imaginary +
          regularizationWeight *
              (realReg1 * realReg1 + imaginaryReg1 * imaginaryReg1 +
               realReg2 * realReg2 + imaginaryReg2 * imaginaryReg2);

      outputFftImageData[2 * (col + row * mFftImageWidth) + 0] =
          inputFftImageData[2 * (col + row * mFftImageWidth) + 0] +
          regularizationWeight *
              (realW1 * realReg1 + imaginaryW1 * imaginaryReg1 +
               realW1 * realReg1 + imaginaryW1 * imaginaryReg1) /
              sum;
      outputFftImageData[2 * (col + row * mFftImageWidth) + 1] =
          inputFftImageData[2 * (col + row * mFftImageWidth) + 1] +
          regularizationWeight *
              (imaginaryW1 * realReg1 - realW1 * imaginaryReg1 +
               imaginaryW2 * realReg2 - realW2 * imaginaryReg2) /
              sum;
    }
  }

  return;
}

}  // namespace Deblurring
}  // namespace Test
