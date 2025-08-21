/*
 * CutOffFftDeblurrer.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#include "CutOffFftDeblurrer.h"

namespace Test {
namespace Deblurring {

CutOffFftDeblurrer::CutOffFftDeblurrer(int imageWidth, int imageHeight)
    : FftDeblurrer(imageWidth, imageHeight),
      mWorkImage(NULL),
      mKernelImage(NULL),
      mCutOffFrequencyX(mFftImageWidth / 2),
      mCutOffFrequencyY(mFftImageHeight / 2) {
  init(imageWidth, imageHeight, NULL);
}

CutOffFftDeblurrer::CutOffFftDeblurrer(int imageWidth, int imageHeight,
                                       void* pExternalMemory)
    : FftDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCutOffFrequencyX(mFftImageWidth / 2),
      mCutOffFrequencyY(mFftImageHeight / 2) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

CutOffFftDeblurrer::~CutOffFftDeblurrer() { deinit(); }

size_t CutOffFftDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageWidth);
  imageHeight =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void CutOffFftDeblurrer::init(int imageWidth, int imageHeight,
                              void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mWorkImage = &pDataBuffer[0 * bufferSize];
    mKernelImage = &pDataBuffer[1 * bufferSize];
  } else {
    mWorkImage = new point_value_t[bufferSize];
    mKernelImage = new point_value_t[bufferSize];
  }
}

void CutOffFftDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mWorkImage;
    delete[] mKernelImage;
  }

  mWorkImage = NULL;
  mKernelImage = NULL;

  mCutOffFrequencyX = 0;
  mCutOffFrequencyY = 0;
}

void CutOffFftDeblurrer::operator()(const uint8_t* inputImageData,
                                    int inputImagePpln,
                                    const SparseBlurKernel& currentBlurKernel,
                                    uint8_t* outputImageData,
                                    int outputImagePpln) {
  convertFromInputToFft(inputImageData, mImageWidth, mImageHeight,
                        inputImagePpln, mWorkImage, mFftImageWidth * 2,
                        mFftImageHeight);

  prepareBordersOfFftImage(mWorkImage, mImageWidth, mImageHeight,
                           mFftImageWidth * 2);

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  prepareKernelFftImage(currentBlurKernel, mKernelImage, 2 * mFftImageWidth,
                        mFftImageHeight);

  calculateFft2D(mKernelImage, mFftImageWidth, mFftImageHeight, FFT_FORWARD);

  invertFftMatrix(mKernelImage, mFftImageWidth, mFftImageHeight, mKernelImage);

  multiplyFftImages(mWorkImage, mKernelImage, mFftImageWidth, mFftImageHeight,
                    mWorkImage);

  applyCutOff(mWorkImage, mFftImageWidth, mFftImageHeight, mCutOffFrequencyX,
              mCutOffFrequencyY);

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mWorkImage, mImageWidth, mImageHeight, mFftImageWidth * 2,
                     outputImageData, outputImagePpln);
}

void CutOffFftDeblurrer::operator()(const uint8_t* inputImageData,
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

  invertFftMatrix(mKernelImage, mFftImageWidth, mFftImageHeight, mKernelImage);

  multiplyFftImages(mWorkImage, mKernelImage, mFftImageWidth, mFftImageHeight,
                    mWorkImage);

  applyCutOff(mWorkImage, mFftImageWidth, mFftImageHeight, mCutOffFrequencyX,
              mCutOffFrequencyY);

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mWorkImage, mImageWidth, mImageHeight, mFftImageWidth * 2,
                     outputImageData, outputImagePpln);
}

bool CutOffFftDeblurrer::setCutOffFrequencyX(int cutOffFrequencyX) {
  if (cutOffFrequencyX > 0 && cutOffFrequencyX < mImageWidth / 2) {
    mCutOffFrequencyX = 1.0 * cutOffFrequencyX * mFftImageWidth / mImageWidth;
    return true;
  } else {
    return false;
  }
}

bool CutOffFftDeblurrer::setCutOffFrequencyY(int cutOffFrequencyY) {
  if (cutOffFrequencyY > 0 && cutOffFrequencyY < mImageHeight / 2) {
    mCutOffFrequencyY = 1.0 * cutOffFrequencyY * mFftImageHeight / mImageHeight;
    return true;
  } else {
    return false;
  }
}

}  // namespace Deblurring
}  // namespace Test
