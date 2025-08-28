/*
 * InverseFftDeblurrer.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#include "InverseFftDeblurrer.h"

namespace Test {
namespace Deblurring {

InverseFftDeblurrer::InverseFftDeblurrer(int imageWidth, int imageHeight)
    : FftDeblurrer(imageWidth, imageHeight),
      mWorkImage(NULL),
      mKernelImage(NULL) {
  init(imageWidth, imageHeight, NULL);
}

InverseFftDeblurrer::InverseFftDeblurrer(int imageWidth, int imageHeight,
                                         void* pExternalMemory)
    : FftDeblurrer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

InverseFftDeblurrer::~InverseFftDeblurrer() { deinit(); }

size_t InverseFftDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageWidth);
  imageHeight =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void InverseFftDeblurrer::init([[maybe_unused]] int imageWidth,
                               [[maybe_unused]] int imageHeight,
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

void InverseFftDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mWorkImage;
    delete[] mKernelImage;
  }

  mWorkImage = NULL;
  mKernelImage = NULL;
}

void InverseFftDeblurrer::operator()(const uint8_t* inputImageData,
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

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mWorkImage, mImageWidth, mImageHeight, mFftImageWidth * 2,
                     outputImageData, outputImagePpln);
}

void InverseFftDeblurrer::operator()(const uint8_t* inputImageData,
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

  calculateFft2D(mWorkImage, mFftImageWidth, mFftImageHeight, FFT_INVERSE);

  convertFftToOutput(mWorkImage, mImageWidth, mImageHeight, mFftImageWidth * 2,
                     outputImageData, outputImagePpln);
}

}  // namespace Deblurring
}  // namespace Test
