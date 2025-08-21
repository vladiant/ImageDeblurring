/*
 * LaplaceFftRegularizer.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#include "LaplaceFftRegularizer.h"

#include <stdlib.h>

namespace Test {
namespace Deblurring {

LaplaceFftRegularizer::LaplaceFftRegularizer(int imageWidth, int imageHeight)
    : FftRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

LaplaceFftRegularizer::LaplaceFftRegularizer(int imageWidth, int imageHeight,
                                             void* pExternalMemory)
    : FftRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftRegularizer::getMemorySize(imageWidth, imageHeight)));
}

LaplaceFftRegularizer::~LaplaceFftRegularizer() { deinit(); }

size_t LaplaceFftRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 2 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void LaplaceFftRegularizer::init(int imageWidth, int imageHeight,
                                 void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mRegularizationFftImage = &pDataBuffer[0];
  } else {
    mRegularizationFftImage = new point_value_t[bufferSize];
  }

  setRegularization();
}

void LaplaceFftRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationFftImage;
  }

  mRegularizationFftImage = NULL;
}

void LaplaceFftRegularizer::setRegularization() {
  calculateLaplaceFft2D(mRegularizationFftImage);
}

void LaplaceFftRegularizer::setRegularizationX() {}

void LaplaceFftRegularizer::setRegularizationY() {}

void LaplaceFftRegularizer::calculateHqpmWeights(
    const point_value_t* inputFftImageData, float beta) {}

void LaplaceFftRegularizer::calculateHqpmWeightsX(
    const point_value_t* inputFftImageData, float beta) {}

void LaplaceFftRegularizer::calculateHqpmWeightsY(
    const point_value_t* inputFftImageData, float beta) {}

void LaplaceFftRegularizer::calculateHqpmWeight(
    const point_value_t* inputImageData, float beta,
    point_value_t* outputImageData) {}

}  // namespace Deblurring
}  // namespace Test
