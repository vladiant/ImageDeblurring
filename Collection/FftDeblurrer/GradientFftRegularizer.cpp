/*
 * GradientFftRegularizer.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#include "GradientFftRegularizer.h"

#include <stdlib.h>

namespace Test {
namespace Deblurring {

GradientFftRegularizer::GradientFftRegularizer(int imageWidth, int imageHeight)
    : FftRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

GradientFftRegularizer::GradientFftRegularizer(int imageWidth, int imageHeight,
                                               void* pExternalMemory)
    : FftRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftRegularizer::getMemorySize(imageWidth, imageHeight)));
}

GradientFftRegularizer::~GradientFftRegularizer() { deinit(); }

size_t GradientFftRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void GradientFftRegularizer::init([[maybe_unused]] int imageWidth,
                                  [[maybe_unused]] int imageHeight,
                                  void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mRegularizationFftImageX = &pDataBuffer[0 * bufferSize];
    mRegularizationFftImageY = &pDataBuffer[1 * bufferSize];
  } else {
    mRegularizationFftImageX = new point_value_t[bufferSize];
    mRegularizationFftImageY = new point_value_t[bufferSize];
  }

  setRegularizationX();
  setRegularizationY();
}

void GradientFftRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationFftImageX;
    delete[] mRegularizationFftImageY;
  }

  mRegularizationFftImage = NULL;
}

void GradientFftRegularizer::setRegularization() {}

void GradientFftRegularizer::setRegularizationX() {
  calculateGradientXFft2D(mRegularizationFftImageX);
}

void GradientFftRegularizer::setRegularizationY() {
  calculateGradientYFft2D(mRegularizationFftImageY);
}

void GradientFftRegularizer::calculateHqpmWeights(
    [[maybe_unused]] const point_value_t* inputFftImageData,
    [[maybe_unused]] float beta) {}

void GradientFftRegularizer::calculateHqpmWeightsX(
    [[maybe_unused]] const point_value_t* inputFftImageData,
    [[maybe_unused]] float beta) {}

void GradientFftRegularizer::calculateHqpmWeightsY(
    [[maybe_unused]] const point_value_t* inputFftImageData,
    [[maybe_unused]] float beta) {}

void GradientFftRegularizer::calculateHqpmWeight(
    [[maybe_unused]] const point_value_t* inputImageData,
    [[maybe_unused]] float beta,
    [[maybe_unused]] point_value_t* outputImageData) {}

}  // namespace Deblurring
}  // namespace Test
