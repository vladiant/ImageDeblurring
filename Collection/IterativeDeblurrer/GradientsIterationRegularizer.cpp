/*
 * GradientsIterationRegularizer.cpp
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#include "GradientsIterationRegularizer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

GradientsIterationRegularizer::GradientsIterationRegularizer(int imageWidth,
                                                             int imageHeight)
    : IterationRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

GradientsIterationRegularizer::GradientsIterationRegularizer(
    int imageWidth, int imageHeight, void* pExternalMemory)
    : IterationRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterationRegularizer::getMemorySize(imageWidth, imageHeight)));
}

GradientsIterationRegularizer::~GradientsIterationRegularizer() { deinit(); }

size_t GradientsIterationRegularizer::getMemorySize(int imageWidth,
                                                    int imageHeight) {
  int requiredMemorySize = 0;

  requiredMemorySize +=
      IterationRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void GradientsIterationRegularizer::init(
    [[maybe_unused]] int imageWidth, [[maybe_unused]] int imageHeight,
    [[maybe_unused]] void* pExternalMemory) {}

void GradientsIterationRegularizer::deinit() {}

void GradientsIterationRegularizer::calculateRegularization(
    [[maybe_unused]] const point_value_t* inputImageData,
    point_value_t* outputImageData, [[maybe_unused]] bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void GradientsIterationRegularizer::calculateRegularizationX(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  float cUpLeft, cLeft, cDownLeft, cUpRight, cRight, cDownRight;
  if (transposeKernel) {
    cUpLeft = -3.0;
    cLeft = -10.0;
    cDownLeft = -3.0;
    cUpRight = 3.0;
    cRight = 10.0;
    cDownRight = 3.0;
  } else {
    cUpLeft = 3.0;
    cLeft = 10.0;
    cDownLeft = 3.0;
    cUpRight = -3.0;
    cRight = -10.0;
    cDownRight = -3.0;
  }

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      outputImageData[col + row * mImageWidth] =
          inputImageData[left + up * mImageWidth] * cUpLeft +
          inputImageData[left + row * mImageWidth] * cLeft +
          inputImageData[left + down * mImageWidth] * cDownLeft +
          inputImageData[right + up * mImageWidth] * cUpRight +
          inputImageData[right + row * mImageWidth] * cRight +
          inputImageData[right + down * mImageWidth] * cDownRight;
    }
  }

  return;
}

void GradientsIterationRegularizer::calculateRegularizationY(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  float cUpLeft, cUp, cDownLeft, cUpRight, cDown, cDownRight;
  if (transposeKernel) {
    cUpLeft = -3.0;
    cUp = -10.0;
    cUpRight = -3.0;
    cDownLeft = 3.0;
    cDown = 10.0;
    cDownRight = 3.0;
  } else {
    cUpLeft = 3.0;
    cUp = 10.0;
    cUpRight = 3.0;
    cDownLeft = -3.0;
    cDown = -10.0;
    cDownRight = -3.0;
  }

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      outputImageData[col + row * mImageWidth] =
          inputImageData[left + up * mImageWidth] * cUpLeft +
          inputImageData[col + up * mImageWidth] * cUp +
          inputImageData[left + down * mImageWidth] * cDownLeft +
          inputImageData[right + up * mImageWidth] * cUpRight +
          inputImageData[col + down * mImageWidth] * cDown +
          inputImageData[right + down * mImageWidth] * cDownRight;
    }
  }

  return;
}

void GradientsIterationRegularizer::calculateIrlsWeights(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void GradientsIterationRegularizer::calculateIrlsWeightsX(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void GradientsIterationRegularizer::calculateIrlsWeightsY(
    [[maybe_unused]] const point_value_t* inputImageData) {}

}  // namespace Deblurring
}  // namespace Test
