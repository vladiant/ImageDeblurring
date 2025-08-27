/*
 * LaplaceIterationRegularizer.cpp
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#include "LaplaceIterationRegularizer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

LaplaceIterationRegularizer::LaplaceIterationRegularizer(int imageWidth,
                                                         int imageHeight)
    : IterationRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

LaplaceIterationRegularizer::LaplaceIterationRegularizer(int imageWidth,
                                                         int imageHeight,
                                                         void* pExternalMemory)
    : IterationRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterationRegularizer::getMemorySize(imageWidth, imageHeight)));
}

LaplaceIterationRegularizer::~LaplaceIterationRegularizer() { deinit(); }

size_t LaplaceIterationRegularizer::getMemorySize(int imageWidth,
                                                  int imageHeight) {
  int requiredMemorySize = 0;

  requiredMemorySize +=
      IterationRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void LaplaceIterationRegularizer::init([[maybe_unused]] int imageWidth,
                                       [[maybe_unused]] int imageHeight,
                                       [[maybe_unused]] void* pExternalMemory) {
}

void LaplaceIterationRegularizer::deinit() {}

void LaplaceIterationRegularizer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  const float cUp = -1.0;
  const float cRight = -1.0;
  const float cDown = -1.0;
  const float cLeft = -1.0;
  const float cCenter = 4.0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      outputImageData[col + row * mImageWidth] =
          +inputImageData[col + up * mImageWidth] * cUp +
          inputImageData[col + row * mImageWidth] * cCenter +
          inputImageData[right + row * mImageWidth] * cRight +
          inputImageData[col + down * mImageWidth] * cDown +
          inputImageData[left + row * mImageWidth] * cLeft;
    }
  }

  return;
}

void LaplaceIterationRegularizer::calculateRegularizationX(
    [[maybe_unused]] const point_value_t* inputImageData,
    point_value_t* outputImageData, [[maybe_unused]] bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void LaplaceIterationRegularizer::calculateRegularizationY(
    [[maybe_unused]] const point_value_t* inputImageData,
    point_value_t* outputImageData, [[maybe_unused]] bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void LaplaceIterationRegularizer::calculateIrlsWeights(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void LaplaceIterationRegularizer::calculateIrlsWeightsX(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void LaplaceIterationRegularizer::calculateIrlsWeightsY(
    [[maybe_unused]] const point_value_t* inputImageData) {}

}  // namespace Deblurring
}  // namespace Test
