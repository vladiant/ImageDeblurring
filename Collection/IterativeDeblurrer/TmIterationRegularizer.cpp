/*
 * TmIterationRegularizer.cpp
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#include "TmIterationRegularizer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

TmIterationRegularizer::TmIterationRegularizer(int imageWidth, int imageHeight)
    : IterationRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

TmIterationRegularizer::TmIterationRegularizer(int imageWidth, int imageHeight,
                                               void* pExternalMemory)
    : IterationRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterationRegularizer::getMemorySize(imageWidth, imageHeight)));
}

TmIterationRegularizer::~TmIterationRegularizer() { deinit(); }

size_t TmIterationRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 0;

  requiredMemorySize +=
      IterationRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void TmIterationRegularizer::init([[maybe_unused]] int imageWidth,
                                  [[maybe_unused]] int imageHeight,
                                  [[maybe_unused]] void* pExternalMemory) {}

void TmIterationRegularizer::deinit() {}

void TmIterationRegularizer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    [[maybe_unused]] bool transposeKernel) {
  memcpy(outputImageData, inputImageData, mImageBlockSize);
}

void TmIterationRegularizer::calculateRegularizationX(
    [[maybe_unused]] const point_value_t* inputImageData,
    point_value_t* outputImageData, [[maybe_unused]] bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void TmIterationRegularizer::calculateRegularizationY(
    [[maybe_unused]] const point_value_t* inputImageData,
    point_value_t* outputImageData, [[maybe_unused]] bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void TmIterationRegularizer::calculateIrlsWeights(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void TmIterationRegularizer::calculateIrlsWeightsX(
    [[maybe_unused]] const point_value_t* inputImageData) {}

void TmIterationRegularizer::calculateIrlsWeightsY(
    [[maybe_unused]] const point_value_t* inputImageData) {}

}  // namespace Deblurring
}  // namespace Test
