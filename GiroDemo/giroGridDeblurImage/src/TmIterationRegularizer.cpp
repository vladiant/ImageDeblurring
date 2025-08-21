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

void TmIterationRegularizer::init(int imageWidth, int imageHeight,
                                  void* pExternalMemory) {}

void TmIterationRegularizer::deinit() {}

void TmIterationRegularizer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memcpy(outputImageData, inputImageData, mImageBlockSize);
}

void TmIterationRegularizer::calculateRegularizationX(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void TmIterationRegularizer::calculateRegularizationY(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void TmIterationRegularizer::calculateIrlsWeights(
    const point_value_t* inputImageData) {}

void TmIterationRegularizer::calculateIrlsWeightsX(
    const point_value_t* inputImageData) {}

void TmIterationRegularizer::calculateIrlsWeightsY(
    const point_value_t* inputImageData) {}

}  // namespace Deblurring
}  // namespace Test
