/*
 * MaskMaskLaplaceIterationRegularizer.cpp
 *
 *  Created on: 07.02.2015
 *      Author: vladiant
 */

#include "MaskLaplaceIterationRegularizer.h"

#include <math.h>
#include <string.h>

namespace Test {
namespace Deblurring {

MaskLaplaceIterationRegularizer::MaskLaplaceIterationRegularizer(
    int imageWidth, int imageHeight)
    : IterationRegularizer(imageWidth, imageHeight),
      mStandardDeviationWeight(STANDARD_DEVIATION_WEIGHT) {
  init(imageWidth, imageHeight, NULL);
}

MaskLaplaceIterationRegularizer::MaskLaplaceIterationRegularizer(
    int imageWidth, int imageHeight, void* pExternalMemory)
    : IterationRegularizer(imageWidth, imageHeight, pExternalMemory),
      mStandardDeviationWeight(STANDARD_DEVIATION_WEIGHT) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterationRegularizer::getMemorySize(imageWidth, imageHeight)));
}

MaskLaplaceIterationRegularizer::~MaskLaplaceIterationRegularizer() {
  deinit();
}

size_t MaskLaplaceIterationRegularizer::getMemorySize(int imageWidth,
                                                      int imageHeight) {
  int requiredMemorySize = 2 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterationRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void MaskLaplaceIterationRegularizer::init(int imageWidth, int imageHeight,
                                           void* pExternalMemory) {
  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mIrlsWeights = &pDataBuffer[0 * mImageBlockSize / sizeof(point_value_t)];
    mOldDeblurredImage =
        &pDataBuffer[1 * mImageBlockSize / sizeof(point_value_t)];
  } else {
    mIrlsWeights = new point_value_t[mImageBlockSize / sizeof(point_value_t)];
    mOldDeblurredImage =
        new point_value_t[mImageBlockSize / sizeof(point_value_t)];
  }
}

void MaskLaplaceIterationRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mIrlsWeights;
    delete[] mOldDeblurredImage;
  }

  mIrlsWeights = NULL;
  mOldDeblurredImage = NULL;
}

void MaskLaplaceIterationRegularizer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
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

void MaskLaplaceIterationRegularizer::calculateRegularizationX(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void MaskLaplaceIterationRegularizer::calculateRegularizationY(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

;

void MaskLaplaceIterationRegularizer::prepareIrls(
    const point_value_t* inputImageData) {
  IterationRegularizer::prepareIrls(inputImageData);

  mMaxIrlsIterations = 1;

  const float mStandardDeviationWeight = 10.0;
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      const float c11 = inputImageData[right + up * mImageWidth];
      const float c12 = inputImageData[col + up * mImageWidth];
      const float c13 = inputImageData[left + up * mImageWidth];
      const float c21 = inputImageData[right + row * mImageWidth];
      const float c22 = inputImageData[col + row * mImageWidth];
      const float c23 = inputImageData[left + row * mImageWidth];
      const float c31 = inputImageData[right + down * mImageWidth];
      const float c32 = inputImageData[col + down * mImageWidth];
      const float c33 = inputImageData[left + down * mImageWidth];

      const float sum = c11 + c12 + c13 + c21 + c22 + c23 + c31 + c32 + c33;
      const float sum2 = c11 * c11 + c12 * c12 + c13 * c13 + c21 * c21 +
                         c22 * c22 + c23 * c23 + c31 * c31 + c32 * c32 +
                         c33 * c33;

      const float mu = sqrt((sum2 - sum * sum / 9.0) / 9.0);

      float weight = mStandardDeviationWeight * mu;

      mIrlsWeights[col + row * mImageWidth] = weight;
    }
  }

  return;
}

void MaskLaplaceIterationRegularizer::calculateIrlsWeights(
    const point_value_t* inputImageData) {}

void MaskLaplaceIterationRegularizer::calculateIrlsWeightsX(
    const point_value_t* inputImageData) {}

void MaskLaplaceIterationRegularizer::calculateIrlsWeightsY(
    const point_value_t* inputImageData) {}

}  // namespace Deblurring
}  // namespace Test
