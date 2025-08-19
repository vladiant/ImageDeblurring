/*
 * HlIterationRegularizer.cpp
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#include "HLIterationRegularizer.h"

#include <math.h>
#include <string.h>

namespace Test {
namespace Deblurring {

HLIterationRegularizer::HLIterationRegularizer(int imageWidth, int imageHeight)
    : IterationRegularizer(imageWidth, imageHeight),
      mPowerRegularizationNorm(0) {
  init(imageWidth, imageHeight, NULL);
}

HLIterationRegularizer::HLIterationRegularizer(int imageWidth, int imageHeight,
                                               void* pExternalMemory)
    : IterationRegularizer(imageWidth, imageHeight, pExternalMemory),
      mPowerRegularizationNorm(0) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterationRegularizer::getMemorySize(imageWidth, imageHeight)));
}

HLIterationRegularizer::~HLIterationRegularizer() { deinit(); }

size_t HLIterationRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterationRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void HLIterationRegularizer::init(int imageWidth, int imageHeight,
                                  void* pExternalMemory) {
  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mIrlsWeights = &pDataBuffer[0 * mImageBlockSize / sizeof(point_value_t)];
    mIrlsWeightsX = &pDataBuffer[1 * mImageBlockSize / sizeof(point_value_t)];
    mIrlsWeightsY = &pDataBuffer[2 * mImageBlockSize / sizeof(point_value_t)];
    mOldDeblurredImage =
        &pDataBuffer[3 * mImageBlockSize / sizeof(point_value_t)];
  } else {
    mIrlsWeights = new point_value_t[mImageBlockSize / sizeof(point_value_t)];
    mIrlsWeightsX = new point_value_t[mImageBlockSize / sizeof(point_value_t)];
    mIrlsWeightsY = new point_value_t[mImageBlockSize / sizeof(point_value_t)];
    mOldDeblurredImage =
        new point_value_t[mImageBlockSize / sizeof(point_value_t)];
  }
}

void HLIterationRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mIrlsWeights;
    delete[] mIrlsWeightsX;
    delete[] mIrlsWeightsY;
    delete[] mOldDeblurredImage;
  }

  mIrlsWeights = NULL;
  mIrlsWeightsX = NULL;
  mIrlsWeightsY = NULL;
  mOldDeblurredImage = NULL;
}

void HLIterationRegularizer::calculateRegularization(
    const point_value_t* inputImageData, point_value_t* outputImageData,
    bool transposeKernel) {
  memset(outputImageData, 0, mImageBlockSize);
}

void HLIterationRegularizer::calculateRegularizationX(
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

void HLIterationRegularizer::calculateRegularizationY(
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

void HLIterationRegularizer::calculateIrlsWeights(
    const point_value_t* inputImageData) {
  const float cUpLeftX = -3.0;
  const float cLeftX = -10.0;
  const float cDownLeftX = -3.0;
  const float cUpRightX = 3.0;
  const float cRightX = 10.0;
  const float cDownRightX = 3.0;

  const float cUpLeftY = -3.0;
  const float cUpY = -10.0;
  const float cUpRightY = -3.0;
  const float cDownLeftY = 3.0;
  const float cDownY = 10.0;
  const float cDownRightY = 3.0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      point_value_t pointValueX =
          inputImageData[left + up * mImageWidth] * cUpLeftX +
          inputImageData[left + row * mImageWidth] * cLeftX +
          inputImageData[left + down * mImageWidth] * cDownLeftX +
          inputImageData[right + up * mImageWidth] * cUpRightX +
          inputImageData[right + row * mImageWidth] * cRightX +
          inputImageData[right + down * mImageWidth] * cDownRightX;

      point_value_t pointValueY =
          inputImageData[left + up * mImageWidth] * cUpLeftY +
          inputImageData[col + up * mImageWidth] * cUpY +
          inputImageData[left + down * mImageWidth] * cDownLeftY +
          inputImageData[right + up * mImageWidth] * cUpRightY +
          inputImageData[col + down * mImageWidth] * cDownY +
          inputImageData[right + down * mImageWidth] * cDownRightY;

      point_value_t pointValue =
          pointValueX * pointValueX + pointValueY * pointValueY;

      pointValue = pow(pointValue, mPowerRegularizationNorm);

      if (isnan(pointValue) || isinf(pointValue)) {
        pointValue = 0;
      }

      pointValue = pointValue > 1.0 ? 1.0 : pointValue;

      mIrlsWeights[col + row * mImageWidth] = pointValue;
    }
  }

  return;
}

void HLIterationRegularizer::calculateIrlsWeightsX(
    const point_value_t* inputImageData) {
  const float cUpLeft = -3.0;
  const float cLeft = -10.0;
  const float cDownLeft = -3.0;
  const float cUpRight = 3.0;
  const float cRight = 10.0;
  const float cDownRight = 3.0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      point_value_t pointValue =
          inputImageData[left + up * mImageWidth] * cUpLeft +
          inputImageData[left + row * mImageWidth] * cLeft +
          inputImageData[left + down * mImageWidth] * cDownLeft +
          inputImageData[right + up * mImageWidth] * cUpRight +
          inputImageData[right + row * mImageWidth] * cRight +
          inputImageData[right + down * mImageWidth] * cDownRight;

      if (pointValue != 0) {
        pointValue = 1.0 / fabs(pointValue);
      }

      pointValue = pointValue > 1.0 ? 1.0 : pointValue;

      mIrlsWeightsX[col + row * mImageWidth] = pointValue;
    }
  }

  return;
}

void HLIterationRegularizer::calculateIrlsWeightsY(
    const point_value_t* inputImageData) {
  const float cUpLeft = -3.0;
  const float cUp = -10.0;
  const float cUpRight = -3.0;
  const float cDownLeft = 3.0;
  const float cDown = 10.0;
  const float cDownRight = 3.0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      int left = col > 0 ? col - 1 : 0;
      int right = col < mImageWidth - 1 ? col + 1 : mImageWidth - 1;
      int up = row > 0 ? row - 1 : 0;
      int down = row < mImageHeight - 1 ? row + 1 : mImageHeight - 1;

      point_value_t pointValue =
          inputImageData[left + up * mImageWidth] * cUpLeft +
          inputImageData[col + up * mImageWidth] * cUp +
          inputImageData[left + down * mImageWidth] * cDownLeft +
          inputImageData[right + up * mImageWidth] * cUpRight +
          inputImageData[col + down * mImageWidth] * cDown +
          inputImageData[right + down * mImageWidth] * cDownRight;

      if (pointValue != 0) {
        pointValue = 1.0 / fabs(pointValue);
      }

      pointValue = pointValue > 1.0 ? 1.0 : pointValue;

      mIrlsWeightsY[col + row * mImageWidth] = pointValue;
    }
  }

  return;
}

}  // namespace Deblurring
}  // namespace Test
