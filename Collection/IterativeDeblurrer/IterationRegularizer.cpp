/*
 * IterationRegularizer.cpp
 *
 *  Created on: Jan 26, 2015
 *      Author: vantonov
 */

#include "IterationRegularizer.h"

#include <stdlib.h>
#include <string.h>

namespace Test {
namespace Deblurring {

IterationRegularizer::IterationRegularizer(int imageWidth, int imageHeight)
    : isExternalMemoryUsed(false),
      mImageWidth(imageWidth),
      mImageHeight(imageHeight),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mMinimalNormIrls(MINIMAL_IRLS_NORM),
      mMaxIrlsIterations(MAX_IRLS_ITERATIONS),
      mIrlsWeights(NULL),
      mIrlsWeightsX(NULL),
      mIrlsWeightsY(NULL),
      mOldDeblurredImage(NULL),
      mRegularizationWeight(0) {
  initalize(imageWidth, imageHeight, NULL);
}

IterationRegularizer::IterationRegularizer(int imageWidth, int imageHeight,
                                           void* pExternalMemory)
    : isExternalMemoryUsed(true),
      mImageWidth(imageWidth),
      mImageHeight(imageHeight),
      mRegularizationImage(NULL),
      mRegularizationImageTransposed(NULL),
      mMinimalNormIrls(MINIMAL_IRLS_NORM),
      mMaxIrlsIterations(MAX_IRLS_ITERATIONS),
      mIrlsWeights(NULL),
      mIrlsWeightsX(NULL),
      mIrlsWeightsY(NULL),
      mOldDeblurredImage(NULL),
      mRegularizationWeight(0) {
  initalize(imageWidth, imageHeight, pExternalMemory);
}

IterationRegularizer::~IterationRegularizer() { release(); }

size_t IterationRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 2 * imageWidth * imageHeight * sizeof(point_value_t);

  return requiredMemorySize;
}

void IterationRegularizer::initalize(int imageWidth, int imageHeight,
                                     void* pExternalMemory) {
  mImageBlockSize = imageWidth * imageHeight * sizeof(point_value_t);

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mRegularizationImage =
        &pDataBuffer[0 * mImageBlockSize / sizeof(point_value_t)];
    mRegularizationImageTransposed =
        &pDataBuffer[1 * mImageBlockSize / sizeof(point_value_t)];
  } else {
    mRegularizationImage =
        new point_value_t[mImageBlockSize / sizeof(point_value_t)];
    mRegularizationImageTransposed =
        new point_value_t[mImageBlockSize / sizeof(point_value_t)];
  }
}

void IterationRegularizer::release() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationImage;
    delete[] mRegularizationImageTransposed;
  }
  isExternalMemoryUsed = false;

  mImageWidth = 0;
  mImageHeight = 0;

  mRegularizationImage = NULL;
  mRegularizationImageTransposed = NULL;

  mMinimalNormIrls = MINIMAL_IRLS_NORM;
  mMaxIrlsIterations = MAX_IRLS_ITERATIONS;
  mIrlsWeights = NULL;
  mIrlsWeightsX = NULL;
  mIrlsWeightsY = NULL;
  mOldDeblurredImage = NULL;

  mRegularizationWeight = 0;
}

void IterationRegularizer::prepareIrls(
    const point_value_t* currentDeblurredImage) {
  if (0 == mImageBlockSize) {
    return;
  }

  setOldDeblurredImage(currentDeblurredImage);

  initializeIrlsWeight(mIrlsWeights);

  initializeIrlsWeight(mIrlsWeightsX);

  initializeIrlsWeight(mIrlsWeightsY);
}

void IterationRegularizer::setOldDeblurredImage(
    const point_value_t* inputImageData) {
  if (mOldDeblurredImage != NULL) {
    if (0 == mImageBlockSize) {
      return;
    }

    memcpy(mOldDeblurredImage, inputImageData, mImageBlockSize);
  }
}

void IterationRegularizer::initializeIrlsWeight(point_value_t* irlsWeight) {
  if (irlsWeight != NULL) {
    for (unsigned int i = 0; i < mImageBlockSize / sizeof(point_value_t); i++) {
      irlsWeight[i] = 1;
    }
  }
}

}  // namespace Deblurring
}  // namespace Test
