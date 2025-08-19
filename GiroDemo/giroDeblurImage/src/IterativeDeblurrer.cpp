/*
 * IterativeDeblurrer.cpp
 *
 *  Created on: 17.01.2015
 *      Author: vladiant
 */

#include "IterativeDeblurrer.h"

#include <math.h>

namespace Test {
namespace Deblurring {

IterativeDeblurrer::IterativeDeblurrer(int mImageWidth, int mImageHeight)
    : ImageDeblurrer(mImageWidth, mImageHeight),
      mMaxIterations(MAX_ITERATIONS),
      mMinimalNorm(MINIMAL_NORM),
      mCurrentIteration(0),
      mCurrentNorm(0),
      mProcessStatus(NOT_INITIALIZED),
      mRegularizer(NULL) {
  setBorderImposers();
}

IterativeDeblurrer::IterativeDeblurrer(int mImageWidth, int mImageHeight,
                                       void* pExternalMemory)
    : ImageDeblurrer(mImageWidth, mImageHeight, pExternalMemory),
      mMaxIterations(MAX_ITERATIONS),
      mMinimalNorm(MINIMAL_NORM),
      mCurrentIteration(0),
      mCurrentNorm(0),
      mProcessStatus(NOT_INITIALIZED),
      mRegularizer(NULL) {
  setBorderImposers();
}

IterativeDeblurrer::~IterativeDeblurrer() { clear(); }

size_t IterativeDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 0;

  requiredMemorySize += ImageDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void IterativeDeblurrer::clear() {
  mBorderType = BORDER_CONTINUOUS;
  mMaxIterations = MAX_ITERATIONS;
  mMinimalNorm = MINIMAL_NORM;
  mCurrentIteration = 0;
  mCurrentNorm = 0;
  mProcessStatus = NOT_INITIALIZED;
  mRegularizer = NULL;
}

void IterativeDeblurrer::blur(const point_value_t* inputImageData,
                              const point_value_t* kernelData, int kernelWidth,
                              int kernelHeight, int kernelPpln,
                              point_value_t* outputImageData,
                              bool transposeKernel) {
  point_coord_t kernelCenterX = kernelWidth / 2;
  point_coord_t kernelCenterY = kernelHeight / 2;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      point_value_t blurredPixelValue = 0;
      for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
        for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
          point_value_t kernelValue =
              kernelData[kernelCol + kernelRow * kernelPpln];

          if (0 == kernelValue) {
            continue;
          }

          point_coord_t coordX = col;
          point_coord_t coordY = row;

          if (transposeKernel) {
            coordX -= kernelCol - kernelCenterX;
            coordY -= kernelRow - kernelCenterY;
          } else {
            coordX += kernelCol - kernelCenterX;
            coordY += kernelRow - kernelCenterX;
          }

          (this->*mBorderImposer[mBorderType])(coordX, coordY, kernelValue);

          blurredPixelValue +=
              inputImageData[coordX + coordY * mImageWidth] * kernelValue;
        }  // for (int kernelCol ...
      }  // for (int kernelRow ...

      outputImageData[col + row * mImageWidth] = blurredPixelValue;

    }  // for (int col = ...
  }  // for (int row = ...
}

void IterativeDeblurrer::blur(const point_value_t* inputImageData,
                              const SparseBlurKernel& currentBlurKernel,
                              point_value_t* outputImageData,
                              bool transposeKernel) {
  mKernelPointsX.clear();
  mKernelPointsY.clear();
  mKernelPointsValues.clear();

  // TODO: Hack - fix this!
  const_cast<SparseBlurKernel&>(currentBlurKernel)
      .extractKernelPoints(mKernelPointsX, mKernelPointsY, mKernelPointsValues);

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      point_value_t blurredPixelValue = 0;
      for (int k = 0; k < currentBlurKernel.getKernelSize(); k++) {
        point_coord_t coordX = col;
        point_coord_t coordY = row;
        point_value_t kernelValue = mKernelPointsValues[k];

        if (transposeKernel) {
          coordX -= mKernelPointsX[k];
          coordY -= mKernelPointsY[k];
        } else {
          coordX += mKernelPointsX[k];
          coordY += mKernelPointsY[k];
        }

        (this->*mBorderImposer[mBorderType])(coordX, coordY, kernelValue);

        blurredPixelValue +=
            inputImageData[coordX + coordY * mImageWidth] * kernelValue;
      }  // for (int k = ...

      outputImageData[col + row * mImageWidth] = blurredPixelValue;

    }  // for (int col = ...
  }  // for (int row = ...

  return;
}

void IterativeDeblurrer::blur(const point_value_t* inputImageData,
                              const GyroBlurKernelBuilder& currentkernelBuilder,
                              point_value_t* outputImageData,
                              bool transposeKernel) {
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      // Calculate blur kernel.
      mBlurKernel.clear();
      mKernelPointsX.clear();
      mKernelPointsY.clear();
      mKernelPointsValues.clear();

      // TODO: Hack - fix this!
      const_cast<GyroBlurKernelBuilder&>(currentkernelBuilder)
          .calculateAtPoint(col, row, mBlurKernel);

      mBlurKernel.extractKernelPoints(mKernelPointsX, mKernelPointsY,
                                      mKernelPointsValues);

      point_value_t blurredPixelValue = 0;
      for (int k = 0; k < mBlurKernel.getKernelSize(); k++) {
        point_coord_t coordX = col;
        point_coord_t coordY = row;
        point_value_t kernelValue = mKernelPointsValues[k];

        if (transposeKernel) {
          coordX -= mKernelPointsX[k];
          coordY -= mKernelPointsY[k];
        } else {
          coordX += mKernelPointsX[k];
          coordY += mKernelPointsY[k];
        }

        (this->*mBorderImposer[mBorderType])(coordX, coordY, kernelValue);

        blurredPixelValue +=
            inputImageData[coordX + coordY * mImageWidth] * kernelValue;
      }  // for (int k = ...

      outputImageData[col + row * mImageWidth] = blurredPixelValue;

    }  // for (int col = ...
  }  // for (int row = ...

  return;
}

void IterativeDeblurrer::setBorderImposers() {
  mBorderImposer[BORDER_CONTINUOUS] =
      &Test::Deblurring::IterativeDeblurrer::imposeContinuousBorder;
  mBorderImposer[BORDER_CONSTANT] =
      &Test::Deblurring::IterativeDeblurrer::imposeConstantBorder;
  mBorderImposer[BORDER_PERIODIC] =
      &Test::Deblurring::IterativeDeblurrer::imposePeriodicBorder;
  mBorderImposer[BORDER_MIRROR] =
      &Test::Deblurring::IterativeDeblurrer::imposeMirrorBorder;
}

void IterativeDeblurrer::imposeContinuousBorder(point_coord_t& coordX,
                                                point_coord_t& coordY,
                                                point_value_t& kernelValue) {
  int maxCoordX = mImageWidth - 1;
  int maxCoordY = mImageHeight - 1;

  if (coordX < 0) {
    coordX = 0;
  } else if (coordX > maxCoordX) {
    coordX = maxCoordX;
  }

  if (coordY < 0) {
    coordY = 0;
  } else if (coordY > maxCoordY) {
    coordY = maxCoordY;
  }

  return;
}

void IterativeDeblurrer::imposeConstantBorder(point_coord_t& coordX,
                                              point_coord_t& coordY,
                                              point_value_t& kernelValue) {
  int maxCoordX = mImageWidth - 1;
  int maxCoordY = mImageHeight - 1;

  if (coordX < 0) {
    coordX = 0;
    kernelValue = mBorderValue;
  } else if (coordX > maxCoordX) {
    coordX = maxCoordX;
    kernelValue = mBorderValue;
  }

  if (coordY < 0) {
    coordY = 0;
    kernelValue = mBorderValue;
  } else if (coordY > maxCoordY) {
    coordY = maxCoordY;
    kernelValue = mBorderValue;
  }

  return;
}

void IterativeDeblurrer::imposePeriodicBorder(point_coord_t& coordX,
                                              point_coord_t& coordY,
                                              point_value_t& kernelValue) {
  int maxCoordX = mImageWidth - 1;
  int maxCoordY = mImageHeight - 1;

  if (coordX < 0) {
    coordX += maxCoordX;
  } else if (coordX > maxCoordX) {
    coordX -= maxCoordX;
  }

  if (coordY < 0) {
    coordY += maxCoordY;
  } else if (coordY > maxCoordY) {
    coordY -= maxCoordY;
  }

  return;
}

void IterativeDeblurrer::imposeMirrorBorder(point_coord_t& coordX,
                                            point_coord_t& coordY,
                                            point_value_t& kernelValue) {
  int maxCoordX = mImageWidth - 1;
  int maxCoordY = mImageHeight - 1;

  if (coordX < 0) {
    coordX += maxCoordX;
  } else if (coordX > maxCoordX) {
    coordX -= maxCoordX;
  }

  if (coordY < 0) {
    coordY += maxCoordY;
  } else if (coordY > maxCoordY) {
    coordY -= maxCoordY;
  }

  return;
}

void IterativeDeblurrer::subtractImages(const point_value_t* firstImageData,
                                        const point_value_t* secondImageData,
                                        point_value_t* outputImageData) {
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      *outputImageData = *firstImageData - *secondImageData;
      firstImageData++;
      secondImageData++;
      outputImageData++;
    }
  }
}

void IterativeDeblurrer::multiplyImages(const point_value_t* firstImageData,
                                        const point_value_t* secondImageData,
                                        point_value_t* outputImageData) {
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      *outputImageData = (*firstImageData) * (*secondImageData);
      firstImageData++;
      secondImageData++;
      outputImageData++;
    }
  }
}

bool IterativeDeblurrer::divideImages(const point_value_t* firstImageData,
                                      const point_value_t* secondImageData,
                                      point_value_t* outputImageData) {
  bool status = true;  // In case when division by zero cannot be handled here.

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      double firstValue = *firstImageData++;
      double secondValue = *secondImageData++;

      if (secondValue != 0) {
        *outputImageData = firstValue / secondValue;
      } else {
        *outputImageData = 0;  // OpenCV behavior
      }

      outputImageData++;
    }
  }

  return status;
}

void IterativeDeblurrer::addWeightedImages(const point_value_t* firstImageData,
                                           double firstImageWeight,
                                           const point_value_t* secondImageData,
                                           double secondImageWeight,
                                           double offset,
                                           point_value_t* outputImageData) {
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      *outputImageData = (*firstImageData) * firstImageWeight +
                         (*secondImageData) * secondImageWeight + offset;
      firstImageData++;
      secondImageData++;
      outputImageData++;
    }
  }
}

double IterativeDeblurrer::calculateDotProductOfImages(
    const point_value_t* firstImageData, const point_value_t* secondImageData) {
  double dotProduct = 0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      double firstValue = *firstImageData++;
      double secondValue = *secondImageData++;

      dotProduct += firstValue * secondValue;
    }
  }

  return dotProduct;
}

double IterativeDeblurrer::calculateL2Norm(const point_value_t* imageData) {
  double normL2 = 0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      normL2 += (*imageData) * (*imageData);
      imageData++;
    }
  }

  normL2 = sqrt(normL2);

  return normL2;
}

double IterativeDeblurrer::calculateL2NormOfDifference(
    const point_value_t* firstImageData, const point_value_t* secondImageData) {
  double normL2 = 0;

  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      double difference = (*firstImageData++) - (*secondImageData++);
      normL2 += difference * difference;
    }
  }

  normL2 = sqrt(normL2);

  return normL2;
}

void IterativeDeblurrer::scaleImage(const point_value_t* inputImageData,
                                    double imageScale, double offset,
                                    point_value_t* outputImageData) {
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      *outputImageData = (*inputImageData) * imageScale + offset;
      inputImageData++;
      outputImageData++;
    }
  }
}

}  // namespace Deblurring
}  // namespace Test
