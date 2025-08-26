/*
 * ImageDeblurrer.cpp
 *
 *  Created on: 30.11.2014
 *      Author: vladiant
 */

#include "ImageDeblurrer.h"

#include <math.h>

namespace Test {
namespace Deblurring {

ImageDeblurrer::ImageDeblurrer(int imageWidth, int imageHeight)
    : isExternalMemoryUsed(false),
      mImageWidth(imageWidth),
      mImageHeight(imageHeight),
      mBorderType(BORDER_CONTINUOUS),
      mBorderValue(0) {}

ImageDeblurrer::ImageDeblurrer(int imageWidth, int imageHeight,
                               [[maybe_unused]] void* pExternalMemory)
    : isExternalMemoryUsed(true),
      mImageWidth(imageWidth),
      mImageHeight(imageHeight),
      mBorderType(BORDER_CONTINUOUS),
      mBorderValue(0) {}

size_t ImageDeblurrer::getMemorySize([[maybe_unused]] int imageWidth,
                                     [[maybe_unused]] int imageHeight) {
  return 0;
}

ImageDeblurrer::~ImageDeblurrer() { release(); }

void ImageDeblurrer::release() {
  isExternalMemoryUsed = false;

  mImageWidth = 0;
  mImageHeight = 0;

  mBorderType = BORDER_CONTINUOUS;
}

void ImageDeblurrer::convertFromInput(const uint8_t* inputImageData,
                                      int imageWidth, int imageHeight,
                                      int imagePpln,
                                      point_value_t* inputBuffer) {
  uint8_t* pInputData = (uint8_t*)inputImageData;
  point_value_t* pOutputData = inputBuffer;

  for (int row = 0; row < imageHeight; row++) {
    for (int col = 0; col < imageWidth; col++) {
      point_value_t value = *pInputData++;
      value /= 255.0;
      *pOutputData++ = value;
    }
    pInputData += imagePpln - imageWidth;
  }
}

void ImageDeblurrer::convertToOutput(const point_value_t* outputBuffer,
                                     int imageWidth, int imageHeight,
                                     int imagePpln, uint8_t* outputImageData) {
  point_value_t* pInputData = (point_value_t*)outputBuffer;
  uint8_t* pOutputData = outputImageData;

  for (int row = 0; row < imageHeight; row++) {
    for (int col = 0; col < imageWidth; col++) {
      point_value_t value = *pInputData++;
      //			value *= 255.0;
      value = round(value * 255.0);
      if (value < 0) {
        value = 0;
      } else if (value > 255) {
        value = 255;
      }
      *pOutputData++ = value;
    }
    pOutputData += imagePpln - imageWidth;
  }
}

}  // namespace Deblurring
}  // namespace Test
