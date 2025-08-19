/*
 * GyroBlurParams.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: vladiant
 */

#include "GyroBlurParams.h"

namespace Test {
namespace Deblurring {

GyroBlurParams::GyroBlurParams()
    : mTimeCapture(0),
      mTimeExposure(0),
      mTimeReadout(0),
      mTimeDelay(0),
      mRollingShutterDuration(0),
      mRelFovX(0),
      mRelFovY(0),
      mRelOptCentX(0),
      mRelOptCentY(0),
      mImageWidth(0),
      mImageHeight(0) {}

GyroBlurParams::~GyroBlurParams() {}

void GyroBlurParams::setTimeParams(Test::Time timeCapture,
                                   Test::Time timeExposure,
                                   Test::Time timeReadout,
                                   Test::Time timeDelay) {
  mTimeCapture = timeCapture;
  mTimeExposure = timeExposure;
  mTimeReadout = timeReadout;
  mTimeDelay = timeDelay;
  if (mImageHeight != 0) {
    mRollingShutterDuration = timeReadout / mImageHeight;
  } else {
    mRollingShutterDuration = 0;
  }
}

void GyroBlurParams::setCameraParams(int imageWidth, int imageHeight,
                                     float optCentX, float optCentY, float fovX,
                                     float fovY) {
  mImageWidth = imageWidth;
  mImageHeight = imageHeight;
  mRollingShutterDuration = mTimeReadout / mImageHeight;
  mRelOptCentX = optCentX / mImageWidth;
  mRelOptCentY = optCentY / mImageHeight;
  mRelFovX = fovX / mImageWidth;
  mRelFovY = fovY / mImageHeight;
}

void GyroBlurParams::setCameraParams(int imageWidth, int imageHeight,
                                     float fovX, float fovY) {
  setCameraParams(imageWidth, imageHeight, imageWidth / 2.0, imageHeight / 2.0,
                  fovX, fovY);
}

}  // namespace Deblurring
}  // namespace Test
