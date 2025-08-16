/*
 * GyroBlurParams.h
 *
 *  Created on: Jul 18, 2014
 *      Author: vladiant
 */

#pragma once

#include <Time.h>

namespace Test {
namespace Deblurring {

class GyroBlurParams {
 public:
  GyroBlurParams();
  virtual ~GyroBlurParams();

  float getFovX() const { return mRelFovX * mImageWidth; }

  float getRelativeFovX() const { return mRelFovX; }

  void setFovX(float fovX) { mRelFovX = fovX / mImageWidth; }

  void setRelativeFovX(float relativeFovX) { mRelFovX = relativeFovX; }

  float getFovY() const { return mRelFovY * mImageHeight; }

  float getRelativeFovY() const { return mRelFovY; }

  void setFovY(float fovY) { mRelFovY = fovY / mImageHeight; }

  void setRelativeFovY(float relativeFovY) { mRelFovY = relativeFovY; }

  int getImageHeight() const { return mImageHeight; }

  void setImageHeight(int imageHeight) {
    mRollingShutterDuration = mTimeReadout / mImageHeight;
    mImageHeight = imageHeight;
  }

  int getImageWidth() const { return mImageWidth; }

  void setImageWidth(int imageWidth) { mImageWidth = imageWidth; }

  float getOptCentX() const { return mRelOptCentX; }

  void setOptCentX(float optCentX) { mRelOptCentX = optCentX; }

  float getOptCentY() const { return mRelOptCentY; }

  void setOptCentY(float optCentY) { mRelOptCentY = optCentY; }

  Test::Time getTimeCapture() const { return mTimeCapture; }

  void setTimeCapture(Test::Time timeCapture) { mTimeCapture = timeCapture; }

  Test::Time getTimeDelay() const { return mTimeDelay; }

  void setTimeDelay(Test::Time timeDelay) { mTimeDelay = timeDelay; }

  Test::Time getTimeExposure() const { return mTimeExposure; }

  void setTimeExposure(Test::Time timeExposure) {
    mTimeExposure = timeExposure;
  }

  Test::Time getTimeReadout() const { return mTimeReadout; }

  void setTimeReadout(Test::Time timeReadout) {
    if (mImageHeight != 0) {
      mRollingShutterDuration = timeReadout / mImageHeight;
    } else {
      mRollingShutterDuration = 0;
    }
    mTimeReadout = timeReadout;
  }

  void setTimeParams(Test::Time timeCapture, Test::Time timeExposure,
                     Test::Time timeReadout, Test::Time timeDelay);

  void setCameraParams(int imageWidth, int imageHeight, float optCentX,
                       float optCentY, float fovX, float fovY);

  void setCameraParams(int imageWidth, int imageHeight, float fovX, float fovY);

  Test::Time getTimeFirstSample() const {
    return mTimeCapture - mTimeExposure - mTimeReadout + mTimeDelay;
  }

  Test::Time getTimeLastSample() const { return mTimeCapture + mTimeDelay; }

  Test::Time getTimeFirstSample(int imageRow) const {
    return mTimeCapture - mTimeExposure - mTimeReadout +
           mRollingShutterDuration * imageRow + mTimeDelay;
  }

  Test::Time getTimeLastSample(int imageRow) const {
    return mTimeCapture - mTimeReadout + mRollingShutterDuration * imageRow +
           mTimeDelay;
  }

  Test::Time getRollingShutterDuration() const {
    return mRollingShutterDuration;
  }

 private:
  // Time parameters
  /// Timestamp which marks frame capture end
  /// acquired from the embedded gyro data
  Test::Time mTimeCapture;

  /// Frame exposure time in TestGyroVStab Time units
  Test::Time mTimeExposure;

  /// Frame readout time in  TestGyroVStab Time units
  Test::Time mTimeReadout;

  /// frameTimestampDelayMinusGyroTimestampDelay
  Test::Time mTimeDelay;

  /// Rolling shutter time per row.
  Test::Time mRollingShutterDuration;

  // Camera parameters:
  /// Field of view in X
  // multiply it to image width to obtain the focal length in x pixels
  float mRelFovX;

  /// Field of view in Y
  // multiply it to image width to obtain the focal length in y pixels
  float mRelFovY;

  // X coordinate of principal optical point
  float mRelOptCentX;

  // Y coordinate of principal optical point
  float mRelOptCentY;

  /// Frame width
  int mImageWidth;

  /// Frame height
  int mImageHeight;
};

}  // namespace Deblurring
}  // namespace Test
