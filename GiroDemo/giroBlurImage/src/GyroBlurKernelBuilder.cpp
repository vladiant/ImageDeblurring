/*
 * GyroBlurKernelBuilder.cpp
 *
 *  Created on: Jul 25, 2014
 *      Author: vladiant
 */

#include "GyroBlurKernelBuilder.h"

#include <limits>
#include <typeinfo>

namespace Test {
namespace Deblurring {

GyroBlurKernelBuilder::GyroBlurKernelBuilder(
    const GyroBlurParams& gyroBlurParams,
    const std::vector<
        Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> > >&
        gyroSamples)
    : mGyroBlurParams(gyroBlurParams), mGyroSamples(gyroSamples) {
  initData();

  updateGyroBlurParams();

  updateInterpolationContainers();

  initKernelCoordinatesExtrema();
}

GyroBlurKernelBuilder::~GyroBlurKernelBuilder() {
  // TODO Auto-generated destructor stub
}

void GyroBlurKernelBuilder::updateGyroBlurParams() {
  mOptCentX = mGyroBlurParams.getOptCentX();
  mOptCentY = mGyroBlurParams.getOptCentY();
  mFovX = mGyroBlurParams.getFovX();
  mFovY = mGyroBlurParams.getFovY();
  mTimeCapture = mGyroBlurParams.getTimeCapture();
  mTimeExposure = mGyroBlurParams.getTimeExposure();
  mTimeReadout = mGyroBlurParams.getTimeReadout();
  mTimeDelay = mGyroBlurParams.getTimeDelay();
  mRollingShutterDuration = mGyroBlurParams.getRollingShutterDuration();
}

void GyroBlurKernelBuilder::updateInterpolationContainers() {
  size_t newSize = mGyroSamples.size();

  if (mTimeGrid.size() < newSize) {
    mTimeGrid.resize(newSize);
  }
  if (mGridPointsX.size() < newSize) {
    mGridPointsX.resize(newSize);
  }
  if (mGridPointsY.size() < newSize) {
    mGridPointsY.resize(newSize);
  }
  if (mGridSteps.size() < newSize) {
    mGridSteps.resize(newSize);
  }
}

void GyroBlurKernelBuilder::initKernelCoordinatesExtrema() {
  mKernelXMax = -std::numeric_limits<float>::max();
  mKernelYMax = -std::numeric_limits<float>::max();
  mKernelXMin = std::numeric_limits<float>::max();
  mKernelYMin = std::numeric_limits<float>::max();
}

void GyroBlurKernelBuilder::initData() {
  mZeroPointX = 0;
  mZeroPointY = 0;

  mNumberGridSamples = 0;
  mMaxGridSteps = 0;
  mInitialPointIndex = -1;
  mFinalPointIndex = -1;
  mIntialGridPointX = 0;
  mIntialGridPointY = 0;
  mFinalGridPointX = 0;
  mFinalGridPointY = 0;
}

int GyroBlurKernelBuilder::getStartIndexforTimeStamp(Test::Time timestamp) {
  int timestampIndex = -1;
  for (size_t i = 0; i < mGyroSamples.size() - 1; i++) {
    Time currentTime = mGyroSamples[i].time;
    Time nexTime = mGyroSamples[i + 1].time;
    if ((timestamp >= currentTime) && (timestamp < nexTime)) {
      timestampIndex = i;
      break;
    }
  }
  return timestampIndex;
}

int GyroBlurKernelBuilder::getEndIndexforTimeStamp(Test::Time timestamp) {
  int timestampIndex = -1;
  for (size_t i = 1; i < mGyroSamples.size(); i++) {
    Time currentTime = mGyroSamples[i - 1].time;
    Time nexTime = mGyroSamples[i].time;
    if ((timestamp > currentTime) && (timestamp <= nexTime)) {
      timestampIndex = i;
      break;
    }
  }

  return timestampIndex;
}

void GyroBlurKernelBuilder::calculateKernelZeroPoint(int imageCol,
                                                     int imageRow) {
  int zeroPointIndex =
      getStartIndexforTimeStamp(mGyroBlurParams.getTimeFirstSample(imageRow));

  float currentPointCenteredX = (imageCol - mOptCentX) / mFovX;
  float currentPointCenteredY = (imageRow - mOptCentY) / mFovY;

  float kernelStartPointX, kernelAfterStartPointX, kernelStartPointY,
      kernelAfterStartPointY;

  Test::Math::Versor<float> initialPointVersor =
      mGyroSamples[zeroPointIndex].orientation;
  Test::Math::Vector<float, 3> nextPointVersor = Test::Math::Vector<float, 3>(
      currentPointCenteredX, currentPointCenteredY, 1.0);
  initialPointVersor.rotateVector(nextPointVersor);

  float homographyPointX = nextPointVersor.x() / nextPointVersor.z();
  float homographyPointY = nextPointVersor.y() / nextPointVersor.z();

  // Kernel star point.
  kernelStartPointX = homographyPointX * mFovX + mOptCentX - imageCol;
  kernelStartPointY = homographyPointY * mFovY + mOptCentY - imageRow;
  Time timeZeroPoint = mGyroSamples[zeroPointIndex].time;

  // Next point after start point.
  initialPointVersor = mGyroSamples[zeroPointIndex + 1].orientation;
  nextPointVersor = Test::Math::Vector<float, 3>(currentPointCenteredX,
                                                 currentPointCenteredY, 1.0);
  initialPointVersor.rotateVector(nextPointVersor);

  homographyPointX = nextPointVersor.x() / nextPointVersor.z();
  homographyPointY = nextPointVersor.y() / nextPointVersor.z();
  kernelAfterStartPointX = homographyPointX * mFovX + mOptCentX - imageCol;
  kernelAfterStartPointY = homographyPointY * mFovY + mOptCentY - imageRow;

  Time timeAfterZeroPoint = mGyroSamples[zeroPointIndex + 1].time;

  // Set the initial and final points.
  mZeroPointX = (kernelAfterStartPointX - kernelStartPointX) * 1.0 *
                    (mTimeCapture - mTimeExposure - mTimeReadout + mTimeDelay -
                     timeZeroPoint) /
                    (timeAfterZeroPoint - timeZeroPoint) +
                kernelStartPointX;
  mZeroPointY = (kernelAfterStartPointY - kernelStartPointY) * 1.0 *
                    (mTimeCapture - mTimeExposure - mTimeReadout + mTimeDelay -
                     timeZeroPoint) /
                    (timeAfterZeroPoint - timeZeroPoint) +
                kernelStartPointY;

  return;
}  // void calculateKernelZeroPoint( ...

void GyroBlurKernelBuilder::setInterpolationGrid(int imageCol, int imageRow) {
  float currentPointCenteredX = (imageCol - mOptCentX) / mFovX;
  float currentPointCenteredY = (imageRow - mOptCentY) / mFovY;

  // Calculate the end point indices.
  mInitialPointIndex =
      getStartIndexforTimeStamp(mGyroBlurParams.getTimeFirstSample(imageRow));
  mFinalPointIndex =
      getEndIndexforTimeStamp(mGyroBlurParams.getTimeLastSample(imageRow));

  mNumberGridSamples = mFinalPointIndex - mInitialPointIndex + 1;

  // Calculate the zero point (center of the kernel).
  calculateKernelZeroPoint(imageCol, imageRow);

  for (int j = mInitialPointIndex; j < mFinalPointIndex + 1; j++) {
    Test::Math::Versor<float> currentPointVersor = mGyroSamples[j].orientation;
    Test::Math::Vector<float, 3> zeroPointVersor = Test::Math::Vector<float, 3>(
        currentPointCenteredX, currentPointCenteredY, 1.0);
    currentPointVersor.rotateVector(zeroPointVersor);

    float homographyPointX = zeroPointVersor.x() / zeroPointVersor.z();
    float homographyPointY = zeroPointVersor.y() / zeroPointVersor.z();

    mGridPointsX[j - mInitialPointIndex] =
        homographyPointX * mFovX + mOptCentX - imageCol;
    mGridPointsY[j - mInitialPointIndex] =
        homographyPointY * mFovY + mOptCentY - imageRow;
    mTimeGrid[j - mInitialPointIndex] = mGyroSamples[j].time;
  }

  // Set the initial and final points.
  mIntialGridPointX =
      (mGridPointsX[1] - mGridPointsX[0]) * 1.0 *
          (mTimeCapture - mTimeExposure - mTimeReadout + mTimeDelay +
           mRollingShutterDuration * imageRow - mTimeGrid[0]) /
          (mTimeGrid[1] - mTimeGrid[0]) +
      mGridPointsX[0];
  mIntialGridPointY =
      (mGridPointsY[1] - mGridPointsY[0]) * 1.0 *
          (mTimeCapture - mTimeExposure - mTimeReadout + mTimeDelay +
           mRollingShutterDuration * imageRow - mTimeGrid[0]) /
          (mTimeGrid[1] - mTimeGrid[0]) +
      mGridPointsY[0];

  int indexLastPoint = mNumberGridSamples - 1;
  int indexBeforeLastPoint = mNumberGridSamples - 2;

  mFinalGridPointX =
      (mGridPointsX[indexLastPoint] - mGridPointsX[indexBeforeLastPoint]) *
          1.0 *
          (mTimeCapture - mTimeReadout + mTimeDelay +
           mRollingShutterDuration * imageRow -
           mTimeGrid[indexBeforeLastPoint]) /
          (mTimeGrid[indexLastPoint] - mTimeGrid[indexBeforeLastPoint]) +
      mGridPointsX[indexBeforeLastPoint];
  mFinalGridPointY =
      (mGridPointsY[indexLastPoint] - mGridPointsY[indexBeforeLastPoint]) *
          1.0 *
          (mTimeCapture - mTimeReadout + mTimeDelay +
           mRollingShutterDuration * imageRow -
           mTimeGrid[indexBeforeLastPoint]) /
          (mTimeGrid[indexLastPoint] - mTimeGrid[indexBeforeLastPoint]) +
      mGridPointsY[indexBeforeLastPoint];

  // Set the number of steps decrease for the first and the last line.
  float initialStepCorrection =
      sqrt(((mGridPointsX[1] - mIntialGridPointX) *
                (mGridPointsX[1] - mIntialGridPointX) +
            (mGridPointsY[1] - mIntialGridPointY) *
                (mGridPointsY[1] - mIntialGridPointY)) /
           ((mGridPointsX[1] - mGridPointsX[0]) *
                (mGridPointsX[1] - mGridPointsX[0]) +
            (mGridPointsY[1] - mGridPointsY[0]) *
                (mGridPointsY[1] - mGridPointsY[0])));
  float finalStepCorrection = sqrt(
      ((mFinalGridPointX - mGridPointsX[indexBeforeLastPoint]) *
           (mFinalGridPointX - mGridPointsX[indexBeforeLastPoint]) +
       (mFinalGridPointY - mGridPointsY[indexBeforeLastPoint]) *
           (mFinalGridPointY - mGridPointsY[indexBeforeLastPoint])) /
      ((mGridPointsX[indexLastPoint] - mGridPointsX[indexBeforeLastPoint]) *
           (mGridPointsX[indexLastPoint] - mGridPointsX[indexBeforeLastPoint]) +
       (mGridPointsY[indexLastPoint] - mGridPointsY[indexBeforeLastPoint]) *
           (mGridPointsY[indexLastPoint] -
            mGridPointsY[indexBeforeLastPoint])));

  // calculate number of steps
  mMaxGridSteps = 0;
  for (int j = 0; j < indexLastPoint; j++) {
    mGridSteps[j] =
        int(2.0 * sqrt((mGridPointsX[j + 1] - mGridPointsX[j]) *
                           (mGridPointsX[j + 1] - mGridPointsX[j]) +
                       (mGridPointsY[j + 1] - mGridPointsY[j]) *
                           (mGridPointsY[j + 1] - mGridPointsY[j])) +
            0.5);
    if (mMaxGridSteps < mGridSteps[j]) mMaxGridSteps = mGridSteps[j];
  }

  /// The same number of steps for all.
  for (int j = 0; j < indexLastPoint; j++) mGridSteps[j] = mMaxGridSteps;

  // Correct the number of steps for the first and last line
  mGridSteps[0] *= initialStepCorrection;
  mGridSteps[indexBeforeLastPoint] *= finalStepCorrection;

  // The final set of points.
  initKernelCoordinatesExtrema();

  mGridPointsX[0] = mIntialGridPointX - mZeroPointX;
  mGridPointsY[0] = mIntialGridPointY - mZeroPointY;

  if (mGridPointsX[0] > mKernelXMax) {
    mKernelXMax = mGridPointsX[0];
  }
  if (mGridPointsX[0] < mKernelXMin) {
    mKernelXMin = mGridPointsX[0];
  }
  if (mGridPointsY[0] > mKernelYMax) {
    mKernelYMax = mGridPointsY[0];
  }
  if (mGridPointsY[0] < mKernelYMin) {
    mKernelYMin = mGridPointsY[0];
  }
  mGridPointsX[indexLastPoint] = mFinalGridPointX - mZeroPointX;
  mGridPointsY[indexLastPoint] = mFinalGridPointY - mZeroPointY;

  if (mGridPointsX[indexLastPoint] > mKernelXMax) {
    mKernelXMax = mGridPointsX[indexLastPoint];
  }
  if (mGridPointsX[indexLastPoint] < mKernelXMin) {
    mKernelXMin = mGridPointsX[indexLastPoint];
  }
  if (mGridPointsY[indexLastPoint] > mKernelYMax) {
    mKernelYMax = mGridPointsY[indexLastPoint];
  }
  if (mGridPointsY[indexLastPoint] < mKernelYMin) {
    mKernelYMin = mGridPointsY[indexLastPoint];
  }

  for (int j = 1; j < indexLastPoint; j++) {
    mGridPointsX[j] -= mZeroPointX;
    mGridPointsY[j] -= mZeroPointY;
    if (mGridPointsX[j] > mKernelXMax) {
      mKernelXMax = mGridPointsX[j];
    }
    if (mGridPointsX[j] < mKernelXMin) {
      mKernelXMin = mGridPointsX[j];
    }
    if (mGridPointsY[j] > mKernelYMax) {
      mKernelYMax = mGridPointsY[j];
    }
    if (mGridPointsY[j] < mKernelYMin) {
      mKernelYMin = mGridPointsY[j];
    }
  }

  return;
}  // void interpolateKernel( ...

void GyroBlurKernelBuilder::calculateAtPoint(int imageCol, int imageRow,
                                             SparseBlurKernel& blurKernel) {
  blurKernel.clear();

  // This is required only when data is changed.
  // Lock here for thread safe execution.
  updateGyroBlurParams();
  updateInterpolationContainers();

  setInterpolationGrid(imageCol, imageRow);

  int indexLastPoint = mNumberGridSamples - 1;
  // Set the kernel values
  for (int j = 0; j < indexLastPoint; j++) {
    point_value_t step = 1.0 / mMaxGridSteps;
    for (int jj = 0; jj < mGridSteps[j]; jj++) {
      // Points for bilinear interpolation.
      // Minus sign is used to mimic mirror movement
      // of the camera and image point.
      point_value_t kernelPointX =
          -1.0 * ((mGridPointsX[j + 1] - mGridPointsX[j]) * jj / mGridSteps[j] +
                  mGridPointsX[j]);
      point_value_t kernelPointY =
          -1.0 * ((mGridPointsY[j + 1] - mGridPointsY[j]) * jj / mGridSteps[j] +
                  mGridPointsY[j]);

      // Top left point.
      blurKernel.addToPointValue(floor(kernelPointX), floor(kernelPointY),
                                 (floor(kernelPointX) - kernelPointX + 1) *
                                     (floor(kernelPointY) - kernelPointY + 1) *
                                     step);

      // Bottom left point.
      blurKernel.addToPointValue(floor(kernelPointX), floor(kernelPointY + 1),
                                 (floor(kernelPointX) - kernelPointX + 1) *
                                     (kernelPointY - floor(kernelPointY)) *
                                     step);

      // Top right point.
      blurKernel.addToPointValue(floor(kernelPointX + 1), floor(kernelPointY),
                                 (kernelPointX - floor(kernelPointX)) *
                                     (floor(kernelPointY) - kernelPointY + 1) *
                                     step);

      // Bottom right point.
      blurKernel.addToPointValue(
          floor(kernelPointX + 1), floor(kernelPointY + 1),
          (kernelPointX - floor(kernelPointX)) *
              (kernelPointY - floor(kernelPointY)) * step);
    }
  }

  // Normalize kernel data.
  blurKernel.normalize();

  return;
}  // void GyroBlurKernelBuilder::calculateAtPoint( ...

} /* namespace Deblurring */
} /* namespace Test */
