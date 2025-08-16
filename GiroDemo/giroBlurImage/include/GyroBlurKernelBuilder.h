/*
 * GyroBlurKernelBuilder.h
 *
 *  Created on: Jul 25, 2014
 *      Author: vladiant
 */

#pragma once

#include <GyroDataCorrection.h>
#include <Time.h>
#include <Versor.h>

#include <vector>

#include "GyroBlurParams.h"
#include "SparseBlurKernel.h"

namespace Test {
namespace Deblurring {

class GyroBlurKernelBuilder {
 public:
  GyroBlurKernelBuilder(const GyroBlurParams& gyroBlurParams,
                        const std::vector<Test::OpticalFlow::TimeOrientation<
                            Test::Math::Versor<float> > >& gyroSamples);
  virtual ~GyroBlurKernelBuilder();

  void calculateAtPoint(int imageCol, int imageRow,
                        SparseBlurKernel& blurKernel);

  float getKernelXMax() const { return mKernelXMax; }

  float getKernelXMin() const { return mKernelXMin; }

  float getKernelYMax() const { return mKernelYMax; }

  float getKernelYMin() const { return mKernelYMin; }

  int getImageWidth() const { return mGyroBlurParams.getImageWidth(); }

  int getImageHeight() const { return mGyroBlurParams.getImageHeight(); }

 private:
  /// Container of the gyro blurring parameters.
  const GyroBlurParams& mGyroBlurParams;

  /// Samples of angular positions as versors together with time stamps.
  const std::vector<
      Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> > >&
      mGyroSamples;

  /// Optical center of the camera image in pixels - X coordinate.
  float mOptCentX;

  /// Optical center of the camera image in pixels - Y coordinate.
  float mOptCentY;

  /// Field of the of the the camera in X direction.
  float mFovX;

  /// Field of the of the the camera in Y direction.
  float mFovY;

  /// Time stamp of capture end.
  Test::Time mTimeCapture;

  /// Exposure duration.
  Test::Time mTimeExposure;

  /// Readout of rolling shutter duration.
  Test::Time mTimeReadout;

  /// Delay of time stamps.
  Test::Time mTimeDelay;

  /// Duration of the rolling shutter per row.
  Test::Time mRollingShutterDuration;

  /// Maximal X coordinate of the kernel.
  float mKernelXMax;

  /// Maximal Y coordinate of the kernel.
  float mKernelYMax;

  /// Minimal X coordinate of the kernel.
  float mKernelXMin;

  /// Minimal Y coordinate of the kernel.
  float mKernelYMin;

  /// Base coordinates (zero point) for kernel estimation
  float mZeroPointX, mZeroPointY;

  // Timestamps of the interpolation grid.
  std::vector<Test::Time> mTimeGrid;

  /// (x,y) coordinates of kernel interpolation grid
  std::vector<float> mGridPointsX;
  std::vector<float> mGridPointsY;

  /// Number of steps for interpolation for each cell of grid
  std::vector<float> mGridSteps;

  /// Number of samples for interpolation
  int mNumberGridSamples;

  // Maximal number of interpolation steps between two samples
  int mMaxGridSteps;

  /// Indices: initial point, final point
  int mInitialPointIndex, mFinalPointIndex;

  // First and last coordinates of interpolation grid
  float mIntialGridPointX, mIntialGridPointY, mFinalGridPointX,
      mFinalGridPointY;

  void updateGyroBlurParams();

  void updateInterpolationContainers();

  void initData();

  void initKernelCoordinatesExtrema();

  int getStartIndexforTimeStamp(Test::Time timestamp);

  int getEndIndexforTimeStamp(Test::Time timestamp);

  void calculateKernelZeroPoint(int imageCol, int imageRow);

  void setInterpolationGrid(int imageCol, int imageRow);
};

} /* namespace Deblurring */
} /* namespace Test */
