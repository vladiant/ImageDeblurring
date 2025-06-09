/*
 *  GyroBlurKernel.h
 *
 *  This class is used to create the
 *  kernel of blurring
 *  using the gyro data,
 *  time and camera parameters
 *
 *  Created on: Apr 21, 2012
 *      Author: vantonov
 */

#pragma once

#include "DiscreteAngularVelocitiesIntegrator.h"
#include "GyroDataCorrection.h"
#include "Versor.h"

typedef struct {
  // Time parameters
  // timestamp which marks frame capture end
  // acquired from the embedded gyro data
  Test::Time TimeCapture;

  // frame exposure time in Time units
  Test::Time TimeExposure;

  // frame readout time in  Time units
  Test::Time TimeReadout;

  // = frameTimestampDelayMinusGyroTimestampDelay
  Test::Time TimeDelay;

  // Camera parameters:
  // Field of view in x
  // multiply it to image width to obtain the focal length in x pixels
  float FOVx;

  // Field of view in x
  // multiply it to image width to obtain the focal length in y pixels
  float FOVy;

  // x coordinate of principal optical point
  float OptCentX;

  // y coordinate of principal optical point
  float OptCentY;

  // frame width
  int ImageWidth;

  // frame height
  int ImageHeight;

} GyroDeblurParams;

class GyroBlurKernel {
 private:
  // Maximal difference between pixels
  static constexpr float MAXPIXELDIFF = 0.2;

  // time parameters, given at class construction
  // see description of GyroDeblurParams
  Test::Time TimeCapture;
  Test::Time TimeExposure;
  Test::Time TimeReadout;
  Test::Time TimeDelay;

  // camera parameters, given at class construction
  // see description of GyroDeblurParams
  float FOVx;
  float FOVy;
  float OptCentX;
  float OptCentY;
  int ImageWidth;
  int ImageHeight;

  // Number of used gyro samples
  int UsedGyroSamples;

  // Pointer to container for the gyro data
  Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator<
      Test::Math::Versor<float>, float>* AngVelIntegr;

  // Indices: initial point, final point, first used sample, last used sample
  int j_i, j_f, j_0, j_1;

  // Rolling shutter duration
  Test::Time rollingShutterDuration;

  // Base coordinates (zero point) for kernel estimation
  float x0b, y0b;

  // Number of samples for interpolation
  int bsamples;

  // First and last coordinates of interpolation grid
  float xbi, ybi, xbf, ybf;

  // (x,y) coordinates of kernel interpolation grid
  float *xb, *yb;

  // Discrete time for interpolation grid
  Test::Time* timeb;

  // Check to free the memory from interpolated data values
  bool isInterpolated;

  // Number of steps for interpolation for each cell of grid
  int* bstep;

  // Maximal number of interpolation steps between two samples
  int bstepsmax;

  // Maximal (x,y) span of the kernel
  float xbmax, ybmax;

  // Image fo blur
  cv::Mat BlurKernel;

  // Check if kernel is created
  bool isKernelCreated;

  // Sparse kernel dimensions
  int SKernelDims[2];

  // Sparse kernel
  cv::SparseMat SparseBlurKernel;

  // Check if sparse kernel is created
  bool isSparseKernelCreated;

  // Sample indices of the first and last gyro data points
  void EndPoints(void);

  // Sample indices of the first and last gyro data points, used for kernel
  // interpolation for a given row
  void InterpolationEndPoints(int Row);

  // Center of the kernel point calculation
  void ZeroPoint(int Row, int Col);

  // Interpolation procedure
  void KernelInterpolation(int Row, int Col);

 public:
  GyroBlurKernel(
      GyroDeblurParams Params,
      Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator<
          Test::Math::Versor<float>, float>& angularVelocityIntegrator,
      int UsedGyroSamples);
  ~GyroBlurKernel();

  // Kernel drawing
  cv::Mat Kernel(int Row, int Col);

  // Sparse kernel drawing
  cv::SparseMat SparseKernel(int Row, int Col);

  // Maximal span of the kernel
  void MaxKernelSpan(int& xkspan, int& ykspan);

  // Maximal size of the blocks, where kernel has less than one pixel change
  void MaxBlockSize(int& xkspan, int& ykspan);
};

GyroBlurKernel::GyroBlurKernel(
    GyroDeblurParams Params,
    Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator<
        Test::Math::Versor<float>, float>& angularVelocityIntegrator,
    int GyroSamples) {
  // Initial initialization
  TimeCapture = Params.TimeCapture;
  TimeExposure = Params.TimeExposure;
  TimeReadout = Params.TimeReadout;
  TimeDelay = Params.TimeDelay;
  FOVx = Params.FOVx;
  FOVy = Params.FOVy;
  OptCentX = Params.OptCentX;
  OptCentY = Params.OptCentY;
  ImageWidth = Params.ImageWidth;
  ImageHeight = Params.ImageHeight;

  UsedGyroSamples = GyroSamples;

  rollingShutterDuration = TimeReadout / ImageHeight;

  AngVelIntegr = &angularVelocityIntegrator;

  isKernelCreated = false;

  isInterpolated = false;

  isSparseKernelCreated = false;

  EndPoints();
}

GyroBlurKernel::~GyroBlurKernel() {
  // Clear the memory
  if (isInterpolated) {
    delete[] timeb;
    delete[] xb;
    delete[] yb;
    delete[] bstep;
  }
}

// Sample indices of the first and last gyro data points
void GyroBlurKernel::EndPoints(void) {
  bool flag1 = 1, flag2 = 1;

  for (int j = 0; j < UsedGyroSamples - 1; j++) {
    Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> >
        CurrentSample = AngVelIntegr->orientations().orientations()[j];

    if ((CurrentSample.time >
         TimeCapture - TimeExposure - TimeReadout + TimeDelay) &&
        (flag1)) {
      j_0 = j - 1;
      flag1 = 0;
    }

    if ((CurrentSample.time > TimeCapture + TimeDelay) && (flag2)) {
      j_1 = j;
      flag2 = 0;
      break;
    }
  }
}

// Sample indices of the first and last gyro data points, used for kernel
// interpolation for a given row
void GyroBlurKernel::InterpolationEndPoints(int Row) {
  bool flag1 = 1, flag2 = 1;

  for (int j = j_0; j < j_1 + 1; j++) {
    Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> >
        CurrentSample = AngVelIntegr->orientations().orientations()[j];

    if ((CurrentSample.time > TimeCapture - TimeExposure - TimeReadout +
                                  rollingShutterDuration * Row + TimeDelay) &&
        (flag1)) {
      j_i = j - 1;
      flag1 = 0;
    }

    if ((CurrentSample.time > TimeCapture - TimeReadout +
                                  rollingShutterDuration * Row + TimeDelay) &&
        (flag2)) {
      j_f = j;
      flag2 = 0;
      break;
    }
  }

  bsamples = j_f - j_i + 1;
}

// Center of the kernel point calculation
void GyroBlurKernel::ZeroPoint(int Row, int Col) {
  float x0 = (Col - OptCentX) / (FOVx * ImageWidth),
        y0 = (Row - OptCentY) / (FOVy * ImageHeight);
  float x0b0, x0b1, y0b0, y0b1;

  Test::Math::Versor<float> CurrentVersor =
      AngVelIntegr->orientations().orientations()[j_0].orientation;
  Test::Math::Vector<float, 3> VectorVersor =
      Test::Math::Vector<float, 3>(x0, y0, 1.0);
  CurrentVersor.rotateVector(VectorVersor);

  float x1 = VectorVersor.x() / VectorVersor.z();
  float y1 = VectorVersor.y() / VectorVersor.z();

  x0b0 = x1 * (FOVx * ImageWidth) + OptCentX - Col;
  y0b0 = y1 * (FOVy * ImageHeight) + OptCentY - Row;
  Test::Time time_j0 = AngVelIntegr->orientations().orientations()[j_0].time;

  CurrentVersor =
      AngVelIntegr->orientations().orientations()[j_0 + 1].orientation;
  VectorVersor = Test::Math::Vector<float, 3>(x0, y0, 1.0);
  CurrentVersor.rotateVector(VectorVersor);
  x1 = VectorVersor.x() / VectorVersor.z();
  y1 = VectorVersor.y() / VectorVersor.z();
  x0b1 = x1 * (FOVx * ImageWidth) + OptCentX - Col;
  y0b1 = y1 * (FOVy * ImageHeight) + OptCentY - Row;
  Test::Time time_j01 =
      AngVelIntegr->orientations().orientations()[j_0 + 1].time;

  // set the initial and final points
  x0b = (x0b1 - x0b0) * 1.0 *
            (TimeCapture - TimeExposure - TimeReadout + TimeDelay - time_j0) /
            (time_j01 - time_j0) +
        x0b0;
  y0b = (y0b1 - y0b0) * 1.0 *
            (TimeCapture - TimeExposure - TimeReadout + TimeDelay - time_j0) /
            (time_j01 - time_j0) +
        y0b0;
}

// Interpolation procedure
void GyroBlurKernel::KernelInterpolation(int Row, int Col) {
  float x0 = (Col - OptCentX) / (FOVx * ImageWidth),
        y0 = (Row - OptCentY) / (FOVy * ImageHeight);

  // Calculate the end point indices
  InterpolationEndPoints(Row);

  bsamples = j_f - j_i + 1;

  if (isInterpolated) {
    delete[] timeb;
    delete[] xb;
    delete[] yb;
    delete[] bstep;
  }

  timeb = new Test::Time[bsamples];
  xb = new float[bsamples];
  yb = new float[bsamples];
  bstep = new int[bsamples - 1];
  isInterpolated = true;

  // Calculate the zero point (center of the kernel)
  ZeroPoint(Row, Col);

  for (int j = j_i; j < j_f + 1; j++) {
    Test::Math::Versor<float> CurrentVersor =
        AngVelIntegr->orientations().orientations()[j].orientation;
    Test::Math::Vector<float, 3> VectorVersor =
        Test::Math::Vector<float, 3>(x0, y0, 1.0);
    CurrentVersor.rotateVector(VectorVersor);

    float x1 = VectorVersor.x() / VectorVersor.z();
    float y1 = VectorVersor.y() / VectorVersor.z();

    xb[j - j_i] = x1 * (FOVx * ImageWidth) + OptCentX - Col;
    yb[j - j_i] = y1 * (FOVy * ImageHeight) + OptCentY - Row;
    timeb[j - j_i] = AngVelIntegr->orientations().orientations()[j].time;
  }

  // set the initial and final points
  xbi = (xb[1] - xb[0]) * 1.0 *
            (TimeCapture - TimeExposure - TimeReadout + TimeDelay +
             rollingShutterDuration * Row - timeb[0]) /
            (timeb[1] - timeb[0]) +
        xb[0];
  ybi = (yb[1] - yb[0]) * 1.0 *
            (TimeCapture - TimeExposure - TimeReadout + TimeDelay +
             rollingShutterDuration * Row - timeb[0]) /
            (timeb[1] - timeb[0]) +
        yb[0];

  xbf = (xb[bsamples - 1] - xb[bsamples - 2]) * 1.0 *
            (TimeCapture - TimeReadout + TimeDelay +
             rollingShutterDuration * Row - timeb[bsamples - 2]) /
            (timeb[bsamples - 1] - timeb[bsamples - 2]) +
        xb[bsamples - 2];
  ybf = (yb[bsamples - 1] - yb[bsamples - 2]) * 1.0 *
            (TimeCapture - TimeReadout + TimeDelay +
             rollingShutterDuration * Row - timeb[bsamples - 2]) /
            (timeb[bsamples - 1] - timeb[bsamples - 2]) +
        yb[bsamples - 2];

  // set the number of steps decrease for the first and the last line
  float bstepsi = sqrt(
      ((xb[1] - xbi) * (xb[1] - xbi) + (yb[1] - ybi) * (yb[1] - ybi)) /
      ((xb[1] - xb[0]) * (xb[1] - xb[0]) + (yb[1] - yb[0]) * (yb[1] - yb[0])));
  float bstepsf = sqrt(((xbf - xb[bsamples - 2]) * (xbf - xb[bsamples - 2]) +
                        (ybf - yb[bsamples - 2]) * (ybf - yb[bsamples - 2])) /
                       ((xb[bsamples - 1] - xb[bsamples - 2]) *
                            (xb[bsamples - 1] - xb[bsamples - 2]) +
                        (yb[bsamples - 1] - yb[bsamples - 2]) *
                            (yb[bsamples - 1] - yb[bsamples - 2])));

  bstepsmax = 0;
  // calculate number of steps
  for (int j = 0; j < bsamples - 1; j++) {
    bstep[j] = int(2.0 * sqrt((xb[j + 1] - xb[j]) * (xb[j + 1] - xb[j]) +
                              (yb[j + 1] - yb[j]) * (yb[j + 1] - yb[j])) +
                   0.5);
    if (bstepsmax < bstep[j]) bstepsmax = bstep[j];
  }

  // same number of steps for all
  for (int j = 0; j < bsamples - 1; j++) bstep[j] = bstepsmax;
  // correct the number f steps for the first and last line
  bstep[0] *= bstepsi;
  bstep[bsamples - 2] *= bstepsf;

  // the final set of points
  xbmax = 0, ybmax = 0;

  xb[0] = xbi - x0b;
  yb[0] = ybi - y0b;
  xb[bsamples - 1] = xbf - x0b;
  yb[bsamples - 1] = ybf - y0b;
  for (int j = 1; j < bsamples - 1; j++) {
    xb[j] -= x0b;
    yb[j] -= y0b;
    if (abs(xb[j]) > abs(xbmax)) xbmax = xb[j];
    if (abs(yb[j]) > abs(ybmax)) ybmax = yb[j];
  }

  if (abs(xb[0]) > abs(xbmax)) xbmax = xb[0];
  if (abs(yb[0]) > abs(ybmax)) ybmax = yb[0];
  if (abs(xb[bsamples - 1]) > abs(xbmax)) xbmax = xb[bsamples - 1];
  if (abs(yb[bsamples - 1]) > abs(ybmax)) ybmax = yb[bsamples - 1];
}

// Kernel drawing
cv::Mat GyroBlurKernel::Kernel(int Row, int Col) {
  KernelInterpolation(Row, Col);

  BlurKernel = cv::Mat(
      cv::Size(2 * int(abs(xbmax) + 0.5) + 3, 2 * int(abs(ybmax) + 0.5) + 3),
      CV_32FC1);
  isKernelCreated = true;
  BlurKernel = 0;

  // set the kernel values
  for (int j = 0; j < bsamples - 1; j++) {
    float step = 1.0 / bstepsmax;
    for (int jj = 0; jj < bstep[j]; jj++) {
      float xd =
          ((xb[j + 1] - xb[j]) * jj / bstep[j] + xb[j]) + BlurKernel.cols / 2;
      float yd =
          ((yb[j + 1] - yb[j]) * jj / bstep[j] + yb[j]) + BlurKernel.rows / 2;
      ((float*)(BlurKernel.data + int(yd) * (BlurKernel.step)))[int(xd)] +=
          (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
      ((float*)(BlurKernel.data + int(yd + 1) * (BlurKernel.step)))[int(xd)] +=
          (int(xd) - xd + 1) * (yd - int(yd)) * step;
      ((float*)(BlurKernel.data + int(yd) * (BlurKernel.step)))[int(xd + 1)] +=
          (xd - int(xd)) * (int(yd) - yd + 1) * step;
      ((float*)(BlurKernel.data +
                int(yd + 1) * (BlurKernel.step)))[int(xd + 1)] +=
          (xd - int(xd)) * (yd - int(yd)) * step;
    }
  }

  // normalize
  float kernelsum = 0;
  for (int row = 0; row < BlurKernel.rows; row++)
    for (int col = 0; col < BlurKernel.cols; col++)
      kernelsum += ((float*)(BlurKernel.data + row * (BlurKernel.step)))[col];

  for (int row = 0; row < BlurKernel.rows; row++)
    for (int col = 0; col < BlurKernel.cols; col++)
      ((float*)(BlurKernel.data + row * (BlurKernel.step)))[col] /= kernelsum;

  return (BlurKernel);
}

// Sparse kernel drawing
cv::SparseMat GyroBlurKernel::SparseKernel(int Row, int Col) {
  KernelInterpolation(Row, Col);

  SKernelDims[0] = 2 * int(abs(xbmax) + 0.5) + 3;
  SKernelDims[1] = 2 * int(abs(ybmax) + 0.5) + 3;

  SparseBlurKernel = cv::SparseMat(2, SKernelDims, CV_32FC1);
  isSparseKernelCreated = true;

  // set the kernel values
  for (int j = 0; j < bsamples - 1; j++) {
    float step = 1.0 / bstepsmax;
    for (int jj = 0; jj < bstep[j]; jj++) {
      float xd =
          ((xb[j + 1] - xb[j]) * jj / bstep[j] + xb[j]) + SKernelDims[0] / 2;
      float yd =
          ((yb[j + 1] - yb[j]) * jj / bstep[j] + yb[j]) + SKernelDims[1] / 2;
      // sparse kernel
      int idx[] = {int(xd), int(yd)};
      SparseBlurKernel.ref<float>(idx) +=
          (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
      idx[0] = int(xd);
      idx[1] = int(yd + 1);
      SparseBlurKernel.ref<float>(idx) +=
          (int(xd) - xd + 1) * (yd - int(yd)) * step;
      idx[0] = int(xd + 1);
      idx[1] = int(yd);
      SparseBlurKernel.ref<float>(idx) +=
          (xd - int(xd)) * (int(yd) - yd + 1) * step;
      idx[0] = int(xd + 1);
      idx[1] = int(yd + 1);
      SparseBlurKernel.ref<float>(idx) +=
          (xd - int(xd)) * (yd - int(yd)) * step;
    }
  }

  float kernelsum = 0;
  // sparse kernel sum
  cv::SparseMatIterator it;
  for (it = SparseBlurKernel.begin(); it != SparseBlurKernel.end(); ++it) {
    kernelsum += it.value<float>();
  }

  // normalize sparse kernel
  for (it = SparseBlurKernel.begin(); it != SparseBlurKernel.end(); ++it) {
    it.value<float>() /= kernelsum;
  }

  return (SparseBlurKernel);
}

// Maximal span of the kernel
void GyroBlurKernel::MaxKernelSpan(int& xkspan, int& ykspan) {
  int xmax = 0, ymax = 0;
  for (int row = 0; row < ImageHeight; row += 10)
    for (int col = 0; col < ImageHeight; col += 10) {
      KernelInterpolation(row, col);
      int x0 = 2 * int(abs(xbmax) + 0.5) + 3;
      if (x0 > xmax) xmax = x0;
      int y0 = 2 * int(abs(ybmax) + 0.5) + 3;
      if (y0 > ymax) ymax = y0;
    }
  xkspan = xmax;
  ykspan = ymax;
}

// Maximal size of the blocks, where kernel has less than one pixel change
void GyroBlurKernel::MaxBlockSize(int& xkspan, int& ykspan) {
  int stepx = 5, stepy = 5;
  bool flag;
  float diff;

  flag = true;
  do {
    stepx += 5;

    for (int row = 0; row < ImageHeight - stepx; row += stepx)
      for (int col = 0; col < ImageHeight - stepy; col += stepy) {
        KernelInterpolation(row, col);
        float xk1 = xbmax;
        float yk1 = ybmax;
        KernelInterpolation(row + stepx, col);
        float xk2 = xbmax;
        float yk2 = ybmax;
        KernelInterpolation(row, col + stepy);
        float xk3 = xbmax;
        float yk3 = ybmax;
        KernelInterpolation(row + stepx, col + stepy);
        float xk4 = xbmax;
        float yk4 = ybmax;
        float diff1 =
            sqrt((xk2 - xk1) * (xk2 - xk1) + (yk2 - yk1) * (yk2 - yk1));
        float diff2 =
            sqrt((xk3 - xk1) * (xk3 - xk1) + (yk3 - yk1) * (yk3 - yk1));
        float diff3 =
            sqrt((xk4 - xk1) * (xk4 - xk1) + (yk4 - yk1) * (yk4 - yk1));
        diff = std::max(diff1, diff2);
        diff = std::max(diff, diff3);
      }

    if ((diff >= MAXPIXELDIFF) && (flag))
      break;
    else
      flag = false;

  } while (diff < MAXPIXELDIFF);  // less than one pixel

  stepx -= 5;

  flag = true;
  do {
    stepy += 5;

    for (int row = 0; row < ImageHeight - stepx; row += stepx)
      for (int col = 0; col < ImageHeight - stepy; col += stepy) {
        KernelInterpolation(row, col);
        float xk1 = xbmax;
        float yk1 = ybmax;
        KernelInterpolation(row + stepx, col);
        float xk2 = xbmax;
        float yk2 = ybmax;
        KernelInterpolation(row, col + stepy);
        float xk3 = xbmax;
        float yk3 = ybmax;
        KernelInterpolation(row + stepx, col + stepy);
        float xk4 = xbmax;
        float yk4 = ybmax;
        float diff1 =
            sqrt((xk2 - xk1) * (xk2 - xk1) + (yk2 - yk1) * (yk2 - yk1));
        float diff2 =
            sqrt((xk3 - xk1) * (xk3 - xk1) + (yk3 - yk1) * (yk3 - yk1));
        float diff3 =
            sqrt((xk4 - xk1) * (xk4 - xk1) + (yk4 - yk1) * (yk4 - yk1));
        diff = std::max(diff1, diff2);
        diff = std::max(diff, diff3);
      }

    if ((diff >= MAXPIXELDIFF) && (flag))
      break;
    else
      flag = false;

  } while (diff < MAXPIXELDIFF);  // less than one pixel

  stepy -= 5;

  ykspan = stepy;
  xkspan = stepx;
}
