/*
 * TvFftRegularizer.cpp
 *
 *  Created on: Feb 5, 2015
 *      Author: vantonov
 */

#include "TvFftRegularizer.h"

#include <math.h>
#include <stdlib.h>

#include <complex>

namespace Test {
namespace Deblurring {

TvFftRegularizer::TvFftRegularizer(int imageWidth, int imageHeight)
    : FftRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

TvFftRegularizer::TvFftRegularizer(int imageWidth, int imageHeight,
                                   void* pExternalMemory)
    : FftRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftRegularizer::getMemorySize(imageWidth, imageHeight)));
}

TvFftRegularizer::~TvFftRegularizer() { deinit(); }

size_t TvFftRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 8 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void TvFftRegularizer::init([[maybe_unused]] int imageWidth,
                            [[maybe_unused]] int imageHeight,
                            void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mRegularizationFftImageX = &pDataBuffer[0 * bufferSize];
    mRegularizationFftImageY = &pDataBuffer[1 * bufferSize];
    mHqpmWeightsFftImageX = &pDataBuffer[2 * bufferSize];
    mHqpmWeightsFftImageY = &pDataBuffer[3 * bufferSize];
  } else {
    mRegularizationFftImageX = new point_value_t[bufferSize];
    mRegularizationFftImageY = new point_value_t[bufferSize];
    mHqpmWeightsFftImageX = new point_value_t[bufferSize];
    mHqpmWeightsFftImageY = new point_value_t[bufferSize];
  }

  setRegularizationX();
  setRegularizationY();
}

void TvFftRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationFftImageX;
    delete[] mRegularizationFftImageY;
    delete[] mHqpmWeightsFftImageX;
    delete[] mHqpmWeightsFftImageY;
  }

  mRegularizationFftImage = NULL;
}

void TvFftRegularizer::setRegularization() {}

void TvFftRegularizer::setRegularizationX() {
  calculateGradientXFft2D(mRegularizationFftImageX);
}

void TvFftRegularizer::setRegularizationY() {
  calculateGradientYFft2D(mRegularizationFftImageY);
}

void TvFftRegularizer::calculateHqpmWeights(
    [[maybe_unused]] const point_value_t* inputFftImageData,
    [[maybe_unused]] float beta) {}

void TvFftRegularizer::calculateHqpmWeightsX(
    const point_value_t* inputFftImageData, float beta) {
  FftDeblurrer::multiplyFftImages(inputFftImageData, mRegularizationFftImageX,
                                  mFftImageWidth, mFftImageHeight,
                                  mHqpmWeightsFftImageX);

  calculateFft2D(mHqpmWeightsFftImageX, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_INVERSE);

  calculateHqpmWeight(mHqpmWeightsFftImageX, beta, mHqpmWeightsFftImageX);

  calculateFft2D(mHqpmWeightsFftImageX, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void TvFftRegularizer::calculateHqpmWeightsY(
    const point_value_t* inputFftImageData, float beta) {
  FftDeblurrer::multiplyFftImages(inputFftImageData, mRegularizationFftImageY,
                                  mFftImageWidth, mFftImageHeight,
                                  mHqpmWeightsFftImageY);

  calculateFft2D(mHqpmWeightsFftImageY, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_INVERSE);

  calculateHqpmWeight(mHqpmWeightsFftImageY, beta, mHqpmWeightsFftImageY);

  calculateFft2D(mHqpmWeightsFftImageY, mFftImageWidth, mFftImageHeight,
                 FftDeblurrer::FFT_FORWARD);
}

void TvFftRegularizer::calculateHqpmWeight(const point_value_t* inputImageData,
                                           float beta,
                                           point_value_t* outputImageData) {
  for (int row = 0; row < mFftImageHeight; row++) {
    for (int col = 0; col < mFftImageWidth; col++) {
      float valueX = inputImageData[2 * (col + row * mFftImageWidth)];
      valueX = WeightCalcAlphaTv(valueX, beta);
      outputImageData[2 * (col + row * mFftImageWidth) + 0] = valueX;
      outputImageData[2 * (col + row * mFftImageWidth) + 1] = 0;
    }
  }
}

float TvFftRegularizer::WeightCalcAlphaTv(float v, float beta) {
  float value = 0;  // return value

  const float EPSILON = 1e-6;

  float signV = 0.0;
  if (v > 0.0) {
    signV = 1.0;
  } else if (v < 0.0) {
    signV = -1.0;
  }

  float k = -1.0 / (4.0 * beta * beta);
  float m = k * signV;

  // Compute the roots (all 3)
  float t1 = (2.0 / 3.0) * v;

  float v2 = v * v;
  float v3 = v2 * v;

  // slow (50% of time), not clear how to speed up...
  const float b1 = 3.0f * sqrt(3.0f);
  std::complex<float> t2 =
      pow(std::complex<float>(
              -27.0f * m - 2.0f * v3 +
              b1 * sqrt(std::complex<float>(27.0f * m * m + 4.0f * m * v3))),
          1.0f / 3.0f);

  std::complex<float> t3 = v2 / t2;

  // find all 3 roots
  std::complex<float> root[3];
  const std::complex<float> a1(1.0, 3.0);
  const std::complex<float> a2(1.0, -3.0);
  const float b2 = pow(2.0, 1.0 / 3.0);
  const float b3 = pow(2.0, 2.0 / 3.0);

  root[0] = t1 + (b2 / 3.0f) * t3 + (t2 / (3.0f * b2));

  root[1] = t1 - (a1 / (3.0f * b3)) * t3 - (a2 / (6.0f * b2)) * t2;

  root[2] = t1 - (a2 / (3.0f * b3)) * t3 - (a1 / (6.0f * b2)) * t2;

  // catch 0/0 case
  if (isnan(root[0].real()) || isinf(root[0].real())) {
    root[0].real(0);
  }
  if (isnan(root[0].imag()) || isinf(root[0].imag())) {
    root[0].imag(0);
  }

  if (isnan(root[1].real()) || isinf(root[1].real())) {
    root[1].real(1);
  }
  if (isnan(root[1].imag()) || isinf(root[1].imag())) {
    root[1].imag(1);
  }

  if (isnan(root[2].real()) || isinf(root[2].real())) {
    root[2].real(2);
  }
  if (isnan(root[2].imag()) || isinf(root[2].imag())) {
    root[2].imag(2);
  }

  // Pick the right root
  // Clever fast approach that avoids lookups
  float rsv2[3];
  rsv2[0] = root[0].real() * signV;
  rsv2[1] = root[1].real() * signV;
  rsv2[2] = root[2].real() * signV;

  // condensed fast version
  // take out imaginary roots above v/2 but below v
  bool c1[3];
  c1[0] = fabs(root[0].imag()) < EPSILON;
  c1[1] = fabs(root[1].imag()) < EPSILON;
  c1[2] = fabs(root[2].imag()) < EPSILON;

  bool c2[3];
  c2[0] = rsv2[0] > (2.0 / 3.0) * fabs(v);
  c2[1] = rsv2[1] > (2.0 / 3.0) * fabs(v);
  c2[2] = rsv2[2] > (2.0 / 3.0) * fabs(v);

  bool c3[3];
  c3[0] = rsv2[0] < fabs(v);
  c3[1] = rsv2[1] < fabs(v);
  c3[2] = rsv2[2] < fabs(v);

  float c[3];
  c[0] = (c1[0] && c2[0] && c3[0]) * rsv2[0];
  c[1] = (c1[1] && c2[1] && c3[1]) * rsv2[1];
  c[2] = (c1[2] && c2[2] && c3[2]) * rsv2[2];

  int index = -1;
  index = c[0] > c[1] ? 0 : 1;
  index = c[2] > c[index] ? 2 : index;

  // take best
  value = signV * c[index];

  return value;
}

float TvFftRegularizer::WeightCalcAlphaTvOther(float v, float beta) {
  // http://www-old.me.gatech.edu/energy/andy_phd/appA

  float value = 0;  // return value

  const float maxValue = fabs(v);
  const float minValue = fabs(v) * (2.0 / 3.0);

  float signV = 0.0;
  if (v > 0.0) {
    signV = 1.0;
  } else if (v < 0.0) {
    signV = -1.0;
  }

  const float C1 = 1.0;
  const float C2 = -2 * v;
  const float C3 = v * v;
  const float C4 = -signV / (4.0 * beta * beta);

  const float p = C2 / C1;
  const float q = C3 / C1;
  const float r = C4 / C1;

  const float A = (3.0 * q - p * p) / 3.0;
  const float B = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

  const float D = (A * A * A / 27.0) + (B * B / 4.0);

  if (D > 0) {
    const float M = pow(-0.5 * B + sqrt(D), 1.0 / 3.0);
    const float N = pow(-0.5 * B - sqrt(D), 1.0 / 3.0);

    const float x1 = M + N - p / 3.0;

    const float mult1 = sqrt(3.0) / 2.0;
    const std::complex<float> x2(-0.5 * (M + N) - p / 3.0, mult1 * (M - N));
    const std::complex<float> x3(-0.5 * (M + N) - p / 3.0, -mult1 * (M - N));

    value = x1 * signV;
    if ((value <= minValue) || (value > maxValue)) {
      value = 0;
    }

  } else if (D < 0) {
    float phi = 0.0;

    if (B > 0) {
      phi = acos(-sqrt(0.25 * B * B / (-A * A * A / 27.0)));
    } else {
      phi = acos(sqrt(0.25 * B * B / (-A * A * A / 27.0)));
    }

    const float x1 =
        2.0 * sqrt(-A / 3.0) * cos(phi + 0.0 * M_PI * 2.0 / 3.0) - p / 3.0;
    const float x2 =
        2.0 * sqrt(-A / 3.0) * cos(phi + 1.0 * M_PI * 2.0 / 3.0) - p / 3.0;
    const float x3 =
        2.0 * sqrt(-A / 3.0) * cos(phi + 2.0 * M_PI * 2.0 / 3.0) - p / 3.0;

    if ((x1 * signV > minValue) && (x1 * signV < maxValue)) {
      value = x1 * signV;
    }

    if ((x2 * signV > minValue) && (x2 * signV < maxValue) &&
        (x2 * signV > value)) {
      value = x2 * signV;
    }

    if ((x3 * signV > minValue) && (x3 * signV < maxValue) &&
        (x3 * signV > value)) {
      value = x3 * signV;
    }

  } else {
    const float M = pow(-0.5 * B + sqrt(D), 1.0 / 3.0);
    const float N = pow(-0.5 * B - sqrt(D), 1.0 / 3.0);

    const float x1 = M + N - p / 3.0;

    const float x2 = -0.5 * (M + N) - p / 3.0;

    //		const float x3 = x2;

    if ((x1 > minValue) && (x1 < maxValue)) {
      value = x1;
    }

    if ((x2 > minValue) && (x2 < maxValue) && (x2 > value)) {
      value = x2;
    }
  }

  return value;
}

}  // namespace Deblurring
}  // namespace Test
