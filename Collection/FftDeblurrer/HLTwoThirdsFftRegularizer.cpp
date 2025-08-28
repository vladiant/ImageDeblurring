/*
 * HLTwoThirdsFftRegularizer.cpp
 *
 *  Created on: Feb 5, 2015
 *      Author: vantonov
 */

#include "HLTwoThirdsFftRegularizer.h"

#include <math.h>
#include <stdlib.h>

#include <complex>

namespace Test {
namespace Deblurring {

HLTwoThirdsFftRegularizer::HLTwoThirdsFftRegularizer(int imageWidth,
                                                     int imageHeight)
    : FftRegularizer(imageWidth, imageHeight) {
  init(imageWidth, imageHeight, NULL);
}

HLTwoThirdsFftRegularizer::HLTwoThirdsFftRegularizer(int imageWidth,
                                                     int imageHeight,
                                                     void* pExternalMemory)
    : FftRegularizer(imageWidth, imageHeight, pExternalMemory) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftRegularizer::getMemorySize(imageWidth, imageHeight)));
}

HLTwoThirdsFftRegularizer::~HLTwoThirdsFftRegularizer() { deinit(); }

size_t HLTwoThirdsFftRegularizer::getMemorySize(int imageWidth,
                                                int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 8 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += FftRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void HLTwoThirdsFftRegularizer::init([[maybe_unused]] int imageWidth,
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

void HLTwoThirdsFftRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationFftImageX;
    delete[] mRegularizationFftImageY;
    delete[] mHqpmWeightsFftImageX;
    delete[] mHqpmWeightsFftImageY;
  }

  mRegularizationFftImage = NULL;
}

void HLTwoThirdsFftRegularizer::setRegularization() {}

void HLTwoThirdsFftRegularizer::setRegularizationX() {
  calculateGradientXFft2D(mRegularizationFftImageX);
}

void HLTwoThirdsFftRegularizer::setRegularizationY() {
  calculateGradientYFft2D(mRegularizationFftImageY);
}

void HLTwoThirdsFftRegularizer::calculateHqpmWeights(
    [[maybe_unused]] const point_value_t* inputFftImageData,
    [[maybe_unused]] float beta) {}

void HLTwoThirdsFftRegularizer::calculateHqpmWeightsX(
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

void HLTwoThirdsFftRegularizer::calculateHqpmWeightsY(
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

void HLTwoThirdsFftRegularizer::calculateHqpmWeight(
    const point_value_t* inputImageData, float beta,
    point_value_t* outputImageData) {
  for (int row = 0; row < mFftImageHeight; row++) {
    for (int col = 0; col < mFftImageWidth; col++) {
      float valueX = inputImageData[2 * (col + row * mFftImageWidth)];
      valueX = WeightCalcAlphaTwoThirds(valueX, beta);
      outputImageData[2 * (col + row * mFftImageWidth) + 0] = valueX;
      outputImageData[2 * (col + row * mFftImageWidth) + 1] = 0;
    }
  }
}

float HLTwoThirdsFftRegularizer::WeightCalcAlphaTwoThirds(float v, float beta) {
  const float EPSILON = 1e-6;

  float value = 0;  // return value

  float signV = 0.0;
  if (v > 0.0) {
    signV = 1.0;
  } else if (v < 0.0) {
    signV = -1.0;
  }

  const float k = 8.0 / (27.0 * beta * beta * beta);
  const float m = k;

  // Now use formula from
  // http://en.wikipedia.org/wiki/Quartic_equation (Ferrari's method)
  // running our coefficients through Mathmetica (quartic_solution.nb)
  // optimized to use as few operations as possible...

  const float v2 = v * v;
  const float v3 = v2 * v;
  const float v4 = v3 * v;
  const float m2 = m * m;
  const float m3 = m2 * m;

  // Compute alpha & beta
  const float alpha = -1.125 * v2;
  const float beta2 = 0.25 * v3;

  // Compute p,q,r and u directly.
  const float q = -0.125 * (m * v2);
  const std::complex<float> r1 =
      -q / 2.0f + sqrt(-m3 / 27.0f + (m2 * v4) / 256.0f);

  const std::complex<float> u = exp(log(r1) / 3.0f);
  const std::complex<float> y =
      2.0f * ((-5.0f / 18.0f) * alpha + u + (m / (3.0f * u)));

  const std::complex<float> W = sqrt(alpha / 3.0f + y);

  // now form all 4 roots
  std::complex<float> root[4];
  root[0] = 0.75f * v + 0.5f * (W + sqrt(-(alpha + y + beta2 / W)));
  root[1] = 0.75f * v + 0.5f * (W - sqrt(-(alpha + y + beta2 / W)));
  root[2] = 0.75f * v + 0.5f * (-W + sqrt(-(alpha + y - beta2 / W)));
  root[3] = 0.75f * v + 0.5f * (-W - sqrt(-(alpha + y - beta2 / W)));

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

  if (isnan(root[3].real()) || isinf(root[3].real())) {
    root[3].real(3);
  }
  if (isnan(root[3].imag()) || isinf(root[3].imag())) {
    root[3].imag(3);
  }

  // Now pick the correct root, including zero option.

  // Clever fast approach that avoids lookups
  float rsv2[4];
  rsv2[0] = root[0].real() * signV;
  rsv2[1] = root[1].real() * signV;
  rsv2[2] = root[2].real() * signV;
  rsv2[3] = root[3].real() * signV;

  // condensed fast version
  // take out imaginary roots above v/2 but below v

  bool c1[4];
  c1[0] = fabs(root[0].imag()) < EPSILON;
  c1[1] = fabs(root[1].imag()) < EPSILON;
  c1[2] = fabs(root[2].imag()) < EPSILON;
  c1[3] = fabs(root[3].imag()) < EPSILON;

  bool c2[4];
  c2[0] = rsv2[0] > 0.5 * fabs(v);
  c2[1] = rsv2[1] > 0.5 * fabs(v);
  c2[2] = rsv2[2] > 0.5 * fabs(v);
  c2[3] = rsv2[3] > 0.5 * fabs(v);

  bool c3[4];
  c3[0] = rsv2[0] < fabs(v);
  c3[1] = rsv2[1] < fabs(v);
  c3[2] = rsv2[2] < fabs(v);
  c3[3] = rsv2[3] < fabs(v);

  float c[4];
  c[0] = (c1[0] && c2[0] && c3[0]) * rsv2[0];
  c[1] = (c1[1] && c2[1] && c3[1]) * rsv2[1];
  c[2] = (c1[2] && c2[2] && c3[2]) * rsv2[2];
  c[3] = (c1[3] && c2[3] && c3[3]) * rsv2[3];

  int index = -1;
  index = c[0] > c[1] ? 0 : 1;
  index = c[2] > c[index] ? 2 : index;
  index = c[3] > c[index] ? 3 : index;

  // take best
  value = signV * c[index];

  return value;
}

}  // namespace Deblurring
}  // namespace Test
