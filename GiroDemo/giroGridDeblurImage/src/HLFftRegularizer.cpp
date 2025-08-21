/*
 * HLFftRegularizer.cpp
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#include "HLFftRegularizer.h"

#include <math.h>
#include <stdlib.h>

#include <complex>
#include <limits>

namespace Test {
namespace Deblurring {

HLFftRegularizer::HLFftRegularizer(int imageWidth, int imageHeight)
    : FftRegularizer(imageWidth, imageHeight),
      mPowerRegularizationNorm(0),
      mWeightLut(NULL) {
  init(imageWidth, imageHeight, NULL);
}

HLFftRegularizer::HLFftRegularizer(int imageWidth, int imageHeight,
                                   void* pExternalMemory)
    : FftRegularizer(imageWidth, imageHeight, pExternalMemory),
      mPowerRegularizationNorm(0),
      mWeightLut(NULL) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               FftRegularizer::getMemorySize(imageWidth, imageHeight)));
}

HLFftRegularizer::~HLFftRegularizer() { deinit(); }

size_t HLFftRegularizer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageWidth);
  imageHeight = FftDeblurrer::calculateOptimalFftSize(
      (1.0 + 2.0 * FftDeblurrer::BORDERS_PADDING) * imageHeight);

  int requiredMemorySize = 8 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize += WEIGHT_LUT_SIZE * sizeof(point_value_t);

  requiredMemorySize += FftRegularizer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void HLFftRegularizer::init(int imageWidth, int imageHeight,
                            void* pExternalMemory) {
  int bufferSize = 2 * mFftImageWidth * mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mRegularizationFftImageX = &pDataBuffer[0 * bufferSize];
    mRegularizationFftImageY = &pDataBuffer[1 * bufferSize];
    mHqpmWeightsFftImageX = &pDataBuffer[2 * bufferSize];
    mHqpmWeightsFftImageY = &pDataBuffer[3 * bufferSize];
    mWeightLut = &pDataBuffer[4 * bufferSize];
  } else {
    mRegularizationFftImageX = new point_value_t[bufferSize];
    mRegularizationFftImageY = new point_value_t[bufferSize];
    mHqpmWeightsFftImageX = new point_value_t[bufferSize];
    mHqpmWeightsFftImageY = new point_value_t[bufferSize];
    mWeightLut = new point_value_t[WEIGHT_LUT_SIZE];
  }

  setRegularizationX();
  setRegularizationY();
}

void HLFftRegularizer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mRegularizationFftImageX;
    delete[] mRegularizationFftImageY;
    delete[] mHqpmWeightsFftImageX;
    delete[] mHqpmWeightsFftImageY;
    delete[] mWeightLut;
  }

  mPowerRegularizationNorm = 0;
  mWeightLut = NULL;

  mRegularizationFftImage = NULL;
}

void HLFftRegularizer::setRegularization() {}

void HLFftRegularizer::setRegularizationX() {
  calculateGradientXFft2D(mRegularizationFftImageX);
}

void HLFftRegularizer::setRegularizationY() {
  calculateGradientYFft2D(mRegularizationFftImageY);
}

void HLFftRegularizer::calculateHqpmWeights(
    const point_value_t* inputFftImageData, float beta) {}

void HLFftRegularizer::calculateHqpmWeightsX(
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

void HLFftRegularizer::calculateHqpmWeightsY(
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

void HLFftRegularizer::calculateHqpmWeight(const point_value_t* inputImageData,
                                           float beta,
                                           point_value_t* outputImageData) {
  prepareWeightLut(beta);

  for (int row = 0; row < mFftImageHeight; row++) {
    for (int col = 0; col < mFftImageWidth; col++) {
      float value = inputImageData[2 * (col + row * mFftImageWidth)];
      value = WeightCalcAlpha(value, beta);
      outputImageData[2 * (col + row * mFftImageWidth) + 0] = value;
      outputImageData[2 * (col + row * mFftImageWidth) + 1] = 0;
    }
  }
}

float HLFftRegularizer::WeightCalcAlpha(float v, float beta) {
  //	mPowerRegularizationNorm;

  int index = (v + 16) * WEIGHT_LUT_SIZE / 32;

  if (index < 0) {
    return mWeightLut[0];

  } else if (index >= WEIGHT_LUT_SIZE - 1) {
    return mWeightLut[WEIGHT_LUT_SIZE - 1];

  } else {
    float indexValue = 32.0 * (index - WEIGHT_LUT_SIZE / 2.0) / WEIGHT_LUT_SIZE;
    float interpolationWeight = v - indexValue;

    float value =
        mWeightLut[index] +
        interpolationWeight * (mWeightLut[index + 1] - mWeightLut[index]);
    return value;
  }
}

double HLFftRegularizer::weightFunction(double w, double alpha, double beta,
                                        double v) {
  return v - alpha * pow(fabs(w), alpha - 2.0) * w / beta;
}

double HLFftRegularizer::solveWeightFunction(double startX, double alpha,
                                             double beta, double v) {
  const double EPSILON = 1e-6;
  const int MAX_ITERATIONS = 100;

  double oldSolution = startX;
  double solution = oldSolution;

  //	double bestSolution = solution;
  //	double bestDiff = std::numeric_limits<float>::max();
  double currentDiff = 0;

  int iteration = 0;
  do {
    oldSolution = solution;
    solution = weightFunction(solution, alpha, beta, v);
    iteration++;

    currentDiff = fabs(solution - oldSolution);

    //		if (bestDiff > currentDiff) {
    //			bestDiff = currentDiff;
    //			bestSolution = solution;
    //		}

  } while (currentDiff > EPSILON && iteration < MAX_ITERATIONS);

  if (isnan(solution) != 0) {
    solution = 0.0;
  }

  if (iteration >= MAX_ITERATIONS) {
    //		solution = bestSolution;
    solution = 0;
  }

  return solution;
}

void HLFftRegularizer::prepareWeightLut(float beta) {
  static float calculatedBeta = 0;

  if (beta != calculatedBeta) {
    calculatedBeta = beta;
  } else {
    return;
  }

  for (int i = 0; i < WEIGHT_LUT_SIZE; i++) {
    float value = 32.0 * (i - WEIGHT_LUT_SIZE / 2.0) / WEIGHT_LUT_SIZE;

    float x = solveWeightFunction(value, mPowerRegularizationNorm, beta, value);

    mWeightLut[i] = x;
  }
}

}  // namespace Deblurring
}  // namespace Test
