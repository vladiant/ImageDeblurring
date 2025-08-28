/*
 * HLFftRegularizer.h
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_HLFFTREGULARIZER_H_
#define FFTDEBLURRER_HLFFTREGULARIZER_H_

#include "FftRegularizer.h"

namespace Test {
namespace Deblurring {

class HLFftRegularizer : public FftRegularizer {
 public:
  static const int WEIGHT_LUT_SIZE = 4096;

  HLFftRegularizer(int imageWidth, int imageHeight);

  HLFftRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~HLFftRegularizer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void calculateHqpmWeights(const point_value_t* inputFftImageData,
                                    float beta);

  virtual void calculateHqpmWeightsX(const point_value_t* inputFftImageData,
                                     float beta);

  virtual void calculateHqpmWeightsY(const point_value_t* inputFftImageData,
                                     float beta);

  float getPowerRegularizationNorm() const { return mPowerRegularizationNorm; }

  void setPowerRegularizationNorm(float powerRegularizationNorm) {
    mPowerRegularizationNorm = powerRegularizationNorm;
  }

  void prepareWeightLut(float beta);

 protected:
  virtual void setRegularization();

  virtual void setRegularizationX();

  virtual void setRegularizationY();

  virtual void calculateHqpmWeight(const point_value_t* inputImageData,
                                   float beta, point_value_t* outputImageData);

  float WeightCalcAlpha(float v, float beta);

  float mPowerRegularizationNorm;

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  double weightFunction(double w, double alpha, double beta, double v);

  double solveWeightFunction(double startX, double alpha, double beta,
                             double v);

  point_value_t* mWeightLut;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_HLFFTREGULARIZER_H_ */
