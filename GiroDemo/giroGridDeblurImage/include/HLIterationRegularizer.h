/*
 * HlIterationRegularizer.h
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_HLITERATIONREGULARIZER_H_
#define ITERATIVEDEBLURRER_HLITERATIONREGULARIZER_H_

#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class HLIterationRegularizer : public Deblurring::IterationRegularizer {
 public:
  HLIterationRegularizer(int imageWidth, int imageHeight);

  HLIterationRegularizer(int imageWidth, int imageHeight,
                         void* pExternalMemory);

  virtual ~HLIterationRegularizer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void calculateRegularization(const point_value_t* inputImageData,
                                       point_value_t* outputImageData,
                                       bool transposeKernel);

  virtual void calculateRegularizationX(const point_value_t* inputImageData,
                                        point_value_t* outputImageData,
                                        bool transposeKernel);

  virtual void calculateRegularizationY(const point_value_t* inputImageData,
                                        point_value_t* outputImageData,
                                        bool transposeKernel);

  virtual void calculateIrlsWeights(const point_value_t* inputImageData);

  virtual void calculateIrlsWeightsX(const point_value_t* inputImageData);

  virtual void calculateIrlsWeightsY(const point_value_t* inputImageData);

  float getPowerRegularizationNorm() const { return mPowerRegularizationNorm; }

  void setPowerRegularizationNorm(float powerRegularizationNorm) {
    mPowerRegularizationNorm = powerRegularizationNorm;
  }

  float mPowerRegularizationNorm;

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_HLITERATIONREGULARIZER_H_ */
