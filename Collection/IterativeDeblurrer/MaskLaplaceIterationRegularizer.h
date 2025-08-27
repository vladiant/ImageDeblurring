/*
 * MaskMaskLaplaceIterationRegularizer.h
 *
 *  Created on: 07.02.2015
 *      Author: vladiant
 */

#ifndef MASKMaskLaplaceIterationRegularizer_H_
#define MASKMaskLaplaceIterationRegularizer_H_

#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class MaskLaplaceIterationRegularizer
    : public Deblurring::IterationRegularizer {
 public:
  static constexpr float STANDARD_DEVIATION_WEIGHT = 10.0;

  MaskLaplaceIterationRegularizer(int imageWidth, int imageHeight);

  MaskLaplaceIterationRegularizer(int imageWidth, int imageHeight,
                                  void* pExternalMemory);

  virtual ~MaskLaplaceIterationRegularizer();

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

  virtual void prepareIrls(const point_value_t* inputImageData);

  virtual void calculateIrlsWeights(const point_value_t* inputImageData);

  virtual void calculateIrlsWeightsX(const point_value_t* inputImageData);

  virtual void calculateIrlsWeightsY(const point_value_t* inputImageData);

  float getStandardDeviationWeight() const { return mStandardDeviationWeight; }

  void setStandardDeviationWeight(float standardDeviationWeight) {
    mStandardDeviationWeight = standardDeviationWeight;
  }

 protected:
  float mStandardDeviationWeight;

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* MASKMaskLaplaceIterationRegularizer_H_ */
