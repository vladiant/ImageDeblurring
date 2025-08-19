/*
 * LaplaceIterationRegularizer.h
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_LAPLACEITERATIONREGULARIZER_H_
#define ITERATIVEDEBLURRER_LAPLACEITERATIONREGULARIZER_H_

#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class LaplaceIterationRegularizer : public Deblurring::IterationRegularizer {
 public:
  LaplaceIterationRegularizer(int imageWidth, int imageHeight);

  LaplaceIterationRegularizer(int imageWidth, int imageHeight,
                              void* pExternalMemory);

  virtual ~LaplaceIterationRegularizer();

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

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_LAPLACEITERATIONREGULARIZER_H_ */
