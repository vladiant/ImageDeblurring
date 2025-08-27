/*
 * TvIterationRegularizer.h
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_TVITERATIONREGULARIZER_H_
#define ITERATIVEDEBLURRER_TVITERATIONREGULARIZER_H_

#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class TvIterationRegularizer : public Deblurring::IterationRegularizer {
 public:
  TvIterationRegularizer(int imageWidth, int imageHeight);

  TvIterationRegularizer(int imageWidth, int imageHeight,
                         void* pExternalMemory);

  virtual ~TvIterationRegularizer();

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

#endif /* ITERATIVEDEBLURRER_TVITERATIONREGULARIZER_H_ */
