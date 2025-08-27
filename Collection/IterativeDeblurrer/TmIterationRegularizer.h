/*
 * TmIterationRegularizer.h
 *
 *  Created on: Jan 27, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_TMITERATIONREGULARIZER_H_
#define ITERATIVEDEBLURRER_TMITERATIONREGULARIZER_H_

#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class TmIterationRegularizer : public Deblurring::IterationRegularizer {
 public:
  TmIterationRegularizer(int imageWidth, int imageHeight);

  TmIterationRegularizer(int imageWidth, int imageHeight,
                         void* pExternalMemory);

  virtual ~TmIterationRegularizer();

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

#endif /* ITERATIVEDEBLURRER_TMITERATIONREGULARIZER_H_ */
