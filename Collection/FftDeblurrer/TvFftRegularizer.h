/*
 * TvFftRegularizer.h
 *
 *  Created on: Feb 5, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_TVFFTREGULARIZER_H_
#define FFTDEBLURRER_TVFFTREGULARIZER_H_

#include "FftRegularizer.h"

namespace Test {
namespace Deblurring {

class TvFftRegularizer : public FftRegularizer {
 public:
  TvFftRegularizer(int imageWidth, int imageHeight);

  TvFftRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~TvFftRegularizer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void calculateHqpmWeights(const point_value_t* inputFftImageData,
                                    float beta);

  virtual void calculateHqpmWeightsX(const point_value_t* inputFftImageData,
                                     float beta);

  virtual void calculateHqpmWeightsY(const point_value_t* inputFftImageData,
                                     float beta);

 protected:
  virtual void setRegularization();

  virtual void setRegularizationX();

  virtual void setRegularizationY();

  virtual void calculateHqpmWeight(const point_value_t* inputImageData,
                                   float beta, point_value_t* outputImageData);

  float WeightCalcAlphaTv(float v, float beta);

  float WeightCalcAlphaTvOther(float v, float beta);

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_TVFFTREGULARIZER_H_ */
