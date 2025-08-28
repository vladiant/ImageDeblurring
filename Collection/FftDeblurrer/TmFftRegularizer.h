/*
 * TmFftRegularizer.h
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_TMFFTREGULARIZER_H_
#define FFTDEBLURRER_TMFFTREGULARIZER_H_

#include "FftRegularizer.h"

namespace Test {
namespace Deblurring {

class TmFftRegularizer : public FftRegularizer {
 public:
  TmFftRegularizer(int imageWidth, int imageHeight);

  TmFftRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~TmFftRegularizer();

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

 private:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_TMFFTREGULARIZER_H_ */
