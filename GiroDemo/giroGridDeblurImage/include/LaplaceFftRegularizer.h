/*
 * LaplaceFftRegularizer.h
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_LAPLACEFFTREGULARIZER_H_
#define FFTDEBLURRER_LAPLACEFFTREGULARIZER_H_

#include "FftRegularizer.h"

namespace Test {
namespace Deblurring {

class LaplaceFftRegularizer : public FftRegularizer {
 public:
  LaplaceFftRegularizer(int imageWidth, int imageHeight);

  LaplaceFftRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~LaplaceFftRegularizer();

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

#endif /* FFTDEBLURRER_LAPLACEFFTREGULARIZER_H_ */
