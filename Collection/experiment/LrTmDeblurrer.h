/*
 * LrTmDeblurrer.h
 *
 *  Created on: Jan 26, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_LRTMDEBLURRER_H_
#define ITERATIVEDEBLURRER_LRTMDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class LrTmDeblurrer : public IterativeDeblurrer {
 public:
  LrTmDeblurrer(int imageWidth, int imageHeight);

  LrTmDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~LrTmDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

  float getRegularizationWeight() const { return mRegularizationWeight; }

  void setRegularizationWeight(float regularizationWeight) {
    mRegularizationWeight = regularizationWeight;
  }

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  virtual void doIterations(const point_value_t* kernelData, int kernelWidth,
                            int kernelHeight, int kernelPpln);

  virtual void doIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void prepareIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln);

  virtual void prepareIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void doRegularization();

  void calculateRegularization(const point_value_t* inputImageData,
                               point_value_t* outputImageData,
                               bool transposeKernel);

  void calculateRegularizationX(const point_value_t* inputImageData,
                                point_value_t* outputImageData,
                                bool transposeKernel);

  void calculateRegularizationY(const point_value_t* inputImageData,
                                point_value_t* outputImageData,
                                bool transposeKernel);

 private:
  // Disable assignment & copy
  LrTmDeblurrer(const LrTmDeblurrer& other);
  LrTmDeblurrer& operator=(const LrTmDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mReblurredImage;
  point_value_t* mBlurredImage;
  point_value_t* mWeightImage;
  point_value_t* mBlurredWeightImage;

  point_value_t* mRegularizationImage;
  point_value_t* mRegularizationImageTransposed;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_LRTMDEBLURRER_H_ */
