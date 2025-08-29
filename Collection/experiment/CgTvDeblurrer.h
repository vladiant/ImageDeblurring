/*
 * CgTvDeblurrer.h
 *
 *  Created on: Jan 26, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_CGTVDEBLURRER_H_
#define ITERATIVEDEBLURRER_CGTVDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class CgTvDeblurrer : public IterativeDeblurrer {
 public:
  CgTvDeblurrer(int imageWidth, int imageHeight);

  CgTvDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~CgTvDeblurrer();

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

  void calculateIrlsWeights(const point_value_t* inputImageData,
                            point_value_t* outputImageData);

  void calculateIrlsWeightsX(const point_value_t* inputImageData,
                             point_value_t* outputImageData);

  void calculateIrlsWeightsY(const point_value_t* inputImageData,
                             point_value_t* outputImageData);

 private:
  // Disable assignment & copy
  CgTvDeblurrer(const CgTvDeblurrer& other);
  CgTvDeblurrer& operator=(const CgTvDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mResidualImage;
  point_value_t* mPreconditionedImage;
  point_value_t* mBlurredPreconditionedImage;
  point_value_t* mDifferenceResidualImage;

  point_value_t* mRegularizationImage;
  point_value_t* mRegularizationImageTransposed;

  point_value_t* mIrlsWeights;
  point_value_t* mIrlsWeightsX;
  point_value_t* mIrlsWeightsY;
  point_value_t* mOldDeblurredImage;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_CGTVDEBLURRER_H_ */
