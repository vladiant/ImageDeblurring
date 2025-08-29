/*
 * CgTmDeblurrer.h
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_CGTMDEBLURRER_H_
#define ITERATIVEDEBLURRER_CGTMDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class CgTmDeblurrer : public IterativeDeblurrer {
 public:
  CgTmDeblurrer(int imageWidth, int imageHeight);

  CgTmDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~CgTmDeblurrer();

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
  CgTmDeblurrer(const CgTmDeblurrer& other);
  CgTmDeblurrer& operator=(const CgTmDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mResidualImage;
  point_value_t* mPreconditionedImage;
  point_value_t* mBlurredPreconditionedImage;
  point_value_t* mDifferenceResidualImage;

  point_value_t* mRegularizationImage;
  point_value_t* mRegularizationImageTransposed;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_CGTMDEBLURRER_H_ */
