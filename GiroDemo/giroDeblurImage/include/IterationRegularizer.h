/*
 * IterationRegularizer.h
 *
 *  Created on: Jan 26, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_ITERATIONREGULARIZER_H_
#define ITERATIVEDEBLURRER_ITERATIONREGULARIZER_H_

#include <stddef.h>

#include "SparseBlurKernel.h"

namespace Test {
namespace Deblurring {

class IterationRegularizer {
 public:
  virtual ~IterationRegularizer();

  static constexpr float MINIMAL_IRLS_NORM = 5e-6;
  static constexpr float MAX_IRLS_ITERATIONS = 1000;

  static size_t getMemorySize(int imageWidth, int imageHeight);

  float getRegularizationWeight() const { return mRegularizationWeight; }

  void setRegularizationWeight(float regularizationWeight) {
    mRegularizationWeight = regularizationWeight;
  }

  virtual void calculateRegularization(const point_value_t* inputImageData,
                                       point_value_t* outputImageData,
                                       bool transposeKernel) = 0;

  virtual void calculateRegularizationX(const point_value_t* inputImageData,
                                        point_value_t* outputImageData,
                                        bool transposeKernel) = 0;

  virtual void calculateRegularizationY(const point_value_t* inputImageData,
                                        point_value_t* outputImageData,
                                        bool transposeKernel) = 0;

  bool isIrlsUsed() {
    return ((mIrlsWeights != NULL) || (mIrlsWeightsX != NULL) ||
            (mIrlsWeightsY != NULL)) &&
           (mOldDeblurredImage != NULL);
  }

  virtual void prepareIrls(const point_value_t* inputImageData);

  void setOldDeblurredImage(const point_value_t* inputImageData);

  point_value_t* getOldDeblurredImage() const { return mOldDeblurredImage; }

  virtual void calculateIrlsWeights(const point_value_t* inputImageData) = 0;

  virtual void calculateIrlsWeightsX(const point_value_t* inputImageData) = 0;

  virtual void calculateIrlsWeightsY(const point_value_t* inputImageData) = 0;

  float getMinimalNormIrls() const { return mMinimalNormIrls; }

  void setMinimalNormIrls(float minimalNormIrls) {
    mMinimalNormIrls = minimalNormIrls;
  }

  int getMaxIrlsIterations() const { return mMaxIrlsIterations; }

  void setMaxIrlsIterations(int maxIrlsIterations) {
    mMaxIrlsIterations = maxIrlsIterations;
  }

  point_value_t* getIrlsWeights() const { return mIrlsWeights; }

  point_value_t* getIrlsWeightsX() const { return mIrlsWeightsX; }

  point_value_t* getIrlsWeightsY() const { return mIrlsWeightsY; }

  point_value_t* getRegularizationImage() const { return mRegularizationImage; }

  point_value_t* getRegularizationImageTransposed() const {
    return mRegularizationImageTransposed;
  }

 protected:
  bool isExternalMemoryUsed;
  int mImageWidth;
  int mImageHeight;
  size_t mImageBlockSize;

  point_value_t* mRegularizationImage;
  point_value_t* mRegularizationImageTransposed;

  float mMinimalNormIrls;
  int mMaxIrlsIterations;

  point_value_t* mIrlsWeights;
  point_value_t* mIrlsWeightsX;
  point_value_t* mIrlsWeightsY;
  point_value_t* mOldDeblurredImage;

  float mRegularizationWeight;

  IterationRegularizer(int imageWidth, int imageHeight);

  IterationRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

 private:
  // Disable assignment & copy
  IterationRegularizer(const IterationRegularizer& other);
  IterationRegularizer& operator=(const IterationRegularizer& other);

  void initalize(int imageWidth, int imageHeight, void* pExternalMemory);

  void release();

  void initializeIrlsWeight(point_value_t* irlsWeight);
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_ITERATIONREGULARIZER_H_ */
