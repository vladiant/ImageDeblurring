/*
 * IterativeDeblurrer.h
 *
 *  Created on: 17.01.2015
 *      Author: vladiant
 */

#ifndef ITERATIVEDEBLURRER_H_
#define ITERATIVEDEBLURRER_H_

#include "ImageDeblurrer.h"
#include "IterationRegularizer.h"

namespace Test {
namespace Deblurring {

class IterativeDeblurrer : public ImageDeblurrer {
 public:
  virtual ~IterativeDeblurrer();

  enum {
    CONVERGENCE_OK,
    NOT_INITIALIZED,
    MAX_ITERATIONS_REACHED,
    ITERATION_FAILED
  };

  static size_t getMemorySize(int imageWidth, int imageHeight);

  typedef void (Test::Deblurring::IterativeDeblurrer::*BorderImposer)(
      point_coord_t&, point_coord_t&, point_value_t&);

  static constexpr float MAX_ITERATIONS = 1000;
  static constexpr float MINIMAL_NORM = 2e-6;

  int getBorderType() const { return mBorderType; }

  void setBorderType(int borderType, point_value_t borderValue = 0) {
    mBorderType = borderType;
    if (BORDER_CONSTANT == borderValue) {
      mBorderValue = borderValue;
    } else {
      mBorderValue = 0;
    }
  }

  point_value_t getBorderValue() const { return mBorderValue; }

  float getMaxIterations() const { return mMaxIterations; }

  void setMaxIterations(float maxIterations) { mMaxIterations = maxIterations; }

  float getMinimalNorm() const { return mMinimalNorm; }

  void setMinimalNorm(float minimalNorm) { mMinimalNorm = minimalNorm; }

  float getCurrentIteration() const { return mCurrentIteration; }

  float getCurrentNorm() const { return mCurrentNorm; }

  int getProcessStatus() const { return mProcessStatus; }

  const IterationRegularizer* getRegularizer() const { return mRegularizer; }

  void setRegularizer(IterationRegularizer* regularizer) {
    mRegularizer = regularizer;
  }

 protected:
  IterativeDeblurrer(int imageWidth, int imageHeight);

  IterativeDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  int mMaxIterations;
  float mMinimalNorm;

  int mCurrentIteration;
  float mCurrentNorm;

  int mProcessStatus;

  std::vector<point_coord_t> mKernelPointsX;
  std::vector<point_coord_t> mKernelPointsY;
  std::vector<point_value_t> mKernelPointsValues;

  BorderImposer mBorderImposer[BORDER_TYPES];

  IterationRegularizer* mRegularizer;

  virtual void doIterations(const point_value_t* kernelData, int kernelWidth,
                            int kernelHeight, int kernelPpln) = 0;

  virtual void doIterations(const SparseBlurKernel& currentBlurKernel) = 0;

  virtual void prepareIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln) = 0;

  virtual void prepareIterations(const SparseBlurKernel& currentBlurKernel) = 0;

  virtual void doRegularization() = 0;

  void blur(const point_value_t* inputImageData,
            const point_value_t* kernelData, int kernelWidth, int kernelHeight,
            int kernelPpln, point_value_t* outputImageData,
            bool transposeKernel = false);

  void blur(const point_value_t* inputImageData,
            const SparseBlurKernel& currentBlurKernel,
            point_value_t* outputImageData, bool transposeKernel = false);

  void imposeContinuousBorder(point_coord_t& coordX, point_coord_t& coordY,
                              point_value_t& kernelValue);

  void imposeConstantBorder(point_coord_t& coordX, point_coord_t& coordY,
                            point_value_t& kernelValue);

  void imposePeriodicBorder(point_coord_t& coordX, point_coord_t& coordY,
                            point_value_t& kernelValue);

  void imposeMirrorBorder(point_coord_t& coordX, point_coord_t& coordY,
                          point_value_t& kernelValue);

  void subtractImages(const point_value_t* firstImageData,
                      const point_value_t* secondImageData,
                      point_value_t* outputImageData);

  void multiplyImages(const point_value_t* firstImageData,
                      const point_value_t* secondImageData,
                      point_value_t* outputImageData);

  bool divideImages(const point_value_t* firstImageData,
                    const point_value_t* secondImageData,
                    point_value_t* outputImageData);

  void addWeightedImages(const point_value_t* firstImageData,
                         double firstImageWeight,
                         const point_value_t* secondImageData,
                         double secondImageWeight, double offset,
                         point_value_t* outputImageData);

  double calculateDotProductOfImages(const point_value_t* firstImageData,
                                     const point_value_t* secondImageData);

  double calculateL2Norm(const point_value_t* imageData);

  double calculateL2NormOfDifference(const point_value_t* firstImageData,
                                     const point_value_t* secondImageData);

  void scaleImage(const point_value_t* inputImageData, double imageScale,
                  double offset, point_value_t* outputImageData);

 private:
  // Disable assignment & copy
  IterativeDeblurrer(const IterativeDeblurrer& other);
  IterativeDeblurrer& operator=(const IterativeDeblurrer& other);

  void setBorderImposers();
  void clear();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_H_ */
