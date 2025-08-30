/*
 * OcvFftDeblurrer.h
 *
 *  Created on: 05.01.2015
 *      Author: vladiant
 */

#ifndef OCVFFTDEBLURRER_H_
#define OCVFFTDEBLURRER_H_

#include <opencv2/core/core.hpp>

#include "ImageDeblurrer.h"

namespace Test {
namespace Deblurring {

class OcvFftDeblurrer : public ImageDeblurrer {
 public:
  OcvFftDeblurrer(int imageWidth, int imageHeight);

  OcvFftDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  ~OcvFftDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight, int kernelWidth,
                              int kernelHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

  static constexpr float epsilon = 1e-4;

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  void buildKernelForFft(const point_value_t* kernelData, int kernelWidth,
                         int kernelHeight, int kernelPpln,
                         point_value_t* kernelFftData);

  void invertFftMatrix(cv::Mat& matrixFft, float gamma);

  void invertFftMatrix(cv::Mat& matrixFft);

  void applyDFTWindow(cv::Mat& imageFft);

 private:
  point_value_t* mInputImageData;
  point_value_t* mKernelImageData;
  point_value_t* mOutputImageData;
  point_value_t* mWeightImageData;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* OCVFFTDEBLURRER_H_ */
