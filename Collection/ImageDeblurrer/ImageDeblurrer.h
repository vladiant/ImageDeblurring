/*
 * ImageDeblurrer.h
 *
 *  Created on: 30.11.2014
 *      Author: vladiant
 */

#ifndef IMAGEDEBLURRER_H_
#define IMAGEDEBLURRER_H_

#include <stddef.h>
#include <stdint.h>

#include "SparseBlurKernel.h"

namespace Test {
namespace Deblurring {

class ImageDeblurrer {
 public:
  virtual ~ImageDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel& currentBlurKernel,
                          uint8_t* outputImageData, int outputImagePpln) = 0;

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const point_value_t* kernelData, int kernelWidth,
                          int kernelHeight, int kernelPpln,
                          uint8_t* outputImageData, int outputImagePpln) = 0;

  static const int BORDER_TYPES = 4;
  static const int BORDER_CONTINUOUS = 0;
  static const int BORDER_CONSTANT = 1;
  static const int BORDER_PERIODIC = 2;
  static const int BORDER_MIRROR = 3;

 protected:
  bool isExternalMemoryUsed;
  int mImageWidth;
  int mImageHeight;
  int mBorderType;
  point_value_t mBorderValue;

  ImageDeblurrer(int imageWidth, int imageHeight);

  ImageDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual void init(int imageWidth, int imageHeight, void* pExternalMemory) = 0;

  virtual void deinit() = 0;

  static void convertFromInput(const uint8_t* inputImageData, int imageWidth,
                               int imageHeight, int inputImagePpln,
                               point_value_t* inputBuffer);

  static void convertToOutput(const point_value_t* outputBuffer, int imageWidth,
                              int imageHeight, int inputImagePpln,
                              uint8_t* outputImageData);

 private:
  // Disable assignment & copy
  ImageDeblurrer(const ImageDeblurrer& other);
  ImageDeblurrer& operator=(const ImageDeblurrer& other);

  void release();
};

}  // namespace Deblurring
}  // namespace Test

#endif /* IMAGEDEBLURRER_H_ */
