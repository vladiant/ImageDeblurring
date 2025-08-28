/*
 * InverseFftDeblurrer.h
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_INVERSEFFTDEBLURRER_H_
#define FFTDEBLURRER_INVERSEFFTDEBLURRER_H_

#include "FftDeblurrer.h"

namespace Test {
namespace Deblurring {

class InverseFftDeblurrer : public FftDeblurrer {
 public:
  InverseFftDeblurrer(int imageWidth, int imageHeight);

  InverseFftDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~InverseFftDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel& currentBlurKernel,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const point_value_t* kernelData, int kernelWidth,
                          int kernelHeight, int kernelPpln,
                          uint8_t* outputImageData, int outputImagePpln);

 protected:
  virtual void init(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual void deinit();

 private:
  // Disable assignment & copy
  InverseFftDeblurrer(const InverseFftDeblurrer& other);
  InverseFftDeblurrer& operator=(const InverseFftDeblurrer& other);

  point_value_t* mWorkImage;
  point_value_t* mKernelImage;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_INVERSEFFTDEBLURRER_H_ */
