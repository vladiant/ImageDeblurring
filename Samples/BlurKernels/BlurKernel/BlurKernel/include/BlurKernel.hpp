#pragma once

#include "PSF.hpp"
extern "C" {
#include "blur_kernel.h"
}

namespace Test {
namespace Deblurring {

class BlurKernel : private PSF {
 public:
  BlurKernel();
  BlurKernel(uint16_t kernelWidth, uint16_t kernelHeight);
  BlurKernel(const PSF& inputPSF);
  BlurKernel(const BlurKernel& inputBlurKernel);
  ~BlurKernel();

  BlurKernel& operator=(const BlurKernel& inputBlurKernel);

  bool init(uint16_t kernelWidth, uint16_t kernelHeight);
  void clear();
  void destroy();

  uint16_t getWidth();
  uint16_t getHeight();
  float* getData();
  BlurKernelResult getErrorCode();
  bool isEmpty();

  bool normalizeKernel();
  void regularizeKernel();

  //        protected:
  virtual void setKernelSize();
  void fillKernel();
  virtual float calcKernelPoint(float positionX, float positionY);

  float centerX;
  float centerY;

 private:
  bool isInitialized;
  BlurKernelResult lastResult;

};  // BlurKernel

}  // namespace Deblurring
}  // namespace Test
