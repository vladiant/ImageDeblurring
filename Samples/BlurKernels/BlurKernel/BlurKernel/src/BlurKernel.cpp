#include "BlurKernel.hpp"

#include <cmath>

namespace Test {
namespace Deblurring {

// Constructors
BlurKernel::BlurKernel()
    : centerX(0),
      centerY(0),
      isInitialized(false),
      lastResult(BLUR_KERNEL_EMPTY) {}

BlurKernel::BlurKernel(uint16_t kernelWidth, uint16_t kernelHeight) {
  lastResult = initBlurKernel((PSF*)this, kernelWidth, kernelHeight);
  if (lastResult == BLUR_KERNEL_OK) {
    isInitialized = true;
  }
}

BlurKernel::BlurKernel(const PSF& inputPSF) {
  lastResult = cloneBlurKernel((PSF*)this, (PSF*)&inputPSF);
  if (lastResult == BLUR_KERNEL_OK) {
    isInitialized = true;
  }
}

BlurKernel::BlurKernel(const BlurKernel& inputBlurKernel) {
  lastResult = cloneBlurKernel((PSF*)this, (PSF*)&inputBlurKernel);
  if (lastResult == BLUR_KERNEL_OK) {
    this->centerX = inputBlurKernel.centerX;
    this->centerY = inputBlurKernel.centerY;
    isInitialized = true;
  }
}

// Destructor
BlurKernel::~BlurKernel() { destroy(); }

// Manipulation functions
void BlurKernel::clear() {
  if (isInitialized) {
    clearBlurKernel((PSF*)this);
  }
}

void BlurKernel::destroy() {
  if (isInitialized) {
    releaseBlurKernel((PSF*)this);

  } else {
    this->height = 0;
    this->width = 0;
    this->buffer = nullptr;
    this->centerX = 0;
    this->centerY = 0;
  }

  isInitialized = false;
  lastResult = BLUR_KERNEL_EMPTY;
}

BlurKernel& BlurKernel::operator=(const BlurKernel& inputBlurKernel) {
  if (this != &inputBlurKernel) {
    destroy();

    lastResult = cloneBlurKernel((PSF*)this, (PSF*)&inputBlurKernel);
    if (lastResult == BLUR_KERNEL_OK) {
      isInitialized = true;
    }
  }

  return *this;
}

bool BlurKernel::init(uint16_t kernelWidth, uint16_t kernelHeight) {
  bool retVal = false;

  lastResult = initBlurKernel((PSF*)this, kernelWidth, kernelHeight);

  if (lastResult == BLUR_KERNEL_OK) {
    isInitialized = true;
    retVal = true;
  }

  return retVal;
}

uint16_t BlurKernel::getWidth() {
  if (isInitialized) {
    return this->width;

  } else {
    return 0;
  }
}

uint16_t BlurKernel::getHeight() {
  if (isInitialized) {
    return this->height;

  } else {
    return 0;
  }
}

float* BlurKernel::getData() {
  if (isInitialized) {
    return this->buffer;

  } else {
    return nullptr;
  }
}

BlurKernelResult BlurKernel::getErrorCode() { return lastResult; }

bool BlurKernel::isEmpty() { return !isInitialized; }

bool BlurKernel::normalizeKernel() {
  bool retVal = false;

  if (isInitialized) {
    lastResult = normalizeBlurKernel((PSF*)this);

    if (lastResult != BLUR_KERNEL_OK) {
      retVal = true;

    } else {
      retVal = false;
    }
  }

  return retVal;
};

void BlurKernel::regularizeKernel() {
  if (isInitialized) {
    regularizeBlurKernel((PSF*)this);
  }
}

// Virtual functions
void BlurKernel::setKernelSize() {
  if (isInitialized) {
    float centerPointValue = calcKernelPoint(centerX, centerY);
    float eps = BLUR_KERNEL_TOL / centerPointValue;

    uint16_t testRadiusX = 0.0f;
    uint16_t testRadiusY = 0.0f;

    bool isRowLargeValued;
    bool isColLargeValued;

    do {
      do {
        isRowLargeValued = false;

        for (int col = -1.0 * testRadiusX; col <= testRadiusX; ++col) {
          if ((calcKernelPoint(col, testRadiusY) > eps) ||
              (calcKernelPoint(col, -1.0 * testRadiusY) > eps)) {
            isRowLargeValued = true;
          }
        }

        if (isRowLargeValued) testRadiusY++;

      } while (isRowLargeValued);

      do {
        isColLargeValued = false;

        for (int row = -1.0 * testRadiusY; row <= testRadiusY; ++row) {
          if ((calcKernelPoint(testRadiusX, row) > eps) ||
              (calcKernelPoint(-1.0 * testRadiusX, row) > eps)) {
            isColLargeValued = true;
          }
        }

        if (isColLargeValued) testRadiusX++;

      } while (isColLargeValued);

    } while (isColLargeValued || isRowLargeValued);

    init(2 * testRadiusX + 1, 2 * testRadiusY + 1);
    centerX = testRadiusX;
    centerY = testRadiusY;
  }
}

void BlurKernel::fillKernel() {
  if (isInitialized) {
    float* pData = this->buffer;
    int row, col;
    for (row = 0; row < this->height; ++row) {
      for (col = 0; col < this->width; ++col) {
        float relRow = row - centerY;
        float relCol = col - centerX;

        float pointBL = calcKernelPoint(relCol - 0.5, relRow - 0.5);
        float pointUL = calcKernelPoint(relCol - 0.5, relRow + 0.5);
        float pointBR = calcKernelPoint(relCol + 0.5, relRow - 0.5);
        float pointUR = calcKernelPoint(relCol + 0.5, relRow + 0.5);

        *pData = 0.25 * (pointBL + pointUL + pointBR + pointUR);
        pData++;
      }
    }
  }
};

float BlurKernel::calcKernelPoint(float positionX, float positionY) {
  // TEST!!!
  float s;

  /*
  float r = 10;
  if ((positionX*positionX+positionY * positionY)<=(r*r))
  {
      s=1.0;

  } else {

      s=0.0;
  }
  */

  float phi = -1.0 * M_PI / 2;
  float a1 = 4;
  float a2 = 2;
  float currentRadius = ((positionX * cos(phi) + positionY * sin(phi)) *
                         (positionX * cos(phi) + positionY * sin(phi))) /
                            (a1 * a1) +
                        ((positionY * cos(phi) - positionX * sin(phi)) *
                         (positionY * cos(phi) - positionX * sin(phi))) /
                            (a2 * a2);
  ;

  if ((currentRadius) <= (1.0)) {
    s = 1.0;

  } else {
    s = 0.0;
  }

  return (s);
};

}  // namespace Deblurring
}  // namespace Test