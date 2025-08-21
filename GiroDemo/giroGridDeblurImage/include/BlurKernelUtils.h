/*
 * BlurKernelUtils.h
 *
 *  Created on: 01.11.2014
 *      Author: vladiant
 */

#ifndef BLURKERNELUTILS_H_
#define BLURKERNELUTILS_H_

#include "GyroBlurKernelBuilder.h"

namespace Test {
namespace Deblurring {

void calculateKernelGridProperties(GyroBlurKernelBuilder& kernelBuilder,
                                   int& gridStepX, int& gridStepY,
                                   float& maxXCoordsSpan,
                                   float& maxYCoordsSpan);

void extractSparseKernelToImage(SparseBlurKernel& blurKernel,
                                point_value_t* imageData, int imageWidth,
                                int imageHeight, int imagePpln);

void optimizePatchSize(int& size);

template <typename InputDataT, typename OutputDataT>
void getPatchFromImage(int patchOriginX, int patchOriginY,
                       const InputDataT* imageData, int imageWidth,
                       int imageHeight, int imagePpln, OutputDataT* patchData,
                       int patchWidth, int patchHeight, int patchPpln) {
  int startCropX;

  int startPutX;
  int endPutX;

  if (patchOriginX < 0) {
    startCropX = 0;
    startPutX = -patchOriginX;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endPutX = patchWidth;
    } else {
      endPutX = imageWidth - patchOriginX;
    }

  } else if (patchOriginX < imageWidth) {
    startCropX = patchOriginX;
    startPutX = 0;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endPutX = patchWidth;
    } else {
      endPutX = imageWidth - patchOriginX;
    }
  } else {
    return;
  }

  int startCropY;
  int endCropY;

  int startPutY;

  if (patchOriginY < 0) {
    startCropY = 0;
    startPutY = -patchOriginY;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }

  } else if (patchOriginY < imageHeight) {
    startCropY = patchOriginY;
    startPutY = 0;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }
  } else {
    return;
  }

  for (int rowInput = startCropY, rowOutput = startPutY; rowInput < endCropY;
       rowInput++, rowOutput++) {
    InputDataT* pInputData =
        (InputDataT*)imageData + rowInput * imagePpln + startCropX;
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + startPutX;
    for (int col = 0; col < endPutX - startPutX; col++) {
      *pOutputData++ = *pInputData++;
    }
  }

  // Top left corner.
  OutputDataT topLeftValue =
      *((OutputDataT*)patchData + startPutY * patchPpln + startPutX);
  for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = topLeftValue;
    }
  }

  // Top side.
  for (int col = startPutX; col < endPutX; col++) {
    OutputDataT topSideValue =
        *((OutputDataT*)patchData + startPutY * patchPpln + col);
    for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
      OutputDataT* pOutputData =
          (OutputDataT*)patchData + rowOutput * patchPpln + col;
      *pOutputData++ = topSideValue;
    }
  }

  // Top right corner.
  OutputDataT topRightValue =
      *((OutputDataT*)patchData + startPutY * patchPpln + endPutX - 1);
  for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = topRightValue;
    }
  }

  // Right side.
  for (int rowOutput = startPutY; rowOutput < startPutY + endCropY - startCropY;
       rowOutput++) {
    OutputDataT rightSideValue =
        *((OutputDataT*)patchData + rowOutput * patchPpln + endPutX - 1);
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = rightSideValue;
    }
  }

  // Bottom right corner.
  OutputDataT bottomRightValue =
      *((OutputDataT*)patchData +
        (startPutY + endCropY - startCropY - 1) * patchPpln + endPutX - 1);
  for (int rowOutput = startPutY + endCropY - startCropY;
       rowOutput < patchHeight; rowOutput++) {
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = bottomRightValue;
    }
  }

  // Bottom side.
  for (int col = startPutX; col < endPutX; col++) {
    OutputDataT bottomSideValue =
        *((OutputDataT*)patchData +
          (startPutY + endCropY - startCropY - 1) * patchPpln + col);
    for (int rowOutput = startPutY + endCropY - startCropY;
         rowOutput < patchHeight; rowOutput++) {
      OutputDataT* pOutputData =
          (OutputDataT*)patchData + rowOutput * patchPpln + col;
      *pOutputData++ = bottomSideValue;
    }
  }

  // Bottom left corner.
  OutputDataT bottomLeftValue =
      *((OutputDataT*)patchData +
        (startPutY + endCropY - startCropY - 1) * patchPpln + startPutX);
  for (int rowOutput = startPutY + endCropY - startCropY;
       rowOutput < patchHeight; rowOutput++) {
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = bottomLeftValue;
    }
  }

  // Left side.
  for (int rowOutput = startPutY; rowOutput < startPutY + endCropY - startCropY;
       rowOutput++) {
    OutputDataT leftSideValue =
        *((OutputDataT*)patchData + rowOutput * patchPpln + startPutX);
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = leftSideValue;
    }
  }

  return;
}  // void getPatchFromImage( ...

template <typename InputDataT, typename OutputDataT>
void getPatchFromMultichannelImage(int patchOriginX, int patchOriginY,
                                   const InputDataT* imageData, int imageWidth,
                                   int imageHeight, int imagePpln,
                                   int imageChannels, OutputDataT* patchData,
                                   int patchWidth, int patchHeight,
                                   int patchPpln, int channelToGet) {
  int startCropX;

  int startPutX;
  int endPutX;

  if (patchOriginX < 0) {
    startCropX = 0;
    startPutX = -patchOriginX;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endPutX = patchWidth;
    } else {
      endPutX = imageWidth - patchOriginX;
    }

  } else if (patchOriginX < imageWidth) {
    startCropX = patchOriginX;
    startPutX = 0;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endPutX = patchWidth;
    } else {
      endPutX = imageWidth - patchOriginX;
    }
  } else {
    return;
  }

  int startCropY;
  int endCropY;

  int startPutY;

  if (patchOriginY < 0) {
    startCropY = 0;
    startPutY = -patchOriginY;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }

  } else if (patchOriginY < imageHeight) {
    startCropY = patchOriginY;
    startPutY = 0;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }
  } else {
    return;
  }

  for (int rowInput = startCropY, rowOutput = startPutY; rowInput < endCropY;
       rowInput++, rowOutput++) {
    InputDataT* pInputData =
        (InputDataT*)imageData +
        (rowInput * imagePpln + startCropX) * imageChannels + channelToGet - 1;
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + startPutX;
    for (int col = 0; col < endPutX - startPutX; col++) {
      *pOutputData++ = *pInputData;
      pInputData += imageChannels;
    }
  }

  // Top left corner.
  OutputDataT topLeftValue =
      *((OutputDataT*)patchData + startPutY * patchPpln + startPutX);
  for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = topLeftValue;
    }
  }

  // Top side.
  for (int col = startPutX; col < endPutX; col++) {
    OutputDataT topSideValue =
        *((OutputDataT*)patchData + startPutY * patchPpln + col);
    for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
      OutputDataT* pOutputData =
          (OutputDataT*)patchData + rowOutput * patchPpln + col;
      *pOutputData++ = topSideValue;
    }
  }

  // Top right corner.
  OutputDataT topRightValue =
      *((OutputDataT*)patchData + startPutY * patchPpln + endPutX - 1);
  for (int rowOutput = 0; rowOutput < startPutY; rowOutput++) {
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = topRightValue;
    }
  }

  // Right side.
  for (int rowOutput = startPutY; rowOutput < startPutY + endCropY - startCropY;
       rowOutput++) {
    OutputDataT rightSideValue =
        *((OutputDataT*)patchData + rowOutput * patchPpln + endPutX - 1);
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = rightSideValue;
    }
  }

  // Bottom right corner.
  OutputDataT bottomRightValue =
      *((OutputDataT*)patchData +
        (startPutY + endCropY - startCropY - 1) * patchPpln + endPutX - 1);
  for (int rowOutput = startPutY + endCropY - startCropY;
       rowOutput < patchHeight; rowOutput++) {
    OutputDataT* pOutputData =
        (OutputDataT*)patchData + rowOutput * patchPpln + endPutX;
    for (int col = endPutX; col < patchWidth; col++) {
      *pOutputData++ = bottomRightValue;
    }
  }

  // Bottom side.
  for (int col = startPutX; col < endPutX; col++) {
    OutputDataT bottomSideValue =
        *((OutputDataT*)patchData +
          (startPutY + endCropY - startCropY - 1) * patchPpln + col);
    for (int rowOutput = startPutY + endCropY - startCropY;
         rowOutput < patchHeight; rowOutput++) {
      OutputDataT* pOutputData =
          (OutputDataT*)patchData + rowOutput * patchPpln + col;
      *pOutputData++ = bottomSideValue;
    }
  }

  // Bottom left corner.
  OutputDataT bottomLeftValue =
      *((OutputDataT*)patchData +
        (startPutY + endCropY - startCropY - 1) * patchPpln + startPutX);
  for (int rowOutput = startPutY + endCropY - startCropY;
       rowOutput < patchHeight; rowOutput++) {
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = bottomLeftValue;
    }
  }

  // Left side.
  for (int rowOutput = startPutY; rowOutput < startPutY + endCropY - startCropY;
       rowOutput++) {
    OutputDataT leftSideValue =
        *((OutputDataT*)patchData + rowOutput * patchPpln + startPutX);
    OutputDataT* pOutputData = (OutputDataT*)patchData + rowOutput * patchPpln;
    for (int col = 0; col < startPutX; col++) {
      *pOutputData++ = leftSideValue;
    }
  }

  return;
}  // void getPatchFromImage( ...

template <typename InputDataT, typename OutputDataT>
void putPatchInImage(int patchOriginX, int patchOriginY,
                     const InputDataT* patchData, int patchWidth,
                     int patchHeight, int patchPpln, OutputDataT* imageData,
                     int imageWidth, int imageHeight, int imagePpln) {
  int startCropX;

  int startGetX;
  int endGetX;

  if (patchOriginX < 0) {
    startCropX = 0;
    startGetX = -patchOriginX;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endGetX = patchWidth;
    } else {
      endGetX = imageWidth - patchOriginX;
    }

  } else if (patchOriginX < imageWidth) {
    startCropX = patchOriginX;
    startGetX = 0;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endGetX = patchWidth;
    } else {
      endGetX = imageWidth - patchOriginX;
    }
  } else {
    return;
  }

  int startCropY;
  int endCropY;

  int startGetY;

  if (patchOriginY < 0) {
    startCropY = 0;
    startGetY = -patchOriginY;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }

  } else if (patchOriginY < imageHeight) {
    startCropY = patchOriginY;
    startGetY = 0;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }
  } else {
    return;
  }

  for (int rowOutput = startCropY, rowInput = startGetY; rowOutput < endCropY;
       rowOutput++, rowInput++) {
    InputDataT* pInputData =
        (InputDataT*)patchData + rowInput * patchPpln + startGetX;
    OutputDataT* pOutputData =
        (OutputDataT*)imageData + rowOutput * imagePpln + startCropX;
    for (int col = 0; col < endGetX - startGetX; col++) {
      *pOutputData++ =
          *pInputData++;  // uint8_t->uint8_t case!
                          // FIXME: For the case of uint8_t output
                          //			float value = *pInputData++;
                          //			if (value < 0){
                          //				value = 0;
                          //			} else if (value > 255){
                          //				value = 255;
                          //			}
                          //			*pOutputData++ = value;
    }
  }

  return;
}  // void putPatchInImage( ...

template <typename InputDataT, typename OutputDataT>
void putPatchInMultichannelImage(int patchOriginX, int patchOriginY,
                                 const InputDataT* patchData, int patchWidth,
                                 int patchHeight, int patchPpln,
                                 int channelToPut, OutputDataT* imageData,
                                 int imageWidth, int imageHeight, int imagePpln,
                                 int imageChannels) {
  int startCropX;

  int startGetX;
  int endGetX;

  if (patchOriginX < 0) {
    startCropX = 0;
    startGetX = -patchOriginX;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endGetX = patchWidth;
    } else {
      endGetX = imageWidth - patchOriginX;
    }

  } else if (patchOriginX < imageWidth) {
    startCropX = patchOriginX;
    startGetX = 0;

    if (patchOriginX + patchWidth < 0) {
      return;
    } else if (patchOriginX + patchWidth < imageWidth) {
      endGetX = patchWidth;
    } else {
      endGetX = imageWidth - patchOriginX;
    }
  } else {
    return;
  }

  int startCropY;
  int endCropY;

  int startGetY;

  if (patchOriginY < 0) {
    startCropY = 0;
    startGetY = -patchOriginY;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }

  } else if (patchOriginY < imageHeight) {
    startCropY = patchOriginY;
    startGetY = 0;

    if (patchOriginY + patchHeight < 0) {
      return;
    } else if (patchOriginY + patchHeight < imageHeight) {
      endCropY = patchHeight + patchOriginY;
    } else {
      endCropY = imageHeight;
    }
  } else {
    return;
  }

  for (int rowOutput = startCropY, rowInput = startGetY; rowOutput < endCropY;
       rowOutput++, rowInput++) {
    InputDataT* pInputData =
        (InputDataT*)patchData + rowInput * patchPpln + startGetX;
    OutputDataT* pOutputData =
        (OutputDataT*)imageData +
        imageChannels * (rowOutput * imagePpln + startCropX) + channelToPut - 1;
    for (int col = 0; col < endGetX - startGetX; col++) {
      *pOutputData = *pInputData++;  // uint8_t->uint8_t case!
      // FIXME: For the case of uint8_t output
      //			float value = *pInputData++;
      //			if (value < 0){
      //				value = 0;
      //			} else if (value > 255){
      //				value = 255;
      //			}
      //			*pOutputData = value;
      pOutputData += imageChannels;
    }
  }

  return;
}  // void putPatchInMultichannelImage( ...

}  // namespace Deblurring
}  // namespace Test

#endif /* BLURKERNELUTILS_H_ */
