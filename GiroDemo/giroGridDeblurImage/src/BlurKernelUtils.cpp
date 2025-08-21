/*
 * BlurKernelUtils.cpp
 *
 *  Created on: 01.11.2014
 *      Author: vladiant
 */

#include "BlurKernelUtils.h"

#include <string.h>

#include <stdexcept>
#include <vector>

namespace Test {
namespace Deblurring {

const float MAX_KERNEL_DIFFERENCE = 0.4;

void calculateKernelGridProperties(GyroBlurKernelBuilder& kernelBuilder,
                                   int& gridStepX, int& gridStepY,
                                   float& maxXCoordsSpan,
                                   float& maxYCoordsSpan) {
  int divisorStepX = 2;
  int divisorStepY = 2;
  gridStepX = kernelBuilder.getImageWidth() / divisorStepX;
  gridStepY = kernelBuilder.getImageHeight() / divisorStepY;

  float maxXCoordsDiff = std::numeric_limits<float>::min();
  float maxYCoordsDiff = std::numeric_limits<float>::min();

  do {
    // Calculate the maximal difference of the kernels.
    maxXCoordsSpan = std::numeric_limits<float>::min();
    maxYCoordsSpan = std::numeric_limits<float>::min();

    std::vector<std::vector<float> > xCoordsSpan;
    std::vector<std::vector<float> > yCoordsSpan;

    // Draw lines and centers.
    for (int gridCenterY = gridStepY / 2;
         gridCenterY < kernelBuilder.getImageHeight();
         gridCenterY += gridStepY) {
      std::vector<float> xCoordsSpanInRow;
      std::vector<float> yCoordsSpanInRow;

      for (int gridCenterX = gridStepX / 2;
           gridCenterX < kernelBuilder.getImageWidth();
           gridCenterX += gridStepX) {
        // Calculate blur kernel.
        SparseBlurKernel currentBlurKernel;
        kernelBuilder.calculateAtPoint(gridCenterX, gridCenterY,
                                       currentBlurKernel);
        float xCoordMin = kernelBuilder.getKernelXMin();
        float yCoordMin = kernelBuilder.getKernelYMin();
        float xCoordMax = kernelBuilder.getKernelXMax();
        float yCoordMax = kernelBuilder.getKernelYMax();

        float xCoordsSpan = xCoordMax - xCoordMin;
        float yCoordsSpan = yCoordMax - yCoordMin;

        if (xCoordsSpan > maxXCoordsSpan) {
          maxXCoordsSpan = xCoordsSpan;
        }

        if (yCoordsSpan > maxYCoordsSpan) {
          maxYCoordsSpan = yCoordsSpan;
        }

        xCoordsSpanInRow.push_back(xCoordsSpan);
        yCoordsSpanInRow.push_back(yCoordsSpan);
      }

      xCoordsSpan.push_back(xCoordsSpanInRow);
      yCoordsSpan.push_back(yCoordsSpanInRow);

    }  // for (int gridCenterY = ...

    // Calculate the maximal difference of the kernels.
    maxXCoordsDiff = std::numeric_limits<float>::min();
    maxYCoordsDiff = std::numeric_limits<float>::min();
    for (size_t i = 0; i < xCoordsSpan.size(); i++) {
      std::vector<float>& xCoordsSpanInCurrentRow = xCoordsSpan[i];
      std::vector<float>& yCoordsSpanInCurrentRow = yCoordsSpan[i];

      if (i == 0) {
        // First row.
        for (size_t j = 1; j < xCoordsSpanInCurrentRow.size(); j++) {
          float xCoordsSpan =
              xCoordsSpanInCurrentRow[j] - xCoordsSpanInCurrentRow[j - 1];
          if (xCoordsSpan > maxXCoordsDiff) {
            maxXCoordsDiff = xCoordsSpan;
          }

          float yCoordsSpan =
              yCoordsSpanInCurrentRow[j] - yCoordsSpanInCurrentRow[j - 1];
          if (yCoordsSpan > maxYCoordsDiff) {
            maxYCoordsDiff = yCoordsSpan;
          }
        }

      } else {  // if (i == 0 ...
        // Medium rows.
        std::vector<float>& xCoordsSpanInPreviousRow = xCoordsSpan[i - 1];
        std::vector<float>& yCoordsSpanInPreviousRow = yCoordsSpan[i - 1];

        for (size_t j = 1; j < xCoordsSpanInCurrentRow.size(); j++) {
          float xCoordsSpan =
              xCoordsSpanInCurrentRow[j] - xCoordsSpanInCurrentRow[j - 1];
          if (xCoordsSpan > maxXCoordsDiff) {
            maxXCoordsDiff = xCoordsSpan;
          }

          xCoordsSpan =
              xCoordsSpanInCurrentRow[j] - xCoordsSpanInPreviousRow[j];
          if (xCoordsSpan > maxXCoordsDiff) {
            maxXCoordsDiff = xCoordsSpan;
          }

          xCoordsSpan =
              xCoordsSpanInCurrentRow[j] - xCoordsSpanInPreviousRow[j - 1];
          if (xCoordsSpan > maxXCoordsDiff) {
            maxXCoordsDiff = xCoordsSpan;
          }

          float yCoordsSpan =
              yCoordsSpanInCurrentRow[j] - yCoordsSpanInCurrentRow[j - 1];
          if (yCoordsSpan > maxYCoordsDiff) {
            maxYCoordsDiff = yCoordsSpan;
          }

          yCoordsSpan =
              yCoordsSpanInCurrentRow[j] - yCoordsSpanInPreviousRow[j];
          if (yCoordsSpan > maxYCoordsDiff) {
            maxYCoordsDiff = yCoordsSpan;
          }

          yCoordsSpan =
              yCoordsSpanInCurrentRow[j] - yCoordsSpanInPreviousRow[j - 1];
          if (yCoordsSpan > maxYCoordsDiff) {
            maxYCoordsDiff = yCoordsSpan;
          }

        }  // for (size_t j = ...
      }  // if (i == 0 ...
    }  // for (size_t i = ...

    if (maxXCoordsDiff > MAX_KERNEL_DIFFERENCE) {
      gridStepX = kernelBuilder.getImageWidth() / divisorStepX;
      divisorStepX *= 2;
    }

    if (maxYCoordsDiff > MAX_KERNEL_DIFFERENCE) {
      gridStepY = kernelBuilder.getImageHeight() / divisorStepY;
      divisorStepY *= 2;
    }

  } while ((maxXCoordsDiff > MAX_KERNEL_DIFFERENCE) ||
           (maxYCoordsDiff > MAX_KERNEL_DIFFERENCE));

  return;
}  // void calculateKernelGridProperties( ...

void extractSparseKernelToImage(SparseBlurKernel& blurKernel,
                                point_value_t* imageData, int imageWidth,
                                int imageHeight, int imagePpln) {
  float xCoordMin, yCoordMin, xCoordMax, yCoordMax;
  blurKernel.calcCoordsSpan(&xCoordMin, &yCoordMin, &xCoordMax, &yCoordMax);

  int kernelSpanX = round(xCoordMax - xCoordMin);
  int kernelSpanY = round(yCoordMax - yCoordMin);

  if (kernelSpanX > imageWidth) {
    throw std::invalid_argument("Image width smaller than kernel width!");
  }

  if (kernelSpanY > imageHeight) {
    throw std::invalid_argument("Image height smaller than kernel height!");
  }

  //	kernelImage = cv::Scalar(0);

  // Clear image.
  for (int row = 0; row < imageHeight; row++) {
    memset(&imageData[row * imagePpln], 0, imageWidth * sizeof(point_value_t));
  }

  std::vector<point_coord_t> kernelPointX;
  std::vector<point_coord_t> kernelPointY;
  std::vector<point_value_t> kernelPointValue;
  blurKernel.extractKernelPoints(kernelPointX, kernelPointY, kernelPointValue);

  for (int k = 0; k < blurKernel.getKernelSize(); k++) {
    int xCoord = kernelPointX[k];  // TODO: Check minus!
    int yCoord = kernelPointY[k];  // TODO: Check minus!

    if (xCoord < 0) {
      xCoord += imageWidth;
    } else if (xCoord > imageWidth - 1) {
      xCoord -= imageWidth;
    }

    if (yCoord < 0) {
      yCoord += imageHeight;
    } else if (yCoord > imageHeight - 1) {
      yCoord -= imageHeight;
    }

    imageData[xCoord + yCoord * imagePpln] = kernelPointValue[k];
  }

  return;
}  // void extractSparseKernelToImage(...

void optimizePatchSize(int& size) {
  size = (1 << (int)ceil(log(size) / log(2)));
}

}  // namespace Deblurring
}  // namespace Test
