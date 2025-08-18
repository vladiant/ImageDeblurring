/*
 * main.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: vladiant
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "BlurKernelUtils.h"
#include "GyroBlurKernelBuilder.h"
#include "GyroBlurParams.h"
#include "GyroDataReader.h"
#include "SparseBlurKernel.h"

void drawKernelOnImage(Test::Deblurring::SparseBlurKernel& currentBlurKernel,
                       int gridCenterX, int gridCenterY, cv::Mat& gridImage);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout
        << "\nProgram to draw the blur kernels using\nthe gyro data from the "
           "parsegyro data file.\n"
        << std::endl;
    std::cout << "Usage: " << argv[0] << "  image_file" << std::endl;
    return 0;
  }

  // *** Sample calibration data *** //
  Test::Math::Versor<float> gyroSpaceToCameraSpaceQuaternion =
      Test::Math::Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
  bool isZAxisInvertedInGyroSpace = true;

  // Field of view for intrinsic camera matrix K
  const float fovX = 2657.2 / 2952;  // Experimentally measured
  const float fovY = 2637.0 / 1944;

  // Delay time.
  Test::Time frameTimestampDelayMinusGyroTimestampDelay =
      -3.78381e+06 - Test::timeFromSeconds<float>(0.0019);
  Test::Time delayTime = frameTimestampDelayMinusGyroTimestampDelay;

  // Exposure time
  Test::Time exposureDuration =
      Test::timeFromSeconds<float>(0.04999);  // 0.04999

  // Time for readout in seconds
  const Test::Time readoutTime = Test::timeFromSeconds<float>(0.03283);

  // *** End of sample calibration data *** //

  // Reading gyro data from JPEG file
  const int numberGyroSamples = 200;
  const std::string imageFilename(argv[1]);

  Test::Deblurring::GyroDataReader gyroDataReader(numberGyroSamples);

  std::cout << "Reading data from image file: " << imageFilename << std::endl;

  gyroDataReader.readDataFromJPEGFile(imageFilename);

  std::cout << "Correcting angular velocities. " << std::endl;

  gyroDataReader.correctAngularVelocities(gyroSpaceToCameraSpaceQuaternion,
                                          isZAxisInvertedInGyroSpace);

  std::cout << "Reading capture timestamp. " << std::endl;

  Test::Time captureTimeStamp = gyroDataReader.getCaptureTimestamp();

  std::cout << "Capture timestamp: " << captureTimeStamp << std::endl;

  // Load image and print sampling points.
  cv::Mat initialImage = cv::imread("blurred.bmp" /*imageFilename*/);
  cv::namedWindow(imageFilename, cv::WINDOW_AUTOSIZE);

  int imageWidth = initialImage.cols;
  int imageHeight = initialImage.rows;

  // Set parameters of the gyro blur kernel.
  Test::Deblurring::GyroBlurParams gyroBlurParams;
  gyroBlurParams.setCameraParams(imageWidth, imageHeight, fovX * imageWidth,
                                 fovY * imageHeight);
  gyroBlurParams.setTimeParams(captureTimeStamp, exposureDuration, readoutTime,
                               delayTime);

  // Read versors.
  std::vector<Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> > >
      samples;
  gyroDataReader.getAngularValuesInInterval(gyroBlurParams.getTimeFirstSample(),
                                            gyroBlurParams.getTimeLastSample(),
                                            samples);

  // Show the versors.
  std::cout << "FirstSample: " << gyroBlurParams.getTimeFirstSample()
            << std::endl;
  std::cout << "LastSample: " << gyroBlurParams.getTimeLastSample()
            << std::endl;
  for (size_t i = 0; i < samples.size(); i++) {
    std::cout << i << " : " << samples[i].time << " : "
              << samples[i].orientation << std::endl;
  }

  // Blur kernel builder.
  Test::Deblurring::GyroBlurKernelBuilder kernelBuilder(gyroBlurParams,
                                                        samples);

  int gridStepX;
  int gridStepY;
  float maxXCoordsSpan;
  float maxYCoordsSpan;
  Test::Deblurring::calculateKernelGridProperties(
      kernelBuilder, gridStepX, gridStepY, maxXCoordsSpan, maxYCoordsSpan);

  // Print blur kernel properties.
  std::cout << "gridStepX= " << gridStepX << "  gridStepY= " << gridStepY
            << std::endl;
  std::cout << "maxXCoordsSpan: " << round(maxXCoordsSpan)
            << "  maxYCoordsSpan: " << round(maxYCoordsSpan) << std::endl;

  // TODO: Routine to set the image patches for processing.
  const float PATCH_SCALE_UP = 3.0;
  int patchWidth = round(PATCH_SCALE_UP * maxXCoordsSpan) + gridStepX;
  int patchHeight = round(PATCH_SCALE_UP * maxYCoordsSpan) + gridStepY;
  Test::Deblurring::optimizePatchSize(patchWidth);
  Test::Deblurring::optimizePatchSize(patchHeight);

  // Print blur kernel properties.
  std::cout << "Patch width: " << patchWidth << std::endl;
  std::cout << "Patch height: " << patchHeight << std::endl;

  cv::Mat kernelImage(patchHeight, patchWidth, CV_32FC1);

  int patchImageType = CV_32FC1;
  typedef float PatchDataT;

  cv::Mat patchImage(patchHeight, patchWidth, patchImageType);
  cv::Mat outputPatchImage(gridStepY, gridStepX, patchImageType);
  cv::Mat grayImage(imageHeight, imageWidth, CV_8UC1);
  cv::cvtColor(initialImage, grayImage, cv::COLOR_BGR2GRAY, 1);
  cv::Mat outputGrayImage(imageHeight, imageWidth, CV_8UC1);

  cv::namedWindow("Gray Image", cv::WINDOW_NORMAL);
  cv::imshow("Gray Image", grayImage);

  cv::Mat gridImage = initialImage.clone();

  uint8_t* pInitialImageData = (uint8_t*)grayImage.data;
  uint8_t* pOutputImageData = (uint8_t*)outputGrayImage.data;
  PatchDataT* mPatchDataBuffer =
      (PatchDataT*)patchImage.data;  // patchHeight, patchWidth

  std::vector<Test::Deblurring::SparseBlurKernel> blurKernels;

  // ProcessImageGridPatches

  for (int gridCenterY = gridStepY / 2; gridCenterY < imageHeight;
       gridCenterY += gridStepY) {
    for (int gridCenterX = gridStepX / 2; gridCenterX < imageWidth;
         gridCenterX += gridStepX) {
      {
        // Draw centers.
        gridImage.data[3 * (gridCenterX) + 0 + 3 * (gridCenterY)*imageWidth] =
            0;
        gridImage.data[3 * (gridCenterX) + 1 + 3 * (gridCenterY)*imageWidth] =
            255;
        gridImage.data[3 * (gridCenterX) + 2 + 3 * (gridCenterY)*imageWidth] =
            0;

        // Draw lines.
        int lineX1 = gridCenterX - gridStepX / 2;
        int lineY1 = gridCenterY - gridStepY / 2;
        int lineX2 = gridCenterX + gridStepX / 2;
        int lineY2 = gridCenterY + gridStepY / 2;
        cv::line(gridImage, cv::Point(lineX1, lineY1),
                 cv::Point(lineX2, lineY1), cv::Scalar(0, 0, 255));
        cv::line(gridImage, cv::Point(lineX1, lineY1),
                 cv::Point(lineX1, lineY2), cv::Scalar(0, 0, 255));
      }

      // Get patch from image.
      patchImage = cv::Scalar(0);
      Test::Deblurring::getPatchFromImage(
          gridCenterX - patchWidth / 2, gridCenterY - patchHeight / 2,
          pInitialImageData, imageWidth, imageHeight, imageWidth,
          mPatchDataBuffer, patchWidth, patchHeight, patchWidth);

      // Calculate blur kernel.
      Test::Deblurring::SparseBlurKernel currentBlurKernel;
      kernelBuilder.calculateAtPoint(gridCenterX, gridCenterY,
                                     currentBlurKernel);
      blurKernels.push_back(currentBlurKernel);

      // Patch processing here.

      // Put back output patch - without intermediate cropping.
      PatchDataT* pOutputCropPatch = (PatchDataT*)mPatchDataBuffer +
                                     (patchWidth - gridStepX + 1) / 2 +
                                     patchWidth * (patchHeight - gridStepY) / 2;
      Test::Deblurring::putPatchInImage(
          gridCenterX - gridStepX / 2, gridCenterY - gridStepY / 2,
          pOutputCropPatch, gridStepX, gridStepY, patchWidth, pOutputImageData,
          imageWidth, imageHeight, imageWidth);

      // Draw kernel on image.
      drawKernelOnImage(currentBlurKernel, gridCenterX, gridCenterY, gridImage);

      // Draw kernel as separate image.
      extractSparseKernelToImage(
          currentBlurKernel, (Test::Deblurring::point_value_t*)kernelImage.data,
          kernelImage.cols, kernelImage.rows, kernelImage.cols);

      cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
      cv::imshow("Kernel", kernelImage);

      cv::namedWindow("Patch", cv::WINDOW_NORMAL);
      cv::imshow("Patch", patchImage / 255);

      cv::namedWindow("Output Gray Image", cv::WINDOW_NORMAL);
      cv::imshow("Output Gray Image", outputGrayImage);

      cv::waitKey(10);

    }  // for (int gridCenterX = ...
  }  // for (int gridCenterY = ...

  cv::imshow(imageFilename, gridImage);
  cv::imwrite("result.bmp", gridImage);
  cv::imwrite("deblurred.bmp", outputGrayImage);
  cv::waitKey(0);

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}

void drawKernelOnImage(Test::Deblurring::SparseBlurKernel& currentBlurKernel,
                       int gridCenterX, int gridCenterY, cv::Mat& gridImage) {
  std::vector<Test::Deblurring::point_coord_t> kernelPointX;
  std::vector<Test::Deblurring::point_coord_t> kernelPointY;
  std::vector<Test::Deblurring::point_value_t> kernelPointValue;
  currentBlurKernel.extractKernelPoints(kernelPointX, kernelPointY,
                                        kernelPointValue);

  for (int k = 0; k < currentBlurKernel.getKernelSize(); k++) {
    int coordX = kernelPointX[k] + gridCenterX;
    int coordY = kernelPointY[k] + gridCenterY;

    if ((coordX < 0) || (coordX > gridImage.cols - 1)) {
      continue;
    }

    if ((coordY < 0) || (coordY > gridImage.rows - 1)) {
      continue;
    }

    gridImage.data[3 * coordX + 0 + 3 * coordY * gridImage.cols] = 0;
    gridImage.data[3 * coordX + 1 + 3 * coordY * gridImage.cols] =
        (255.0 * (kernelPointValue[k]));
    gridImage.data[3 * coordX + 2 + 3 * coordY * gridImage.cols] =
        (255.0 * (kernelPointValue[k]));
  }
}  // drawKernelOnImage(...
