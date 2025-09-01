/*
 * main.cpp
 *
 *  Created on: 29.11.2014
 *      Author: vladiant
 */

#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CgDeblurrer.h"
#include "CgTmDeblurrer.h"
#include "CgTvDeblurrer.h"
#include "CutOffFftDeblurrer.h"
#include "GradientFftRegularizer.h"
#include "GradientsIterationRegularizer.h"
#include "HLFftRegularizer.h"
#include "HLIterationRegularizer.h"
#include "HLTwoThirdsFftRegularizer.h"
#include "ImageDeblurrer.h"
#include "InverseFftDeblurrer.h"
#include "LaplaceFftRegularizer.h"
#include "LaplaceIterationRegularizer.h"
#include "LrDeblurrer.h"
#include "LrTmDeblurrer.h"
#include "LrTvDeblurrer.h"
#include "LsFftDeblurrer.h"
#include "MaskLaplaceIterationRegularizer.h"
#include "OcvFftDeblurrer.h"
#include "OcvFftTmDeblurrer.h"
#include "OcvFftTvDeblurrer.h"
#include "TmFftRegularizer.h"
#include "TmIterationRegularizer.h"
#include "TvFftRegularizer.h"
#include "TvIterationRegularizer.h"
#include "vanCittertDeblurrer.h"
#include "vanCittertTmDeblurrer.h"
#include "vanCittertTvDeblurrer.h"

void createTestImage(cv::Mat& image);

void prepareBlurKernel(cv::Mat& blurKernel);

void prepareSpareBlurKernel(
    Test::Deblurring::SparseBlurKernel& sparseBlurKernel);

void convertSparseToDenseKernel(
    const Test::Deblurring::SparseBlurKernel& testSparseBlurKernel,
    cv::Mat& blurKernelSparse);

int main(int argc, char* argv[]) {
  const std::string initialImageWindowName("Initial Image");
  const std::string kernelImageWindowName("Blur Kernel");
  const std::string sparseKernelImageWindowName("Sparse Blur Kernel");
  const std::string blurredImageWindowName("Blurred Image");
  const std::string sparseBlurredImageWindowName("Sparse Blurred Image");
  const std::string deblurredImageWindowName("Deblurred Image");
  const std::string sparseDeblurredImageWindowName("Sparse Deblurred Image");

  cv::Mat initialImage;

  if (argc > 1) {
    std::cout << "Loading: " << argv[1] << std::endl;
    initialImage = cv::imread(std::string(argv[1]), cv::IMREAD_GRAYSCALE);
    if (initialImage.empty()) {
      std::cout << "Loading of " << argv[1] << " failed, using synthetic image."
                << std::endl;
    }
  }

  if (initialImage.empty()) {
    initialImage.create(480, 640, CV_8UC1);
    createTestImage(initialImage);
  }

  cv::Mat blurKernel;
  prepareBlurKernel(blurKernel);

  Test::Deblurring::SparseBlurKernel testSparseBlurKernel;
  prepareSpareBlurKernel(testSparseBlurKernel);

  cv::Mat blurKernelSparse;
  convertSparseToDenseKernel(testSparseBlurKernel, blurKernelSparse);

  cv::Mat blurredImage(initialImage.rows, initialImage.cols, CV_8UC1);
  cv::filter2D(initialImage, blurredImage, CV_8UC1, blurKernel);
  cv::imwrite("blurred_lena.bmp", blurredImage);
  cv::Mat gaussian_noise = blurredImage.clone();
  gaussian_noise = cv::Scalar(0);
  cv::randn(gaussian_noise, 0, 5);
  cv::add(blurredImage, gaussian_noise, blurredImage);

  cv::Mat sparseBlurredImage(initialImage.rows, initialImage.cols, CV_8UC1);
  cv::filter2D(initialImage, sparseBlurredImage, CV_8UC1, blurKernelSparse);
  cv::add(sparseBlurredImage, gaussian_noise, sparseBlurredImage);

  // FFT regularizers
  Test::Deblurring::TmFftRegularizer testTmFftRegularizer(initialImage.cols,
                                                          initialImage.rows);
  testTmFftRegularizer.setRegularizationWeight(0.01);

  Test::Deblurring::GradientFftRegularizer testGradientFftRegularizer(
      initialImage.cols, initialImage.rows);
  testGradientFftRegularizer.setRegularizationWeight(0.00002);

  Test::Deblurring::LaplaceFftRegularizer testLaplaceFftRegularizer(
      initialImage.cols, initialImage.rows);
  testLaplaceFftRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::TvFftRegularizer testTvFftRegularizer(initialImage.cols,
                                                          initialImage.rows);
  testTvFftRegularizer.setRegularizationWeight(0.0000001);

  Test::Deblurring::HLTwoThirdsFftRegularizer testHLTwoThirdsFftRegularizer(
      initialImage.cols, initialImage.rows);
  testHLTwoThirdsFftRegularizer.setRegularizationWeight(0.00000005);

  Test::Deblurring::HLFftRegularizer testHLFftRegularizer(initialImage.cols,
                                                          initialImage.rows);
  testHLFftRegularizer.setRegularizationWeight(0.00000005);
  testHLFftRegularizer.setPowerRegularizationNorm(0.8);

  Test::Deblurring::FftRegularizer* testFftRegularizer = &testTvFftRegularizer;

  // Regularizers
  Test::Deblurring::TmIterationRegularizer testTmIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testTmIterationRegularizer.setRegularizationWeight(0.01);

  Test::Deblurring::LaplaceIterationRegularizer testLaplaceIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::GradientsIterationRegularizer
      testGradientsIterationRegularizer(initialImage.cols, initialImage.rows);
  testGradientsIterationRegularizer.setRegularizationWeight(0.0001);

  Test::Deblurring::TvIterationRegularizer testTvIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testTvIterationRegularizer.setRegularizationWeight(0.0001);

  Test::Deblurring::HLIterationRegularizer testHlIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testHlIterationRegularizer.setRegularizationWeight(0.00005);
  testHlIterationRegularizer.setPowerRegularizationNorm(0.8);

  Test::Deblurring::MaskLaplaceIterationRegularizer
      testMaskLaplaceIterationRegularizer(initialImage.cols, initialImage.rows);
  testMaskLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::IterationRegularizer* testRegularizer =
      &testHlIterationRegularizer;

  // OpenCV FFT deblurrers
  Test::Deblurring::OcvFftDeblurrer testOcvFftDeblurrer(initialImage.cols,
                                                        initialImage.rows);

  Test::Deblurring::OcvFftTmDeblurrer testOcvFftTmDeblurrer(initialImage.cols,
                                                            initialImage.rows);
  //	testOcvFftTmDeblurrer.setRegularizationWeight(0.005); // Laplace
  //	testOcvFftTmDeblurrer.setRegularizationWeight(0.005); // TM
  //	testOcvFftTmDeblurrer.setRegularizationWeight(0.0001); // Gradients
  testOcvFftTmDeblurrer.setRegularizationWeight(0.00001);

  Test::Deblurring::OcvFftTvDeblurrer testOcvFftTvDeblurrer(initialImage.cols,
                                                            initialImage.rows);
  testOcvFftTvDeblurrer.setRegularizationWeight(0.0000001);

  // FFT deblurrers
  Test::Deblurring::InverseFftDeblurrer testInverseFftDeblurrer(
      initialImage.cols, initialImage.rows);

  Test::Deblurring::CutOffFftDeblurrer testCutOffFftDeblurrer(
      initialImage.cols, initialImage.rows);
  testCutOffFftDeblurrer.setCutOffFrequencyX(100);
  testCutOffFftDeblurrer.setCutOffFrequencyY(100);

  Test::Deblurring::LsFftDeblurrer testLsFftDeblurrer(initialImage.cols,
                                                      initialImage.rows);
  testLsFftDeblurrer.setRegularizer(testFftRegularizer);

  // Iterative deblurrers
  Test::Deblurring::vanCittertDeblurrer testVanCittertDeblurrer(
      initialImage.cols, initialImage.rows);
  testVanCittertDeblurrer.setMinimalNorm(5e-7);
  testVanCittertDeblurrer.setRegularizer(testRegularizer);

  Test::Deblurring::LrDeblurrer testLrDeblurrer(initialImage.cols,
                                                initialImage.rows);
  testLrDeblurrer.setMaxIterations(100);
  testLrDeblurrer.setMinimalNorm(5e-7);
  testLrDeblurrer.setRegularizer(testRegularizer);

  Test::Deblurring::CgDeblurrer testCgDeblurrer(initialImage.cols,
                                                initialImage.rows);
  testCgDeblurrer.setMaxIterations(100);
  testCgDeblurrer.setMinimalNorm(5e-7);
  testCgDeblurrer.setRegularizer(testRegularizer);

  // Experimental iterative deblurrers
  Test::Deblurring::vanCittertTmDeblurrer testVanCittertTmDeblurrer(
      initialImage.cols, initialImage.rows);
  testVanCittertTmDeblurrer.setMaxIterations(100);
  testVanCittertTmDeblurrer.setMinimalNorm(1e-6);
  //	testVanCittertTmDeblurrer.setRegularizationWeight(0.01); // Laplacian
  //	testVanCittertTmDeblurrer.setRegularizationWeight(0.00001); // Gradients
  //	testVanCittertTmDeblurrer.setRegularizationWeight(0.0001); // Tikhonov

  Test::Deblurring::vanCittertTvDeblurrer testVanCittertTvDeblurrer(
      initialImage.cols, initialImage.rows);
  testVanCittertTvDeblurrer.setMaxIterations(100);
  testVanCittertTvDeblurrer.setRegularizationWeight(0.00003);
  testVanCittertTvDeblurrer.setMinimalNorm(5e-7);

  Test::Deblurring::LrTmDeblurrer testLrTmDeblurrer(initialImage.cols,
                                                    initialImage.rows);
  testLrTmDeblurrer.setMaxIterations(100);
  testLrTmDeblurrer.setMinimalNorm(5e-7);
  //	testLrTmDeblurrer.setRegularizationWeight(0.01); // Laplace, TM
  //	testLrTmDeblurrer.setRegularizationWeight(0.0001); // Gradients
  testLrTmDeblurrer.setRegularizationWeight(0.0001);

  Test::Deblurring::LrTvDeblurrer testLrTvDeblurrer(initialImage.cols,
                                                    initialImage.rows);
  testLrTvDeblurrer.setMaxIterations(100);
  testLrTvDeblurrer.setMinimalNorm(5e-7);
  //	testLrTvDeblurrer.setRegularizationWeight(0.00005); // adapted TV
  testLrTvDeblurrer.setRegularizationWeight(0.00001);

  Test::Deblurring::CgTmDeblurrer testCgTmDeblurrer(initialImage.cols,
                                                    initialImage.rows);
  testCgTmDeblurrer.setMaxIterations(100);
  testCgTmDeblurrer.setMinimalNorm(5e-7);
  //	testCgTmDeblurrer.setRegularizationWeight(0.0001); // Gradients
  //	testCgTmDeblurrer.setRegularizationWeight(0.01); // TM, Laplace
  testCgTmDeblurrer.setRegularizationWeight(0.01);

  Test::Deblurring::CgTvDeblurrer testCgTvDeblurrer(initialImage.cols,
                                                    initialImage.rows);
  testCgTvDeblurrer.setMaxIterations(100);
  testCgTvDeblurrer.setMinimalNorm(5e-7);
  testCgTvDeblurrer.setRegularizationWeight(0.0000001);

  cv::Mat deblurredImage(initialImage.rows, initialImage.cols, CV_8UC1);
  cv::Mat sparseDeblurredImage(initialImage.rows, initialImage.cols, CV_8UC1);

  // testVanCittertDeblurrer, testLrDeblurrer, testCgDeblurrer
  Test::Deblurring::ImageDeblurrer* pImageDeblurrer = &testLrDeblurrer;

  //	Test::Deblurring::ImageDeblurrer& testImageDeblurrer = testLrDeblurrer;

  // TODO: Add time measurement!

  //	testImageDeblurrer((uint8_t*) blurredImage.data, blurredImage.cols,
  //			(Test::Deblurring::point_value_t*) blurKernel.data,
  // blurKernel.cols, 			blurKernel.rows, blurKernel.cols,
  // (uint8_t*) deblurredImage.data, 			deblurredImage.cols);
  //
  //	testImageDeblurrer((uint8_t*) sparseBlurredImage.data,
  //			sparseBlurredImage.cols, testSparseBlurKernel,
  //			(uint8_t*) sparseDeblurredImage.data,
  // sparseDeblurredImage.cols);

  (*pImageDeblurrer)((uint8_t*)blurredImage.data, blurredImage.cols,
                     (Test::Deblurring::point_value_t*)blurKernel.data,
                     blurKernel.cols, blurKernel.rows, blurKernel.cols,
                     (uint8_t*)deblurredImage.data, deblurredImage.cols);

  (*pImageDeblurrer)((uint8_t*)sparseBlurredImage.data, sparseBlurredImage.cols,
                     testSparseBlurKernel, (uint8_t*)sparseDeblurredImage.data,
                     sparseDeblurredImage.cols);

  float isnrDeblurring =
      10.0 *
      log(cv::norm(initialImage - blurredImage, cv::NORM_L2) /
          cv::norm(initialImage - deblurredImage, cv::NORM_L2)) /
      log(10.0);

  float isnrSparseDeblurring =
      10.0 *
      log(cv::norm(initialImage - sparseBlurredImage, cv::NORM_L2) /
          cv::norm(initialImage - sparseDeblurredImage, cv::NORM_L2)) /
      log(10.0);

  std::cout << "ISNR deblurring: " << isnrDeblurring << std::endl;
  std::cout << "ISNR sparse deblurring: " << isnrSparseDeblurring << std::endl;

  cv::namedWindow(initialImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(initialImageWindowName, initialImage);

  cv::namedWindow(blurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(blurredImageWindowName, blurredImage);

  cv::namedWindow(sparseBlurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(sparseBlurredImageWindowName, sparseBlurredImage);

  cv::namedWindow(deblurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(deblurredImageWindowName, deblurredImage);

  cv::namedWindow(sparseDeblurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(sparseDeblurredImageWindowName, sparseDeblurredImage);

  cv::namedWindow(kernelImageWindowName, cv::WINDOW_NORMAL);
  cv::imshow(kernelImageWindowName, blurKernel);

  cv::namedWindow(sparseKernelImageWindowName, cv::WINDOW_NORMAL);
  cv::imshow(sparseKernelImageWindowName, blurKernelSparse);

  std::cout << "Press <ESC> to exit..." << std::endl;

  while (cv::waitKey(0) != 27) {
  }

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}

void createTestImage(cv::Mat& image) {
  image = cv::Scalar(0);

  cv::rectangle(image, cv::Point(1 * image.cols / 5, 1 * image.rows / 5),
                cv::Point(3 * image.cols / 5, 7 * image.rows / 10),
                cv::Scalar(128), -1);

  cv::circle(image, cv::Point(13 * image.cols / 20, 3 * image.rows / 5),
             cv::min(image.cols / 4, image.rows / 4), cv::Scalar(255), -1);

  //	cv::circle(image, cv::Point(image.cols / 2, image.rows / 2),
  //			cv::min(image.cols / 4, image.rows / 4),
  // cv::Scalar(255), -1);
}

void prepareBlurKernel(cv::Mat& blurKernel) {
  blurKernel.create(cv::Size(13, 13), CV_32FC1);

  //	blurKernel = cv::Scalar(1.0 / (blurKernel.rows * blurKernel.cols));

  blurKernel = cv::Scalar(0);
  cv::line(blurKernel, cv::Point(blurKernel.cols / 2, blurKernel.rows / 2),
           cv::Point(blurKernel.cols - 1, blurKernel.rows - 1), cv::Scalar(1.0),
           1);
  blurKernel = blurKernel * (1.0 / cv::sum(blurKernel).val[0]);
}

void prepareSpareBlurKernel(
    Test::Deblurring::SparseBlurKernel& sparseBlurKernel) {
  sparseBlurKernel.addToPointValue(0, 0, 0.127524);
  sparseBlurKernel.addToPointValue(1, 0, 0.136666);
  sparseBlurKernel.addToPointValue(2, 0, 0.0489374);
  sparseBlurKernel.addToPointValue(0, 1, 7.24361e-05);
  sparseBlurKernel.addToPointValue(1, 1, 0.0121379);
  sparseBlurKernel.addToPointValue(2, 1, 0.0953891);
  sparseBlurKernel.addToPointValue(3, 1, 0.00993582);
  sparseBlurKernel.addToPointValue(2, 2, 0.0555902);
  sparseBlurKernel.addToPointValue(3, 2, 0.0352052);
  sparseBlurKernel.addToPointValue(2, 3, 0.0314632);
  sparseBlurKernel.addToPointValue(3, 3, 0.0528525);
  sparseBlurKernel.addToPointValue(2, 4, 0.020618);
  sparseBlurKernel.addToPointValue(3, 4, 0.0761621);
  sparseBlurKernel.addToPointValue(2, 5, 0.0129349);
  sparseBlurKernel.addToPointValue(3, 5, 0.108659);
  sparseBlurKernel.addToPointValue(2, 6, 0.0028684);
  sparseBlurKernel.addToPointValue(3, 6, 0.157102);
  sparseBlurKernel.addToPointValue(4, 6, 0.00476125);
  sparseBlurKernel.addToPointValue(3, 7, 0.0102104);
  sparseBlurKernel.addToPointValue(4, 7, 0.000878081);
}

void convertSparseToDenseKernel(
    const Test::Deblurring::SparseBlurKernel& testSparseBlurKernel,
    cv::Mat& blurKernelSparse) {
  Test::Deblurring::point_value_t xCoordMin;
  Test::Deblurring::point_value_t yCoordMin;
  Test::Deblurring::point_value_t xCoordMax;
  Test::Deblurring::point_value_t yCoordMax;

  // TODO: Hack - solve!
  const_cast<Test::Deblurring::SparseBlurKernel&>(testSparseBlurKernel)
      .calcCoordsSpan(&xCoordMin, &yCoordMin, &xCoordMax, &yCoordMax);

  int sparseKernelRadiusX = std::max(abs(xCoordMin), abs(xCoordMax));
  int sparseKernelRadiusY = std::max(abs(yCoordMin), abs(yCoordMax));
  blurKernelSparse.create(
      cv::Size(2 * sparseKernelRadiusX + 1, 2 * sparseKernelRadiusY + 1),
      CV_32FC1);

  blurKernelSparse = cv::Scalar(0);

  std::vector<Test::Deblurring::point_coord_t> kernelPointX;
  std::vector<Test::Deblurring::point_coord_t> kernelPointY;
  std::vector<Test::Deblurring::point_value_t> kernelPointValue;

  // TODO: Hack - solve!
  const_cast<Test::Deblurring::SparseBlurKernel&>(testSparseBlurKernel)
      .extractKernelPoints(kernelPointX, kernelPointY, kernelPointValue);

  for (int k = 0; k < testSparseBlurKernel.getKernelSize(); k++) {
    int coordX = kernelPointX[k] + blurKernelSparse.cols / 2;
    int coordY = kernelPointY[k] + blurKernelSparse.rows / 2;

    float* imageData = (float*)blurKernelSparse.data;

    imageData[coordX + coordY * blurKernelSparse.cols] = kernelPointValue[k];
  }

  return;
}
