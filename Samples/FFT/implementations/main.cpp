/*
 *  Created on: 02.02.2015
 *      Author: vladiant
 */

/*
 http://paulbourke.net/miscellaneous/dft/
 http://www.codeproject.com/Articles/9388/How-to-implement-the-FFT-algorithm
 http://www.drdobbs.com/cpp/a-simple-and-efficient-fft-implementatio/199500857
 http://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "fft.hpp"
#include "impl1.hpp"
#include "impl2.hpp"
#include "impl3.hpp"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " image_name\n";
    return EXIT_FAILURE;
  }

  std::string inputImageName = argv[1];

  cv::Mat inputImage;

  if ((inputImage = cv::imread(inputImageName, cv::IMREAD_GRAYSCALE)).empty()) {
    std::cout << "Error reading " << inputImage << '\n';
    return EXIT_FAILURE;
  }

  int imageWidth = inputImage.cols;
  int imageHeight = inputImage.rows;

  std::vector<cv::Mat> inputImageChannels(2);
  inputImageChannels[0].create(inputImage.rows, inputImage.cols, CV_32FC1);
  inputImageChannels[0] = cv::Scalar(0);
  inputImageChannels[1].create(inputImage.rows, inputImage.cols, CV_32FC1);
  inputImageChannels[1] = cv::Scalar(0);
  inputImage.convertTo(inputImageChannels[0], CV_32FC1, 1.0 / 255.0, 0.0);

  cv::Mat inputFftImage;  //(inputImage.rows, inputImage.cols, CV_32FC2);
  cv::merge(inputImageChannels, inputFftImage);
  float* pInputFftImage = (float*)inputFftImage.data;

  complex_t* testImage = (complex_t*)pInputFftImage;

  double t;

  t = (double)cv::getTickCount();

  cv::dft(inputFftImage, inputFftImage);
  cv::dft(inputFftImage, inputFftImage, cv::DFT_INVERSE | cv::DFT_SCALE);

  t = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
  std::cout << "OpenCV implementation [ms]: " << t << std::endl;

  t = (double)cv::getTickCount();

  FFT2D((float*)inputFftImage.data, imageWidth, imageHeight, 1);
  FFT2D((float*)inputFftImage.data, imageWidth, imageHeight, -1);

  t = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
  std::cout << "First implementation [ms]: " << t << std::endl;

  t = (double)cv::getTickCount();

  FFT2D_((float*)inputFftImage.data, imageWidth, imageHeight, 1);
  FFT2D_((float*)inputFftImage.data, imageWidth, imageHeight, -1);

  t = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
  std::cout << "Third implementation [ms]: " << t << std::endl;

  t = (double)cv::getTickCount();

  fft2d(pInputFftImage, imageWidth, imageHeight);
  ifft2d(pInputFftImage, imageWidth, imageHeight);

  t = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
  std::cout << "Second implementation [ms]: " << t << std::endl;

  t = (double)cv::getTickCount();

  fft2d(pInputFftImage, pInputFftImage, imageWidth, imageHeight);
  fftInverse2d(pInputFftImage, pInputFftImage, imageWidth, imageHeight);

  t = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
  std::cout << "Naiive implementation [ms]: " << t << std::endl;

  std::vector<cv::Mat> outputImageChannels(2);
  cv::split(inputFftImage, outputImageChannels);

  cv::namedWindow(inputImageName, cv::WINDOW_NORMAL);
  cv::namedWindow("Real", cv::WINDOW_NORMAL);
  cv::namedWindow("Imaginary", cv::WINDOW_NORMAL);

  cv::imshow(inputImageName, inputImage);
  cv::imshow("Real", outputImageChannels[0]);
  cv::imshow("Imaginary", outputImageChannels[1]);

  cv::waitKey(0);

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
