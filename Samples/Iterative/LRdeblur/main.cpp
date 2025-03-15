#include <stdlib.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void createTestImage(cv::Mat& image);

int main(int argc, char* argv[]) {
  const std::string initialImageWindowName("Initial Image");
  const std::string blurredImageWindowName("Blurred Image");
  const std::string deblurredImageWindowName("Deblurred Image");

  cv::Mat initialImage(480, 640, CV_8UC1);
  createTestImage(initialImage);

  cv::Mat blurKernel(13, 13, CV_32FC1);
  //	blurKernel = cv::Scalar(1.0 / (blurKernel.rows * blurKernel.cols));

  blurKernel = cv::Scalar(0);
  cv::line(blurKernel, cv::Point(blurKernel.cols / 2, blurKernel.rows / 2),
           cv::Point(blurKernel.cols - 1, blurKernel.rows - 1), cv::Scalar(1.0),
           1);
  blurKernel = blurKernel * (1.0 / cv::sum(blurKernel).val[0]);

  //	cv::Mat blurKernel = cv::getGaussianKernel(12, 3, CV_32F);

  cv::Mat flippedBlurKernel;
  cv::flip(blurKernel, flippedBlurKernel, -1);

  cv::Mat blurredImageU(initialImage.rows, initialImage.cols, CV_8UC1);
  cv::filter2D(initialImage, blurredImageU, CV_8UC1, blurKernel);

  cv::Mat gaussian_noise = blurredImageU.clone();
  gaussian_noise = cv::Scalar(0);
  cv::randn(gaussian_noise, 0, 5);
  cv::add(blurredImageU, gaussian_noise, blurredImageU);

  cv::Mat restoredImage(initialImage.rows, initialImage.cols, CV_8UC1);

  cv::Mat blurredImage(initialImage.rows, initialImage.cols, CV_32FC1);
  cv::Mat kernel(initialImage.rows, initialImage.cols, CV_32FC1);

  blurredImageU.convertTo(blurredImage, CV_32FC1, 1.0 / 255.0, 0.0);
  blurKernel.copyTo(kernel);

  float residualNorm;  // norm of the images

  // Work images
  cv::Mat deblurredImage = blurredImage.clone();

  // Initial approximation of blurred image.
  cv::filter2D(deblurredImage, blurredImage, CV_32FC1, flippedBlurKernel);

  cv::Mat reblurredImage = blurredImage.clone();
  cv::Mat reblurredTransposedImage = blurredImage.clone();
  cv::Mat weightImage = blurredImage.clone();

  double regularizationWeight = 0.001;
  cv::Mat regularizationImage = blurredImage.clone();
  cv::Mat regularizationImageTransposed = blurredImage.clone();

  // Richardson-Lucy starts here
  int iteration = 0;
  do {
    cv::filter2D(deblurredImage, reblurredImage, CV_32FC1, kernel);
    cv::filter2D(reblurredImage, reblurredTransposedImage, CV_32FC1,
                 flippedBlurKernel);

    // Add regularization
    cv::Laplacian(deblurredImage, regularizationImageTransposed, CV_32FC1);
    cv::Laplacian(regularizationImageTransposed, regularizationImage, CV_32FC1);
    //		deblurredImage.copyTo(regularizationImage);
    cv::addWeighted(regularizationImage, regularizationWeight,
                    reblurredTransposedImage, 1.0, 0.0,
                    reblurredTransposedImage);

    cv::divide(blurredImage, reblurredTransposedImage, weightImage);

    cv::Mat oldDeblurredImage = deblurredImage.clone();

    // pixel by pixel multiply
    cv::multiply(weightImage, deblurredImage, deblurredImage);

    residualNorm =
        sqrt(cv::norm(deblurredImage - oldDeblurredImage, cv::NORM_L2)) /
        (blurredImage.cols * blurredImage.rows);

    std::cout << iteration << "  " << residualNorm << std::endl;

    cv::imshow("Deblurred", deblurredImage);
    char c = cv::waitKey(10);

    if (c == 27) break;

    iteration++;
  } while ((residualNorm > 5e-7) && (iteration < 301));

  deblurredImage.convertTo(restoredImage, CV_8UC1, 255.0, 0.0);

  cv::namedWindow(initialImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(initialImageWindowName, initialImage);

  cv::namedWindow(blurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(blurredImageWindowName, blurredImageU);

  cv::namedWindow(deblurredImageWindowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(deblurredImageWindowName, restoredImage);

  cv::waitKey(0);

  return EXIT_SUCCESS;
}

void createTestImage(cv::Mat& image) {
  image = cv::Scalar(0);

  cv::rectangle(image, cv::Point(1 * image.cols / 5, 1 * image.rows / 5),
                cv::Point(3 * image.cols / 5, 7 * image.rows / 10),
                cv::Scalar(128), -1);

  cv::circle(image, cv::Point(13 * image.cols / 20, 3 * image.rows / 5),
             cv::min(image.cols / 4, image.rows / 4), cv::Scalar(255), -1);
}
