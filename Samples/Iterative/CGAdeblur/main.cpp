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

  //	cv::Mat blurKernel = cv::getGaussianKernel(13, 3, CV_32F);

  cv::Mat flippedBlurKernel;
  cv::flip(blurKernel, flippedBlurKernel, -1);

  //	cv::namedWindow("blurKernel", cv::WINDOW_NORMAL);
  //	cv::imshow("blurKernel", blurKernel);
  //	cv::namedWindow("flippedBlurKernel", cv::WINDOW_NORMAL);
  //	cv::imshow("flippedBlurKernel", flippedBlurKernel);
  //	cv::waitKey(0);

  cv::Mat blurredImageU(initialImage.rows, initialImage.cols, CV_8UC1);
  cv::filter2D(initialImage, blurredImageU, CV_8UC1, blurKernel);

  cv::Mat restoredImage(initialImage.rows, initialImage.cols, CV_8UC1);

  cv::Mat blurredImage(initialImage.rows, initialImage.cols, CV_32FC1);
  cv::Mat kernel(initialImage.rows, initialImage.cols, CV_32FC1);

  blurredImageU.convertTo(blurredImage, CV_32FC1, 1.0 / 255.0, 0.0);
  blurKernel.copyTo(kernel);

  cv::Mat deblurredImage;               // deblurred image
  cv::Mat residualImage;                // residual image
  cv::Mat preconditionedResidualImage;  // preconditioned residual
  cv::Mat preconditionedImage;          // preconditioned image
  cv::Mat blurredPreconditionedImage;   // blurred preconditioned image
  cv::Mat differenceResidualImage;      // temp vector

  double preconditionWeight, updateWeight, residualNorm, initialNorm;
  int it;  // Iteration counter

  // initial approximation of the restored image
  deblurredImage = blurredImage.clone();

  // initial approximation of the residual
  residualImage = blurredImage.clone();

  // Normal blur.
  cv::filter2D(deblurredImage, residualImage, CV_32FC1, kernel);
  cv::subtract(blurredImage, residualImage, residualImage);
  initialNorm = sqrt(cv::norm(residualImage, cv::NORM_L2)) /
                (residualImage.cols * residualImage.rows);

  differenceResidualImage = residualImage.clone();
  differenceResidualImage = cv::Scalar(0);

  // initial approximation of preconditioned residual
  preconditionedResidualImage = blurredImage.clone();

  // Transposed blur.
  cv::filter2D(residualImage, preconditionedResidualImage, CV_32FC1,
               flippedBlurKernel);

  // initial approximation of preconditioner
  preconditionedImage = preconditionedResidualImage.clone();

  // initial approximation of preconditioned blurred image
  blurredPreconditionedImage = blurredImage.clone();

  double bestNorm = initialNorm;
  cv::Mat bestRestoredImage = deblurredImage.clone();

  // reset iteration counter
  it = 0;

  do {
    // init
    updateWeight = 1.0 / preconditionedResidualImage.dot(residualImage);

    // alpha_k
    //  Normal blur.
    cv::filter2D(preconditionedImage, blurredPreconditionedImage, CV_32FC1,
                 kernel);
    preconditionWeight = residualImage.dot(preconditionedResidualImage) /
                         preconditionedImage.dot(blurredPreconditionedImage);

    // x_k
    cv::addWeighted(deblurredImage, 1.0, preconditionedImage,
                    preconditionWeight, 0.0, deblurredImage);

    // r_k
    residualImage.copyTo(differenceResidualImage);
    cv::addWeighted(residualImage, 1.0, blurredPreconditionedImage,
                    -preconditionWeight, 0.0, residualImage);
    cv::subtract(residualImage, differenceResidualImage,
                 differenceResidualImage);

    // norm calculation
    residualNorm = sqrt(cv::norm(residualImage, cv::NORM_L2)) /
                   (residualImage.cols * residualImage.rows);

    // z_k
    cv::filter2D(residualImage, preconditionedResidualImage, CV_32FC1,
                 flippedBlurKernel);

    // beta_k
    updateWeight *= preconditionedResidualImage.dot(differenceResidualImage);

    // If updateWeight < 0 - reinitialize search
    // Actually does not help.
    //		if (updateWeight < 0) {
    //
    //			// The restart does not improve results!
    //			break;
    //
    //			std::cout << "updateWeight: " << updateWeight <<
    //std::endl; 			std::cout << "Reinitialize CG search!" << std::endl;
    //			updateWeight = 0;
    //
    //			cv::filter2D(deblurredImage, residualImage, CV_32FC1,
    //kernel); 			cv::subtract(blurredImage, residualImage, residualImage);
    //
    //			cv::filter2D(residualImage, preconditionedResidualImage,
    //CV_32FC1, 					kernel.t());
    //		}

    cv::addWeighted(preconditionedResidualImage, 1.0, preconditionedImage,
                    1.0 * updateWeight, 0.0, preconditionedImage);

    if (residualNorm < bestNorm) {
      bestNorm = residualNorm;
      deblurredImage.copyTo(bestRestoredImage);
    }

    cv::imshow("Deblurred", deblurredImage);

    // Reblur
    cv::filter2D(deblurredImage, differenceResidualImage, CV_32FC1, blurKernel);

    cv::imshow("Reblurred", differenceResidualImage);

    char c = cv::waitKey(10);
    if (c == 27) break;

    //		std::cout << " Iteration: " << it << " Norm: " << residualNorm
    //				<< std::endl;

    std::cout << " Iteration: " << it << " Norm: " << residualNorm
              << " preconditionWeight: " << preconditionWeight
              << " updateWeight: " << updateWeight << std::endl;

    it++;
  } while ((residualNorm > 1e-6) && (it < 101));

  //	imgx.convertTo(deblurredImage, CV_8UC1, 255.0, 0.0);
  bestRestoredImage.convertTo(restoredImage, CV_8UC1, 255.0, 0.0);
  std::cout << "Best norm: " << bestNorm << std::endl;

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
