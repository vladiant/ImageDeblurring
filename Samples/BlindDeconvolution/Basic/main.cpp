#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Simulate blur
Mat addBlur(const Mat& img, Mat& kernel, int ksize = 21, double sigma = 3.0) {
  kernel = getGaussianKernel(ksize, sigma, CV_32F);
  kernel = kernel * kernel.t();
  Mat blurred;
  filter2D(img, blurred, -1, kernel, Point(-1, -1), 0, BORDER_REFLECT);
  return blurred;
}

// Normalize kernel to sum to 1
void normalizeKernel(Mat& kernel) {
  kernel = max(kernel, 0);
  kernel /= sum(kernel)[0] + 1e-8;
}

cv::Mat shrinkKernel(const cv::Mat& padded, const cv::Size& size) {
  // Get the size of the matrix.
  int rows = size.height;
  int cols = size.width;

  // Split the input matrix into quadrants.
  // Top-left to bottom-right and top-right to bottom-left
  int midRow = rows / 2;
  int midCol = cols / 2;

  const int endCol = midCol;
  const int endRow = midRow;

  // Top-left quadrant
  cv::Mat q0(padded, cv::Rect(padded.cols - midCol, padded.rows - midRow,
                              midCol, midRow));
  // Bottom-right quadrant
  cv::Mat q1(padded, cv::Rect(0, 0, endCol + 1, endRow + 1));
  // Bottom-left quadrant
  cv::Mat q2(padded, cv::Rect(padded.cols - midCol, 0, midCol, endRow + 1));
  // Top-right quadrant
  cv::Mat q3(padded, cv::Rect(0, padded.rows - midRow, endCol + 1, midRow));

  cv::Mat psf(size, CV_32FC1, cv::Scalar::all(0));

  // Swap quadrants
  q0.copyTo(psf(cv::Rect(0, 0, midCol, midRow)));
  q1.copyTo(psf(cv::Rect(midCol, midRow, endCol + 1, endRow + 1)));
  q2.copyTo(psf(cv::Rect(0, midRow, midCol, endRow + 1)));
  q3.copyTo(psf(cv::Rect(midCol, 0, endCol + 1, midRow)));

  return psf;
}

cv::Mat expandKernel(const cv::Mat& psf, const cv::Size& size) {
  // Get the size of the matrix.
  int rows = psf.rows;
  int cols = psf.cols;

  // Split the input matrix into quadrants.
  // Top-left to bottom-right and top-right to bottom-left
  int midRow = rows / 2;
  int midCol = cols / 2;

  cv::Mat padded(size, psf.type(), cv::Scalar::all(0));

  const int endCol = cols % 2 == 0 ? midCol : midCol + 1;
  const int endRow = rows % 2 == 0 ? midRow : midRow + 1;

  // Top-left quadrant
  cv::Mat q0(psf, cv::Rect(0, 0, midCol, midRow));
  // Bottom-right quadrant
  cv::Mat q1(psf, cv::Rect(midCol, midRow, endCol, endRow));
  // Bottom-left quadrant
  cv::Mat q2(psf, cv::Rect(0, midRow, midCol, endRow));
  // Top-right quadrant
  cv::Mat q3(psf, cv::Rect(midCol, 0, endCol, midRow));

  // Swap quadrants
  q0.copyTo(padded(
      cv::Rect(padded.cols - midCol, padded.rows - midRow, midCol, midRow)));
  q1.copyTo(padded(cv::Rect(0, 0, endCol, endRow)));
  q2.copyTo(padded(cv::Rect(padded.cols - midCol, 0, midCol, endRow)));
  q3.copyTo(padded(cv::Rect(0, padded.rows - midRow, endCol, midRow)));

  return padded;
}

cv::Mat psf2otf(const cv::Mat& psf, const cv::Size& size) {
  // Get the size of the matrix.
  int rows = psf.rows;
  int cols = psf.cols;

  // Split the input matrix into quadrants.
  // Top-left to bottom-right and top-right to bottom-left
  int midRow = rows / 2;
  int midCol = cols / 2;

  cv::Mat padded(size, psf.type(), cv::Scalar::all(0));

  const int endCol = cols % 2 == 0 ? midCol : midCol + 1;
  const int endRow = rows % 2 == 0 ? midRow : midRow + 1;

  // Top-left quadrant
  cv::Mat q0(psf, cv::Rect(0, 0, midCol, midRow));
  // Bottom-right quadrant
  cv::Mat q1(psf, cv::Rect(midCol, midRow, endCol, endRow));
  // Bottom-left quadrant
  cv::Mat q2(psf, cv::Rect(0, midRow, midCol, endRow));
  // Top-right quadrant
  cv::Mat q3(psf, cv::Rect(midCol, 0, endCol, midRow));

  // Swap quadrants
  q0.copyTo(padded(
      cv::Rect(padded.cols - midCol, padded.rows - midRow, midCol, midRow)));
  q1.copyTo(padded(cv::Rect(0, 0, endCol, endRow)));
  q2.copyTo(padded(cv::Rect(padded.cols - midCol, 0, midCol, endRow)));
  q3.copyTo(padded(cv::Rect(0, padded.rows - midRow, endCol, midRow)));

  cv::Mat otf;
  cv::dft(padded, otf, cv::DFT_COMPLEX_OUTPUT);
  return otf;
}

// Convolve image with kernel
Mat convolve(const Mat& img, const Mat& kernel) {
  Mat result;

  cv::Mat otf = psf2otf(kernel, img.size());
  cv::Mat imgDFT;
  cv::dft(img, imgDFT, cv::DFT_COMPLEX_OUTPUT);
  cv::Mat resultDFT;
  cv::mulSpectrums(imgDFT, otf, resultDFT, 0);
  cv::dft(resultDFT, result,
          cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

  return result;
}

// Flip kernel (180 degrees)
Mat flipKernel(const Mat& kernel) {
  Mat flipped;
  flip(kernel, flipped, -1);
  return flipped;
}

// Richardson-Lucy update for image
Mat richardsonLucyImageStep(const Mat& blurred, const Mat& latent,
                            const Mat& psf) {
  Mat estBlurred = convolve(latent, psf);
  Mat ratio;
  divide(blurred, estBlurred + 1e-8, ratio);
  Mat update = convolve(ratio, flipKernel(psf));
  Mat nextLatent = latent.mul(update);
  return nextLatent;
}

// Richardson-Lucy update for kernel (naive version)
Mat richardsonLucyKernelStep(const Mat& blurred, const Mat& latent,
                             const Mat& psf) {
  Mat estBlurred = convolve(latent, psf);
  Mat ratio;
  divide(blurred, estBlurred + 1e-8, ratio);

  Mat update = convolve(flipKernel(latent), ratio);

  Mat expandedKernel = expandKernel(psf, update.size());

  Mat nextExpandedKernel = expandedKernel.mul(update);
  Mat nextKernel = shrinkKernel(nextExpandedKernel, psf.size());
  normalizeKernel(nextKernel);
  return nextKernel;
}

int main() {
  // Load image
  Mat img = imread("lena.png", IMREAD_GRAYSCALE);
  if (img.empty()) {
    cerr << "Failed to load image\n";
    return -1;
  }

  img.convertTo(img, CV_32F, 1.0 / 255.0);

  // Create blurred image with true kernel
  Mat trueKernel;
  Mat blurred = addBlur(img, trueKernel, 21, 3.0);

  // Add small Gaussian noise
  Mat noise = Mat::zeros(blurred.size(), blurred.type());
  randn(noise, 0, 0.005);
  blurred += noise;

  // Initial guesses
  Mat latent = blurred.clone();  // Initial latent estimate

  // Mat psf = Mat::ones(23, 23, CV_32F); // Initial kernel estimate

  // Mat psf = Mat::zeros(23, 23, CV_32F); // Initial kernel estimate
  // circle(psf, Point(11, 11), 6, Scalar(1), -1);

  // normalizeKernel(psf);

  Mat psf;  // Initial kernel estimate
  addBlur(img, psf, 21, 4.0);

  imshow("Original", img);
  imshow("Blurred", blurred);

  double maxTrueKernelVal;
  cv::minMaxLoc(psf, nullptr, &maxTrueKernelVal);
  imshow("True Kernel", trueKernel / maxTrueKernelVal);

  double maxKernelVal;
  cv::minMaxLoc(psf, nullptr, &maxKernelVal);

  // Blind Richardson-Lucy loop
  int iterations = 300;
  for (int i = 0; i < iterations; ++i) {
    latent = richardsonLucyImageStep(blurred, latent, psf);
    for (int j = 0; j < 30; ++j) {
      psf = richardsonLucyKernelStep(blurred, latent, psf);
    }

    cv::minMaxLoc(psf, nullptr, &maxKernelVal);

    // Optional: visualize progress
    if (i % 10 == 0) {
      cout << "Iteration " << i << endl;
    }

    imshow("Current Kernel", psf / maxKernelVal);
    imshow("Current latent", latent);
    waitKey(10);
  }

  // Show results
  imshow("Restored (Latent)", latent);
  imshow("Estimated Kernel", psf / maxKernelVal);

  waitKey(0);
  return 0;
}
