//   "Acceleration of iterative image restoration algorithms, by D.S.C. Biggs
//   and M. Andrews, Applied Optics, Vol. 36, No. 8, 1997.
//   "Deconvolutions of Hubble Space Telescope Images and Spectra",
//   R.J. Hanisch, R.L. White, and R.L. Gilliland. in "Deconvolution of Images
//   and Spectra", Ed. P.A. Jansson, 2nd ed., Academic Press, CA, 1997.

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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
  cv::Mat q1(padded, cv::Rect(0, 0, endCol, endRow));
  // Bottom-left quadrant
  cv::Mat q2(padded, cv::Rect(padded.cols - midCol, 0, midCol, endRow));
  // Top-right quadrant
  cv::Mat q3(padded, cv::Rect(0, padded.rows - midRow, endCol, midRow));

  cv::Mat psf(size, CV_64FC1, cv::Scalar::all(0));

  // Swap quadrants
  q0.copyTo(psf(cv::Rect(0, 0, midCol, midRow)));
  q1.copyTo(psf(cv::Rect(midCol, midRow, endCol, endRow)));
  q2.copyTo(psf(cv::Rect(0, midRow, midCol, endRow)));
  q3.copyTo(psf(cv::Rect(midCol, 0, endCol, midRow)));

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

Mat convolve2D(const Mat& img, const Mat& kernel) {
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

// Correlate: convolve with flipped kernel
Mat correlate2D(const Mat& img, const Mat& kernel) {
  Mat flipped;
  flip(kernel, flipped, -1);
  return convolve2D(img, flipped);
}

// Lucy-Richardson deconvolution
Mat deconvLucy(const Mat& blurred, const Mat& original, int numit = 10) {
  Mat I = blurred.clone();
  if (I.type() != CV_64F) {
    I.convertTo(I, CV_64F);
  }

  Mat originalClone = original.clone();
  if (originalClone.type() != CV_64F) {
    originalClone.convertTo(originalClone, CV_64F);
  }

  int rows = I.rows;
  int cols = I.cols;

  // Initialize with kernel image guess
  Mat J(I.size(), CV_64F);
  cv::circle(J, Point(0, 0), 15, Scalar(1), -1);
  cv::circle(J, Point(I.cols, 0), 15, Scalar(1), -1);
  cv::circle(J, Point(I.cols, I.rows), 15, Scalar(1), -1);
  cv::circle(J, Point(0, I.rows), 15, Scalar(1), -1);
  {
    double psfSum = sum(J)[0];
    if (psfSum > 0) {
      J = J / psfSum;
    }
  }

  // Add small constant to prevent division by zero
  double eps = 1e-6;

  // Main iterations
  for (int iter = 0; iter < numit; ++iter) {
    // Step 1: Convolve current estimate with initial image
    Mat Jconv = convolve2D(J, originalClone);

    // Step 2: Compute ratio I / (J convolved with PSF + eps)
    Mat ratio(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double denominator = Jconv.at<double>(i, j) + eps;
        ratio.at<double>(i, j) = I.at<double>(i, j) / denominator;
      }
    }

    // Step 3: Correlate ratio with PSF (convolution with flipped PSF)
    Mat correction = correlate2D(ratio, originalClone);

    // Step 4: Multiply with current estimate and enforce positivity
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double val = J.at<double>(i, j) * correction.at<double>(i, j);
        J.at<double>(i, j) = max(0.0, val);
      }
    }

    // Prevent overflow
    double maxVal = 0;
    minMaxLoc(J, nullptr, &maxVal);
    if (maxVal > 1e6) {
      J = J / maxVal;
    }

    Size kernelSize{45, 45};
    Mat kernel = shrinkKernel(J, kernelSize);

    // Normalize PSF to sum to 1
    double psfSum = sum(J)[0];
    if (psfSum > 0) {
      kernel = kernel / psfSum;
    }

    expandKernel(kernel, J.size()).copyTo(J);

    cout << "Iteration " << iter + 1 << ", max value: " << maxVal << endl;
  }

  return J;
}

int main() {
  // Create test image (checkerboard pattern)
  Mat I = Mat::ones(256, 256, CV_64F) * 0.2;
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      if ((i / 16 + j / 16) % 2 == 0) {
        I.at<double>(i, j) = 0.8;
      }
    }
  }
  // Mat I = imread("Lenna.png", IMREAD_GRAYSCALE);
  // if (I.empty()) {
  //     cerr << "Failed to load image" << endl;
  //     return -1;
  // }
  // I.convertTo(I, CV_64F, 1.0 / 255.0);

  // Create Gaussian PSF (small, tight blur)
  Mat PSF = getGaussianKernel(15, 2.0, CV_64F);
  PSF = PSF * PSF.t();
  PSF = PSF / sum(PSF)[0];

  // Blur using the same method we'll use in deconvolution
  Mat Blurred = convolve2D(I, PSF);

  // Add minimal noise
  Mat noise(I.rows, I.cols, CV_64F);
  randn(noise, 0, 0.002);
  Mat BlurredNoisy = Blurred + noise;
  BlurredNoisy = max(BlurredNoisy, 0.0);
  BlurredNoisy = min(BlurredNoisy, 1.0);

  // Verify data ranges
  double minVal, maxVal;
  minMaxLoc(BlurredNoisy, &minVal, &maxVal);
  cout << "Blurred+Noisy range: [" << minVal << ", " << maxVal << "]" << endl;

  // Deconvolve
  Mat kernel = deconvLucy(BlurredNoisy, I, 100);

  minMaxLoc(kernel, &minVal, &maxVal);
  cout << "kernel range: [" << minVal << ", " << maxVal << "]" << endl;

  // Convert to 8-bit for display
  Mat I_disp, Blurred_disp;
  I.convertTo(I_disp, CV_8U, 255.0);
  BlurredNoisy.convertTo(Blurred_disp, CV_8U, 255.0);

  // Display
  imshow("Original", I_disp);
  imshow("Blurred+Noisy", Blurred_disp);
  imshow("PSF", PSF / maxVal);
  imshow("Deconvolved Kernel 100 iter",
         shrinkKernel(kernel, PSF.size()) / maxVal);

  waitKey(0);

  return 0;
}
