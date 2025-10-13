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

// 2D convolution that preserves size
Mat convolve2Dnaiive(const Mat& img, const Mat& kernel) {
  int rows = img.rows;
  int cols = img.cols;
  int k_rows = kernel.rows;
  int k_cols = kernel.cols;
  int offset_r = k_rows / 2;
  int offset_c = k_cols / 2;

  Mat result = Mat::zeros(rows, cols, CV_64F);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      double sum = 0.0;

      for (int ki = 0; ki < k_rows; ++ki) {
        for (int kj = 0; kj < k_cols; ++kj) {
          int img_i = i - offset_r + ki;
          int img_j = j - offset_c + kj;

          // Reflective boundary handling
          if (img_i < 0) img_i = -img_i;
          if (img_i >= rows) img_i = 2 * rows - 2 - img_i;
          if (img_j < 0) img_j = -img_j;
          if (img_j >= cols) img_j = 2 * cols - 2 - img_j;

          sum += img.at<double>(img_i, img_j) * kernel.at<double>(ki, kj);
        }
      }

      result.at<double>(i, j) = sum;
    }
  }

  return result;
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
Mat deconvLucy(const Mat& blurred, const Mat& PSF, int numit = 10) {
  Mat I = blurred.clone();
  if (I.type() != CV_64F) {
    I.convertTo(I, CV_64F);
  }

  Mat psf = PSF.clone();
  if (psf.type() != CV_64F) {
    psf.convertTo(psf, CV_64F);
  }

  int rows = I.rows;
  int cols = I.cols;

  // Normalize PSF to sum to 1
  double psfSum = sum(psf)[0];
  if (psfSum > 0) {
    psf = psf / psfSum;
  }

  // Initialize with blurred image
  Mat J = I.clone();

  // Add small constant to prevent division by zero
  double eps = 1e-6;

  // Main iterations
  for (int iter = 0; iter < numit; ++iter) {
    // Step 1: Convolve current estimate with PSF
    Mat Jconv = convolve2D(J, psf);

    // Step 2: Compute ratio I / (J convolved with PSF + eps)
    Mat ratio(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double denominator = Jconv.at<double>(i, j) + eps;
        ratio.at<double>(i, j) = I.at<double>(i, j) / denominator;
      }
    }

    // Step 3: Correlate ratio with PSF (convolution with flipped PSF)
    Mat correction = correlate2D(ratio, psf);

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
  Mat PSF = getGaussianKernel(15, 1.0, CV_64F);
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
  Mat J10 = deconvLucy(BlurredNoisy, PSF, 30);

  minMaxLoc(J10, &minVal, &maxVal);
  cout << "J10 range: [" << minVal << ", " << maxVal << "]" << endl;

  // Convert to 8-bit for display
  Mat I_disp, Blurred_disp, J10_disp;
  I.convertTo(I_disp, CV_8U, 255.0);
  BlurredNoisy.convertTo(Blurred_disp, CV_8U, 255.0);
  J10.convertTo(J10_disp, CV_8U, 255.0);

  // Display
  imshow("Original", I_disp);
  imshow("Blurred+Noisy", Blurred_disp);
  imshow("Deconvolved 100 iter", J10_disp);

  waitKey(0);

  return 0;
}
