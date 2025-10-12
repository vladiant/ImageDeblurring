// Reference:
// Peter Kovesi, "Phase Preserving Denoising of Images".
// The Australian Pattern Recognition Society Conference: DICTA'99.
// December 1999. Perth WA. pp 212-217
// http://www.cs.uwa.edu.au/pub/robvis/papers/pk/denoise.ps.gz.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/**
 * Denoise an image using phase-preserving log-Gabor filtering
 *
 * @param image Input image (CV_32F or CV_64F)
 * @param k Number of standard deviations of noise to reject (2-3)
 * @param nscale Number of filter scales to use (5-7)
 * @param mult Multiplying factor between scales (2.5-3)
 * @param norient Number of orientations to use (6)
 * @param softness Degree of soft thresholding (0=hard, 1=soft)
 * @return Denoised image
 */
Mat noisecomp(const Mat& image, double k, int nscale, double mult, int norient,
              double softness) {
  // Convert to float if necessary
  Mat img = image.clone();
  if (img.type() != CV_32F && img.type() != CV_64F) {
    img.convertTo(img, CV_64F);
  }

  int rows = img.rows;
  int cols = img.cols;

  // Parameters
  double minWaveLength = 2.0;
  double sigmaOnf = 0.55;
  double dThetaOnSigma = 1.0;
  double epsilon = 0.00001;

  double thetaSigma = M_PI / norient / dThetaOnSigma;

  // FFT of image
  Mat imagefft_complex = img.clone();
  dft(imagefft_complex, imagefft_complex, DFT_COMPLEX_OUTPUT);

  // Extract real and imaginary parts
  vector<Mat> channels;
  split(imagefft_complex, channels);
  Mat fft_real = channels[0].clone();
  Mat fft_imag = channels[1].clone();

  // Create coordinate matrices
  Mat x = Mat::zeros(1, cols, CV_64F);
  Mat y = Mat::zeros(rows, 1, CV_64F);

  for (int i = 0; i < cols; i++) {
    x.at<double>(0, i) = ((double)i - cols / 2.0) / (cols / 2.0);
  }
  for (int i = 0; i < rows; i++) {
    y.at<double>(i, 0) = ((double)i - rows / 2.0) / (rows / 2.0);
  }

  // Expand to full grid
  Mat X = repeat(x, rows, 1);
  Mat Y = repeat(y, 1, cols);

  // Calculate radius and theta
  Mat radius = Mat::zeros(rows, cols, CV_64F);
  Mat theta = Mat::zeros(rows, cols, CV_64F);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double xx = X.at<double>(i, j);
      double yy = Y.at<double>(i, j);
      radius.at<double>(i, j) = sqrt(xx * xx + yy * yy);
      theta.at<double>(i, j) = atan2(-yy, xx);
    }
  }

  // Handle zero radius
  if (radius.at<double>(rows / 2, cols / 2) == 0.0) {
    radius.at<double>(rows / 2, cols / 2) = 1.0;
  }

  vector<double> estMeanEn;
  vector<double> sig;
  Mat totalEnergy = Mat::zeros(rows, cols, CV_64F);

  // Process each orientation
  for (int o = 0; o < norient; o++) {
    cout << "Processing orientation " << (o + 1) << endl;

    double angl = o * M_PI / norient;
    double wavelength = minWaveLength;

    // Pre-compute angular filter component
    Mat spread = Mat::zeros(rows, cols, CV_64F);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        double ds = sin(theta.at<double>(i, j)) * cos(angl) -
                    cos(theta.at<double>(i, j)) * sin(angl);
        double dc = cos(theta.at<double>(i, j)) * cos(angl) +
                    sin(theta.at<double>(i, j)) * sin(angl);
        double dtheta = abs(atan2(ds, dc));
        spread.at<double>(i, j) =
            exp(-(dtheta * dtheta) / (2 * thetaSigma * thetaSigma));
      }
    }

    // Variables to store noise statistics from first scale
    double RayMean = 0.0;
    double RayVar = 0.0;

    // Process each scale
    for (int s = 0; s < nscale; s++) {
      double fo = 1.0 / wavelength;
      double rfo = fo / 0.5;
      double sigmaLog = log(sigmaOnf);

      // Construct radial filter component (log-Gabor)
      Mat logGabor = Mat::zeros(rows, cols, CV_64F);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          double r = radius.at<double>(i, j);
          if (r > epsilon) {
            double logR = log(r / rfo);
            logGabor.at<double>(i, j) =
                exp(-(logR * logR) / (2 * sigmaLog * sigmaLog));
          }
        }
      }
      logGabor.at<double>(rows / 2, cols / 2) = 0.0;

      // Multiply by angular spread
      Mat filter = logGabor.mul(spread);

      // Apply fftshift
      Mat shiftedFilter = Mat::zeros(rows, cols, CV_64F);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int ii = (i + rows / 2) % rows;
          int jj = (j + cols / 2) % cols;
          shiftedFilter.at<double>(ii, jj) = filter.at<double>(i, j);
        }
      }

      // Convolve: multiply in frequency domain
      Mat EO_real = Mat::zeros(rows, cols, CV_64F);
      Mat EO_imag = Mat::zeros(rows, cols, CV_64F);

      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          double f = shiftedFilter.at<double>(i, j);
          EO_real.at<double>(i, j) = fft_real.at<double>(i, j) * f;
          EO_imag.at<double>(i, j) = fft_imag.at<double>(i, j) * f;
        }
      }

      // Inverse FFT to get complex result
      Mat EO_complex = Mat::zeros(rows, cols, CV_64F);
      merge(vector<Mat>{EO_real, EO_imag}, EO_complex);
      idft(EO_complex, EO_complex, DFT_SCALE);

      // Extract real and imaginary parts after IDFT
      split(EO_complex, channels);
      Mat EO_real_ifft = channels[0].clone();
      Mat EO_imag_ifft = channels[1].clone();

      // Get amplitude
      Mat aEO = Mat::zeros(rows, cols, CV_64F);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          double r = EO_real_ifft.at<double>(i, j);
          double im = EO_imag_ifft.at<double>(i, j);
          aEO.at<double>(i, j) = sqrt(r * r + im * im);
        }
      }

      // Update EO_real and EO_imag for later use
      EO_real = EO_real_ifft.clone();
      EO_imag = EO_imag_ifft.clone();

      // Estimate mean and variance from smallest scale
      if (s == 0) {
        vector<double> aEOFlat;
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            aEOFlat.push_back(aEO.at<double>(i, j));
          }
        }
        sort(aEOFlat.begin(), aEOFlat.end());
        double medianEn = aEOFlat[rows * cols / 2];
        double meanEn = medianEn * 0.5 * sqrt(-M_PI / log(0.5));

        RayVar = (4 - M_PI) * meanEn * meanEn / M_PI;
        RayMean = meanEn;

        estMeanEn.push_back(meanEn);
        sig.push_back(sqrt(RayVar));
      }

      // Apply soft thresholding at all scales
      double T = (RayMean + k * sqrt(RayVar)) / pow(mult, (double)s);

      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          bool validEO = aEO.at<double>(i, j) > T;
          double eo_real = EO_real.at<double>(i, j);
          double eo_imag = EO_imag.at<double>(i, j);
          double a = aEO.at<double>(i, j);

          double v_real, v_imag;
          if (validEO) {
            v_real = softness * T * eo_real / (a + epsilon);
            v_imag = softness * T * eo_imag / (a + epsilon);
          } else {
            v_real = eo_real;
            v_imag = eo_imag;
          }

          EO_real.at<double>(i, j) = eo_real - v_real;
          EO_imag.at<double>(i, j) = eo_imag - v_imag;
        }
      }

      // Add to total energy
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          totalEnergy.at<double>(i, j) += EO_real.at<double>(i, j);
        }
      }

      wavelength *= mult;
    }
  }

  cout << "Estimated mean noise in each orientation:" << endl;
  for (size_t i = 0; i < estMeanEn.size(); i++) {
    cout << "Orientation " << i << ": " << estMeanEn[i] << endl;
  }

  // Check total energy range
  double minEnergy, maxEnergy;
  minMaxLoc(totalEnergy, &minEnergy, &maxEnergy);
  cout << "Total energy range: [" << minEnergy << ", " << maxEnergy << "]"
       << endl;

  return totalEnergy;
}

// Example usage
int main(int argc, char** argv) {
  Mat image;

  // If image file provided as argument, load it; otherwise create synthetic
  // image
  if (argc > 1) {
    image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
      cerr << "Error loading image: " << argv[1] << endl;
      return -1;
    }
    cout << "Loaded image: " << image.rows << "x" << image.cols << endl;
  } else {
    // Create a synthetic test image with a pattern
    cout << "Creating synthetic test image..." << endl;
    image = Mat(256, 256, CV_8U);

    // Draw some patterns
    circle(image, Point(64, 64), 30, Scalar(200), -1);
    rectangle(image, Point(140, 40), Point(220, 120), Scalar(150), -1);
    line(image, Point(50, 200), Point(200, 200), Scalar(180), 5);

    // Draw concentric circles
    for (int r = 10; r < 50; r += 10) {
      circle(image, Point(200, 180), r, Scalar(100 + r * 2), 2);
    }
  }

  // Convert to double
  Mat imageDouble;
  image.convertTo(imageDouble, CV_64F);

  // Add Gaussian noise to the image
  Mat noise = Mat::zeros(imageDouble.size(), CV_64F);
  randn(noise, 0, 15);  // mean=0, stddev=15
  Mat noisyImage = imageDouble + noise;

  // Clip values to valid range
  for (int i = 0; i < noisyImage.rows; i++) {
    for (int j = 0; j < noisyImage.cols; j++) {
      double val = noisyImage.at<double>(i, j);
      noisyImage.at<double>(i, j) = max(0.0, min(255.0, val));
    }
  }

  cout << "Image size: " << image.rows << "x" << image.cols << endl;
  cout << "Noisy image - min: " << noisyImage.at<double>(0, 0)
       << ", max: " << noisyImage.at<double>(0, 0) << endl;

  // Get actual min/max
  double minVal, maxVal;
  minMaxLoc(noisyImage, &minVal, &maxVal);
  cout << "Noisy image range: [" << minVal << ", " << maxVal << "]" << endl;

  cout << "Applying denoising filter..." << endl;

  // Start timing
  auto start = chrono::high_resolution_clock::now();

  // Apply denoising with parameters
  // k=2.0, nscale=6, mult=2.5, norient=6, softness=0.5
  Mat cleanimage = noisecomp(noisyImage, 2.0, 6, 2.5, 6, 0.5);

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  cout << "Denoising completed in " << duration.count() << " ms" << endl;

  // Convert results to 8-bit for display and saving
  Mat cleanNormalized, noisyNormalized;
  normalize(cleanimage, cleanNormalized, 0, 255, NORM_MINMAX);
  normalize(noisyImage, noisyNormalized, 0, 255, NORM_MINMAX);

  cleanNormalized.convertTo(cleanNormalized, CV_8U);
  noisyNormalized.convertTo(noisyNormalized, CV_8U);
  Mat original = imageDouble.clone();
  normalize(original, original, 0, 255, NORM_MINMAX);
  original.convertTo(original, CV_8U);

  // Create comparison image (side by side)
  Mat comparison(noisyNormalized.rows, noisyNormalized.cols * 3, CV_8U);
  Mat roi1 = comparison(Rect(0, 0, noisyNormalized.cols, noisyNormalized.rows));
  Mat roi2 = comparison(Rect(noisyNormalized.cols, 0, noisyNormalized.cols,
                             noisyNormalized.rows));
  Mat roi3 = comparison(Rect(noisyNormalized.cols * 2, 0, noisyNormalized.cols,
                             noisyNormalized.rows));

  original.copyTo(roi1);
  noisyNormalized.copyTo(roi2);
  cleanNormalized.copyTo(roi3);

  // Save results
  imwrite("original.png", original);
  imwrite("noisy.png", noisyNormalized);
  imwrite("denoised.png", cleanNormalized);
  imwrite("comparison.png", comparison);

  cout << "Saved: original.png, noisy.png, denoised.png, comparison.png"
       << endl;

  // Display with OpenCV (optional - comment out if no display available)
  namedWindow("Original", WINDOW_NORMAL);
  namedWindow("Noisy", WINDOW_NORMAL);
  namedWindow("Denoised", WINDOW_NORMAL);

  imshow("Original", original);
  imshow("Noisy", noisyNormalized);
  imshow("Denoised", cleanNormalized);

  cout << "Press any key to close windows..." << endl;
  waitKey(0);
  destroyAllWindows();

  return 0;
}
