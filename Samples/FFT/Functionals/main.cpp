/*
 * This code performs Fourier defocusing deblurring
 * Tiknhonov method is used
 * Enhanced rough noise estimation is used to set parameter
 * for the method
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "blurrf.hpp"
#include "fdb.hpp"
#include "noise.hpp"

using namespace std;

// the function for LV1noise
// make it as you wish
float LV1f(float x) {
  float y;

  if (x > 0) {
    y = sqrt(x) / 10;
  } else {
    y = x;
  }

  return y;
}

/*
 * Function to calculate
 * the standard deviation
 * of the noise
 */

float NoiseDev(const cv::Mat &imga) {
  cv::Mat imgd = cv::Mat(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  imgd = imga / 255.0;
  cv::dft(imgd, imgd);
  for (int row = int(shar * imgd.rows / 2); row < (imgd.rows / 2) - 1; row++) {
    for (int col = int(shar * imgd.cols / 2); col < (imgd.cols / 2) - 1;
         col++) {
      FGet2D(imgd, col, row, &re, &im);
      sum += sqrt(re * re + im * im);
    }
  }
  sum /= ((imgd.rows / 2) - int(shar * imgd.rows / 2)) *
         ((imgd.cols / 2) - int(shar * imgd.cols / 2)) *
         sqrt((imgd.rows) * (imgd.cols));
  return (sum);
}

/*
 * Function to calculate
 * the norm of the noise
 */

float NoiseDevN(const cv::Mat &imga) {
  cv::Mat imgd = cv::Mat(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  re = 0;
  im = 0;
  imgd = imga / 255.0;
  cv::dft(imgd, imgd);
  for (int row = 0; row < int(shar * imgd.rows / 2); row++) {
    for (int col = 0; col < int(shar * imgd.cols / 2); col++) {
      FSet2D(imgd, col, row, &re, &im);
    }
  }
  cv::dft(imgd, imgd, cv::DFT_INVERSE | cv::DFT_SCALE);
  sum = cv::norm(imgd, cv::NORM_L2);
  return (sum);
}

/*
 * Function to calculate the standard deviation
 * of Fourier transform at the edges
 * which is supposed to be the error
 * of the noise estimation
 */

float NoiseDevErr(const cv::Mat &imga) {
  cv::Mat imgd(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum, sigm;

  shar = 0.8;
  sum = 0.0;
  sigm = NoiseDev(imga);
  imga.convertTo(imgd, CV_32F);
  imgd = imgd / 255.0;
  sigm *= sqrt((imgd.rows) * (imgd.cols));
  cv::dft(imgd, imgd);
  for (int row = int(shar * imgd.rows / 2); row < (imgd.rows / 2) - 1; row++) {
    for (int col = int(shar * imgd.cols / 2); col < (imgd.cols / 2) - 1;
         col++) {
      FGet2D(imgd, col, row, &re, &im);
      sum +=
          (sqrt(re * re + im * im) - sigm) * (sqrt(re * re + im * im) - sigm);
    }
  }
  sum = sqrt(sum);
  sum /= ((imgd.rows / 2) - int(shar * imgd.rows / 2)) *
         ((imgd.cols / 2) - int(shar * imgd.cols / 2)) *
         sqrt((imgd.rows) * (imgd.cols));
  return (sum);
}

/* Calculates the norm
 * of a Fourier image
 */

float FNorm(const cv::Mat &imgw) {
  float a1, a2, sum = 0, tmp;  // temporal variables for matrix inversion
  int w, h, h2, w2;            // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  tmp = (((float *)(imgw.data))[0]) * (((float *)(imgw.data))[0]);
  sum += tmp;

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    tmp = (a1) * (a1) + a2 * a2;
    sum += 2 * tmp;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[0]) *
          (((float *)(imgw.data + (h - 1) * imgw.step))[0]);
    sum += tmp;
  }

  if (w % 2 == 0) {
    // sets upper right
    tmp = (1.0 * ((float *)(imgw.data))[w - 1]) *
          (1.0 * ((float *)(imgw.data))[w - 1]);
    sum += tmp;

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      tmp = (a1) * (a1) + a2 * a2;
      sum += 2 * tmp;
    }

    // sets down right
    if (h % 2 == 0) {
      tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]) *
            (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]);
      sum += tmp;
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      tmp = (a1) * (a1) + a2 * a2;
      sum += 2 * tmp;
    }
  }

  return (sum);
}

/*
 * Function to create the norm
 * of the filtered image
 * for Tikhonov regularization
 */

float xfilt2_tikh(cv::Mat &imga1, cv::Mat &imgb1, float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();
  float res;
  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);

  // inverts Fourier transformed blurred
  FMatInv(imgb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgb, 0);

  res = FNorm(imgb);

  return (res);
}

/* Calculates the 1-Phi
 * diagonal for blurr matrix
 */

float F1f(const cv::Mat &imgb, const cv::Mat &imgw, float gamma) {
  float a1, a2, sum = 0, tmp, tmp1;  // temporal variables for matrix inversion
  int w, h, h2, w2;                  // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  tmp = (((float *)(imgw.data))[0]) * (((float *)(imgw.data))[0]);
  tmp = gamma / (tmp + gamma);
  tmp1 = (((float *)(imgb.data))[0]) * (((float *)(imgb.data))[0]);
  sum += tmp * tmp * tmp1;

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    tmp = (a1) * (a1) + a2 * a2;
    tmp = gamma / (tmp + gamma);
    a1 = ((float *)(imgb.data + row * imgb.step))[0];
    a2 = ((float *)(imgb.data + (row + 1) * imgb.step))[0];
    tmp1 = (a1) * (a1) + a2 * a2;
    sum += tmp1 * tmp * tmp;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[0]) *
          (((float *)(imgw.data + (h - 1) * imgw.step))[0]);
    tmp = gamma / (tmp + gamma);
    tmp1 = (((float *)(imgb.data + (h - 1) * imgb.step))[0]) *
           (((float *)(imgb.data + (h - 1) * imgb.step))[0]);
    sum += tmp1 * tmp * tmp;
  }

  if (w % 2 == 0) {
    // sets upper right
    tmp = (((float *)(imgw.data))[w - 1]) * (((float *)(imgw.data))[w - 1]);
    tmp = gamma / (tmp + gamma);
    tmp1 = (((float *)(imgb.data))[w - 1]) * (((float *)(imgb.data))[w - 1]);
    sum += tmp1 * tmp * tmp;

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      tmp = (a1) * (a1) + a2 * a2;
      tmp = gamma / (tmp + gamma);
      a1 = ((float *)(imgb.data + row * imgb.step))[w - 1];
      a2 = ((float *)(imgb.data + (row + 1) * imgb.step))[w - 1];
      tmp1 = (a1) * (a1) + a2 * a2;
      sum += tmp1 * tmp * tmp;
    }

    // sets down right
    if (h % 2 == 0) {
      tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]) *
            (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]);
      tmp = gamma / (tmp + gamma);
      tmp1 = (((float *)(imgb.data + (h - 1) * imgb.step))[w - 1]) *
             (((float *)(imgb.data + (h - 1) * imgb.step))[w - 1]);
      sum += tmp1 * tmp * tmp;
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      tmp = (a1) * (a1) + a2 * a2;
      tmp = gamma / (tmp + gamma);
      a1 = ((float *)(imgb.data + row * imgb.step))[col];
      a2 = ((float *)(imgb.data + row * imgb.step))[col + 1];
      tmp1 = (a1) * (a1) + a2 * a2;
      sum += tmp1 * tmp * tmp;
    }
  }

  return (sum);
}

/* Calculates the 1-Phi
 * diagonal for blurr matrix
 * and returns it
 */

void IFMatInv(cv::Mat &imgw, float gamma) {
  float a1, a2, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;   // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] = 1.0 - 1.0 / (((float *)(imgw.data))[0] + gamma);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    sum = a1 * a1 + a2 * a2 + gamma;
    ((float *)(imgw.data + row * imgw.step))[0] = 1.0 - a1 / sum;
    ((float *)(imgw.data + (row + 1) * imgw.step))[0] = a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        1.0 - 1.0 / (((float *)(imgw.data + (h - 1) * imgw.step))[0] + gamma);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float *)(imgw.data))[w - 1] =
        1.0 - 1.0 / (((float *)(imgw.data))[w - 1] + gamma);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float *)(imgw.data + row * imgw.step))[w - 1] = 1.0 - a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          1.0 -
          1.0 / (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] + gamma);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float *)(imgw.data + row * imgw.step))[col] = 1.0 - a1 / sum;
      ((float *)(imgw.data + row * imgw.step))[col + 1] = (a2 / sum);
    }
  }
}

/*
 * Program to calculate
 * the Ax-b norm
 */

float Axbfilt2_tikh(cv::Mat &imga1, cv::Mat &imgb1, float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();
  float res;
  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);

  IFMatInv(imgb, gamma);
  cv::mulSpectrums(imga, imgb, imgb, 0);

  res = cv::norm(imgb, cv::NORM_L2);

  return (res);
}

/* Calculates the trace of the
 * diagonal transform matrix
 * for the GCV functional
 */

float FTr(const cv::Mat &imgw, float gamma) {
  float a1, a2, sum = 0, tmp;  // temporal variables for matrix inversion
  int w, h, h2, w2;            // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  tmp = (((float *)(imgw.data))[0]) * (((float *)(imgw.data))[0]);
  sum += tmp / (tmp + gamma);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    tmp = (a1) * (a1) + a2 * a2;
    sum += 2 * tmp / (tmp + gamma);
  }

  // sets down left if needed
  if (h % 2 == 0) {
    tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[0]) *
          (((float *)(imgw.data + (h - 1) * imgw.step))[0]);
    sum += tmp / (tmp + gamma);
  }

  if (w % 2 == 0) {
    // sets upper right
    tmp = (((float *)(imgw.data))[w - 1]) * (((float *)(imgw.data))[w - 1]);
    sum += tmp / (tmp + gamma);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      tmp = (a1) * (a1) + a2 * a2;
      sum += 2 * tmp / (tmp + gamma);
    }

    // sets down right
    if (h % 2 == 0) {
      tmp = (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]) *
            (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1]);
      sum += tmp / (tmp + gamma);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      tmp = (a1) * (a1) + a2 * a2;
      sum += 2 * tmp / (tmp + gamma);
    }
  }

  return (sum);
}

/*
 * Total variation calculation
 */

float TotVar(cv::Mat &imga) {
  float s1, s2, s3 = 0;
  int x1, x2, y1, y2;

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      if (row == (imga.rows - 1)) {
        y1 = row;

      } else {
        y1 = row + 1;
      }

      if (col == (imga.cols - 1)) {
        x2 = col;
      } else {
        x2 = col + 1;
      }

      if (row == 0) {
        y2 = 0;

      } else {
        y2 = row - 1;
      }

      if (col == 0) {
        x1 = col;
      } else {
        x1 = col - 1;
      }

      s1 = ((float *)(imga.data + row * imga.step))[x2] -
           ((float *)(imga.data + row * imga.step))[x1];
      s2 = ((float *)(imga.data + y2 * imga.step))[col] -
           ((float *)(imga.data + y1 * imga.step))[col];
      s3 += sqrt(s1 * s1 + s2 * s2);
    }
  }
  return (s3);
}

// GCV functional

float GCV(cv::Mat &imgb1, cv::Mat &imgw1, cv::Mat &imgw2, float alpha) {
  cv::Mat imgl1 = imgb1.clone();
  float tr1, resul;

  // calculates the trace
  cv::dft(imgb1, imgl1);
  tr1 = (imgb1.cols) * (imgb1.rows) - FTr(imgl1, alpha);

  // creates test image for the norm
  FCFilter(imgw1, imgb1, imgl1);
  cv::subtract(imgl1, imgw2, imgl1);

  resul = cv::norm(imgl1, cv::NORM_L2) / (tr1 * tr1);

  return (resul);
}

int main(int argc, char **argv) {
  cv::Mat imgi;

  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
  } else {
    return 1;
  }

  cv::Mat img2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_8UC1);

  int r = 6, kw, kh;  // radius of blurring, kernel width and height
  float alpha;        // this is alpha for Tikhonov regularization
  cv::Mat kernel;     // kernel for blurring
  cv::Mat img;
  // blurr and deblurred image
  cv::Mat img1, imgb;
  // images for the initial DFT and DCT transformations
  cv::Mat imgDFTi;
  // images for the DFT and DCT transformations
  cv::Mat imgDFT;

  cv::Mat tmp;  // for border removal
  cv::Scalar ksum;

  // Norm image declaration
  cv::Mat imgl;
  float tr;

  // cut off variables
  float re, im, stdev, step, f1, f2, al1, al2, err;

  // iteration starts here
  stdev = 0;
  cv::namedWindow("test", 0);
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);

  // declaring initial images
  kernel = cv::Mat(cv::Size(2 * r + 1, 2 * r + 1), CV_32FC1);
  img = cv::Mat(
      cv::Size(imgi.cols + 2 * kernel.cols, imgi.rows + 2 * kernel.rows),
      imgi.depth(), 1);
  img1 = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1);
  imgb = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  imgDFTi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  imgDFT = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  imgl = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);

  // create border
  // standard method gives rise to ringing!!!
  // cv::copyMakeBorder(imgi,img,cvPoint(kernel.cols,kernel.rows),
  // cv::BORDER_REPLICATE);
  MirrorBorder(imgi, img, kernel.cols, kernel.cols);
  EdgeTaper(img, kernel.cols, kernel.cols);

  // defocusing
  kset(kernel, psfDefoc, r);

  // special created defocusing
  // ksetDefocus(kernel,r);

  // creates smearing image
  Blurrset(imgb, kernel);

  img.convertTo(imgDFTi, CV_32F, 1.0 / 255.0);

  // add some salt and pepper noise
  // SPnoise(imgDFTi, stdev);

  // add some gauss noise
  // Gnoise(imgDFTi, 0.0, stdev);

  // add some local variant gauss noise
  // LVnoise(imgDFTi, imgDFTi);

  // add some local variant gauss noise another version
  // LV1noise(imgDFTi, LV1f);

  // add some poisson noise
  // Pnoise(imgDFTi, stdev*500);  // sensitive!!!

  // add some speckle noise
  // SPEnoise(imgDFTi, stdev);

  // add some uniform noise
  // Unoise(imgDFTi, 1.00, stdev);

  // cout << (imgb.rows)*(imgb.cols)*(NoiseDev(imgDFTi)) << '\n';

  FDFilter(imgDFTi, imgb, imgDFT);
  // cv::imshow("Initial Image",imgDFTi);
  // cv::imshow("test",imgDFT);

  al1 = 1e-4;
  step = 1e-4;
  al2 = al1 + step;

  // GCV regularization
  do {
    // Tikhonov regularization
    FDFilter(imgDFTi, imgb, imgDFT, al2 * al2);
    f2 = GCV(imgb, imgDFT, imgDFTi, al2 * al2);
    FDFilter(imgDFTi, imgb, imgDFT, al1 * al1);
    f1 = GCV(imgb, imgDFT, imgDFTi, al1 * al1);

    step = f2 * (al2 - al1) / (f2 - f1);
    al1 = al2;
    al2 = al2 - step;

    cv::imshow("Initial Image", imgDFTi);
    cv::imshow("test", imgDFT);

    // cout  << "   stdev=" << stdev << "   stdev_m=" << alpha  << "  +/-" <<
    // NoiseDevErr(imgi) <<'\n'; cout << "   ||xfilt||2= " << cv::norm( imgDFT,
    // 0, cv::NORM_L2 ) << " ||A.xfilt-b||2= " << cv::norm( imgl, 0, cv::NORM_L2
    // ); cout << "
    // ||xfilt||2calc= " << xfilt2_tikh(imgDFT,imgb, stdev*stdev) << "
    // ||A.xfilt-b||2calc= " << Axbfilt2_tikh(imgDFT,imgb, stdev*stdev) <<'\n';
    // cout << stdev << " ||A.xfilt-b||2= " << cv::norm( imgl, 0, cv::NORM_L2
    // )<< " TV= " << TotVar(imgDFT)<< " " <<'\n'; cout << stdev << "
    // ||A.xfilt-b||2= " << Axbfilt2_tikh(imgDFT,imgb, stdev*stdev)<< "   TV= "
    // << TotVar(imgDFT)<<'\n';

    // TV
    // cout << stdev << "   " << cv::norm( imgl, 0, cv::NORM_L2 ) -
    // stdev*stdev*TotVar(imgDFT)<< " " <<'\n';

    // GCV - seems to work best
    // cout << stdev << "   " << cv::norm( imgl, 0, cv::NORM_L2 )/(tr*tr)<< " "
    // <<'\n'; cout << al2 << "   " << step <<'\n';

    // cout << al2 << "  " << f2-f1 << "   " <<  '\n';

    char c = cv::waitKey(10);
    // if(c == 27) stdev=1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    cout << '-';

  } while (abs(f2 - f1) > 1e-11);

  cout << '\n' << "GCV:  " << al1 * al1 << '\n';

  // cout  << TotVar(imgDFT) << '\n';

  al1 = 1e-4;
  step = 1e-4;
  al2 = al1 + step;

  // TV regularization
  do {
    // Tikhonov regularization
    FDFilter(imgDFTi, imgb, imgDFT, al2 * al2);
    f2 = abs(Axbfilt2_tikh(imgDFTi, imgb, al2 * al2) - 1.0 * TotVar(imgDFT));
    FDFilter(imgDFTi, imgb, imgDFT, al2 * al2);
    f1 = abs(Axbfilt2_tikh(imgDFTi, imgb, al1 * al1) - 1.0 * TotVar(imgDFT));

    step = f2 * (al2 - al1) / (f2 - f1);
    al1 = al2;
    al2 = al2 - step;

    cv::imshow("Initial Image", imgDFTi);
    cv::imshow("test", imgDFT);

    // TV
    // cout << stdev << "   " << cv::norm( imgl, 0, cv::NORM_L2 ) -
    // stdev*stdev*TotVar(imgDFT)<< " " <<'\n';

    // cout << al2 <<"  " << step << "  " << f2-f1 << "   " <<  '\n';

    char c = cv::waitKey(10);
    // if(c == 27) stdev=1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    cout << '-';

  } while (abs(step) > 8e-4);

  FDFilter(imgDFTi, imgb, imgDFT, al1 * al1);

  cout << '\n' << "TV:  " << al1 * al1 << '\n';

  err = NoiseDevN(imgDFTi) * (imgDFTi.cols) * (imgDFTi.rows);
  // cout << err << "  " << TotVar(imgDFT) << '\n';

  al1 = 1e-4;
  step = 1e-4;
  al2 = al1 + step;

  // DP regularization
  do {
    // Tikhonov regularization
    FDFilter(imgDFTi, imgb, imgDFT, al2 * al2);
    f2 = abs(Axbfilt2_tikh(imgDFTi, imgb, al2 * al2) - 1.0 * err);
    FDFilter(imgDFTi, imgb, imgDFT, al2 * al2);
    f1 = abs(Axbfilt2_tikh(imgDFTi, imgb, al1 * al1) - 1.0 * err);

    step = f2 * (al2 - al1) / (f2 - f1);
    al1 = al2;
    al2 = al2 - step;

    cv::imshow("Initial Image", imgDFTi);
    cv::imshow("test", imgDFT);

    // cout << al2 <<"  " << step << "  " << f2 << "   " <<  '\n';

    char c = cv::waitKey(10);
    // if(c == 27) stdev=1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    cout << '-';

  } while (abs(step) > 1e-9);

  FDFilter(imgDFTi, imgb, imgDFT, al1 * al1);

  cout << '\n' << "DP:  " << al1 * al1 << '\n';

  /*
  al1=1e-4;
  step=1e-5;
  al2=al1+step;

  float r1, r2, s1, s2, df;

  f1=0;

  //L curve regularization - does not work
  do
  {

  //Tikhonov regularization
  FDFilter(imgDFTi,imgb,imgDFT, al2*al2);
  r1=log(cv::norm( imgDFT, 0, cv::NORM_L2 ));
  FCFilter(imgDFT,imgb,imgl);
  cv::subtract(imgl,imgDFTi,imgl);
  s1=log(cv::norm( imgl, 0, cv::NORM_L2 ));

  FDFilter(imgDFTi,imgb,imgDFT, al1*al1);
  r2=log(cv::norm( imgDFT, 0, cv::NORM_L2 ));
  FCFilter(imgDFT,imgb,imgl);
  cv::subtract(imgl,imgDFTi,imgl);
  s2=log(cv::norm( imgl, 0, cv::NORM_L2 ));

  f2=(s2*r1-s1*r2)/pow(((s2-s1)*(s2-s1)+(r2-r1)*(r2-r1)),1.5);

step=f2*(al2-al1)/(f2-f1);
  al1=al2;
  al2=al2-step;
  df=f2-f1;
  f1=f2;

  cv::imshow("Initial Image",imgDFTi);
  cv::imshow("test",imgDFT);

  //cout  << "   stdev=" << stdev << "   stdev_m=" << alpha  << "  +/-" <<
NoiseDevErr(imgi) <<'\n';
  //cout << "   ||xfilt||2= " << cv::norm( imgDFT, 0, cv::NORM_L2 ) << "
||A.xfilt-b||2= " << cv::norm( imgl, 0, cv::NORM_L2 );
  //cout << "   ||xfilt||2calc= " << xfilt2_tikh(imgDFT,imgb, stdev*stdev) << "
||A.xfilt-b||2calc= " << Axbfilt2_tikh(imgDFT,imgb, stdev*stdev) <<'\n';
  //cout << stdev << " ||A.xfilt-b||2= " << cv::norm( imgl, 0, cv::NORM_L2 )<< "
TV= "
<< TotVar(imgDFT)<< " " <<'\n';
  //cout << stdev << " ||A.xfilt-b||2= " << Axbfilt2_tikh(imgDFT,imgb,
stdev*stdev)<< "   TV= " << TotVar(imgDFT)<<'\n';

  //TV
  //cout << stdev << "   " << cv::norm( imgl, 0, cv::NORM_L2 ) -
stdev*stdev*TotVar(imgDFT)<< " " <<'\n';


  //cout << al1 << "   " << step << "  " << df << "   " <<  '\n';

  //variables to be exported outside the function
  kw=kernel.cols;
  kh=kernel.rows;

  cout << '-' ;

  }
  while(abs(step)>1e-9);

  cout  << '\n' << "Lcurve:  " << al1*al1 <<'\n';
  */

  /*
  al1=1e-4;
  step=1e-4;
  al2=al1+step;


//General regularization - it requires some unknown parameter, replaced by 400
  do
  {

  //Tikhonov regularization
  FDFilter(imgDFTi,imgb,imgDFT, al2*al2);
  f2=abs(Axbfilt2_tikh(imgDFTi, imgb,
al2*al2)-400*cv::norm(imgDFT,0,cv::NORM_L2)); FDFilter(imgDFTi,imgb,imgDFT,
al2*al2); f1=abs(Axbfilt2_tikh(imgDFTi, imgb,
al1*al1)-400*cv::norm(imgDFT,0,cv::NORM_L2));

step=f2*(al2-al1)/(f2-f1);
  al1=al2;
  al2=al2-step;

  cv::imshow("Initial Image",imgDFTi);
  cv::imshow("test",imgDFT);

  cout << al2 <<"  " << step << "  " << f2 << "   " <<  '\n';

  char c = cv::waitKey(10);
  //if(c == 27) stdev=1.0;

  //variables to be exported outside the function
  kw=kernel.cols;
  kh=kernel.rows;

  cout << '-';

  }
  while(abs(step)>1e-9);

  FDFilter(imgDFTi,imgb,imgDFT, al1*al1);

  cout  << '\n' << "General:  " << al1*al1 <<'\n';
  */

  imgDFT.convertTo(img1, CV_8U, 255);

  // remove borders and clear the memory
  img1(cv::Rect(kw, kh, imgi.cols, imgi.rows)).copyTo(img2);

  cv::rectangle(img2, cv::Point(kw / 2, kh / 2),
                cv::Point(imgi.cols - kw / 2, imgi.rows - kh / 2),
                cv::Scalar(255));

  cv::namedWindow("Deblurred Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Deblurred Image", img2);
  cout << "\n Press 'Esc' Key to Exit \n";
  while (1) {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }
  cv::destroyWindow("Deblurred Image");
  cv::destroyWindow("Initial Image");
  cv::destroyWindow("test");

  return (0);
}