/*
 * This code performs Fourier motion deblurring
 * GCV functional is used
 * with the following norms:
 * |x|, |grad x|, |Lapl x|
 * |Hlapl x|
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

void Ximgset(cv::Mat &imga)  // generates the image
{
  imga = 0;
  // rectangle coordinates
  int i1 = round((imga.rows) / 5), i2 = round(3 * (imga.cols) / 5),
      j1 = round((imga.rows) / 5), j2 = round(3 * (imga.cols) / 5);

  // circle radius
  int r = round(max(imga.rows, imga.cols) / 5);

  // draws rectangle
  cv::rectangle(imga, cv::Point(i1, j1), cv::Point(i2, j2), cv::Scalar(0.5),
                -1);

  // draws circle
  cv::circle(imga,
             cv::Point(round(5 * (imga.cols) / 8), round(3 * (imga.rows) / 5)),
             r, cv::Scalar(1.0), -1);
}

/*
 * Total variation operator
 */

void TotVar(cv::Mat &imga, cv::Mat &imgb) {
  float s1, s2;
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
      ((float *)(imgb.data + row * imgb.step))[col] = sqrt(s1 * s1 + s2 * s2);
    }
  }
}

/*
 * Gradient operator
 */

void GradNorm(cv::Mat &imga, cv::Mat &imgb) {
  float s1, s2;
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
      ((float *)(imgb.data + row * imgb.step))[col] = abs(s1) + abs(s2);
    }
  }
}

/*
 * Hyper Laplacian operator
 */

void HLNorm(cv::Mat &imga, cv::Mat &imgb) {
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
      ((float *)(imgb.data + row * imgb.step))[col] =
          sqrt(pow(abs(s1), 0.8) + pow(abs(s2), 0.8));
    }
  }
}

/*
 * This function is used to calculate the
 * GCV functional based on three
 * Fourier transformed images:
 * imgh - image for blurr operator
 * imgq - image for regularization operator
 * imgg - blurred image
 */
float GCV(float lambda, cv::Mat &imgg, cv::Mat &imgh, cv::Mat &imgq) {
  float a1, a2, a3, s1 = 0, s2 = 0,
                    sum = 0;  // temporal variables for matrix inversion
  int w, h, h2, w2;           // image width and height, help variables

  w = imgg.cols;
  h = imgg.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // proceeds upper left
  a1 = (((float *)(imgq.data))[0]) * (((float *)(imgq.data))[0]);
  a2 = (((float *)(imgh.data))[0]) * (((float *)(imgh.data))[0]);
  a3 = (((float *)(imgg.data))[0]) * (((float *)(imgg.data))[0]);
  s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
  s2 += a1 / (a2 + lambda * a1);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = (((float *)(imgq.data + row * imgq.step))[0]) *
             (((float *)(imgq.data + row * imgq.step))[0]) +
         (((float *)(imgq.data + (row + 1) * imgq.step))[0]) *
             (((float *)(imgq.data + (row + 1) * imgq.step))[0]);
    a2 = (((float *)(imgh.data + row * imgh.step))[0]) *
             (((float *)(imgh.data + row * imgh.step))[0]) +
         (((float *)(imgh.data + (row + 1) * imgh.step))[0]) *
             (((float *)(imgh.data + (row + 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + row * imgg.step))[0]) *
             (((float *)(imgg.data + row * imgg.step))[0]) +
         (((float *)(imgg.data + (row + 1) * imgg.step))[0]) *
             (((float *)(imgg.data + (row + 1) * imgg.step))[0]);
    s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
    s2 += a1 / (a2 + lambda * a1);
  }

  // sets down left if needed
  if (h % 2 == 0) {
    a1 = (((float *)(imgq.data + (h - 1) * imgq.step))[0]) *
         (((float *)(imgq.data + (h - 1) * imgq.step))[0]);
    a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[0]) *
         (((float *)(imgh.data + (h - 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[0]) *
         (((float *)(imgg.data + (h - 1) * imgg.step))[0]);
    s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
    s2 += a1 / (a2 + lambda * a1);
  }

  if (w % 2 == 0) {
    // sets upper right
    a1 = (((float *)(imgq.data))[w - 1]) * (((float *)(imgq.data))[w - 1]);
    a2 = (((float *)(imgh.data))[w - 1]) * (((float *)(imgh.data))[w - 1]);
    a3 = (((float *)(imgg.data))[w - 1]) * (((float *)(imgg.data))[w - 1]);
    s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
    s2 += a1 / (a2 + lambda * a1);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = (((float *)(imgq.data + row * imgq.step))[w - 1]) *
               (((float *)(imgq.data + row * imgq.step))[w - 1]) +
           (((float *)(imgq.data + (row + 1) * imgq.step))[w - 1]) *
               (((float *)(imgq.data + (row + 1) * imgq.step))[w - 1]);
      a2 = (((float *)(imgh.data + row * imgh.step))[w - 1]) *
               (((float *)(imgh.data + row * imgh.step))[w - 1]) +
           (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]) *
               (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[w - 1]) *
               (((float *)(imgg.data + row * imgg.step))[w - 1]) +
           (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]) *
               (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]);
      s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
      s2 += a1 / (a2 + lambda * a1);
    }

    // sets down right
    if (h % 2 == 0) {
      a1 = (((float *)(imgq.data + (h - 1) * imgq.step))[w - 1]) *
           (((float *)(imgq.data + (h - 1) * imgq.step))[w - 1]);
      a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]) *
           (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]) *
           (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]);
      s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
      s2 += a1 / (a2 + lambda * a1);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = (((float *)(imgq.data + row * imgq.step))[col]) *
               (((float *)(imgq.data + row * imgq.step))[col]) +
           (((float *)(imgq.data + row * imgq.step))[col + 1]) *
               (((float *)(imgq.data + row * imgq.step))[col + 1]);
      a2 = (((float *)(imgh.data + row * imgh.step))[col]) *
               (((float *)(imgh.data + row * imgh.step))[col]) +
           (((float *)(imgh.data + row * imgh.step))[col + 1]) *
               (((float *)(imgh.data + row * imgh.step))[col + 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[col]) *
               (((float *)(imgg.data + row * imgg.step))[col]) +
           (((float *)(imgg.data + row * imgg.step))[col + 1]) *
               (((float *)(imgg.data + row * imgg.step))[col + 1]);
      s1 += a1 * a1 * a3 / ((a2 + lambda * a1) * (a2 + lambda * a1));
      s2 += a1 / (a2 + lambda * a1);
    }
  }
  sum = s1 / (s2 * s2);
  return (sum);
}

/*
 * This function is used to calculate the
 * GCV functional based on two
 * Fourier transformed images:
 * imgh - image for blurr operator
 * imgg - blurred image
 * Regularization operator is the norm
 * of the image
 */
float GCV(float lambda, cv::Mat &imgg, cv::Mat &imgh) {
  float a2, a3, s1 = 0, s2 = 0,
                sum = 0;  // temporal variables for matrix inversion
  int w, h, h2, w2;       // image width and height, help variables

  w = imgg.cols;
  h = imgg.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // proceeds upper left
  a2 = (((float *)(imgh.data))[0]) * (((float *)(imgh.data))[0]);
  a3 = (((float *)(imgg.data))[0]) * (((float *)(imgg.data))[0]);
  s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
  s2 += 1.0 / (a2 + lambda);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a2 = (((float *)(imgh.data + row * imgh.step))[0]) *
             (((float *)(imgh.data + row * imgh.step))[0]) +
         (((float *)(imgh.data + (row + 1) * imgh.step))[0]) *
             (((float *)(imgh.data + (row + 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + row * imgg.step))[0]) *
             (((float *)(imgg.data + row * imgg.step))[0]) +
         (((float *)(imgg.data + (row + 1) * imgg.step))[0]) *
             (((float *)(imgg.data + (row + 1) * imgg.step))[0]);
    s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
    s2 += 1.0 / (a2 + lambda);
  }

  // sets down left if needed
  if (h % 2 == 0) {
    a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[0]) *
         (((float *)(imgh.data + (h - 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[0]) *
         (((float *)(imgg.data + (h - 1) * imgg.step))[0]);
    s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
    s2 += 1.0 / (a2 + lambda);
  }

  if (w % 2 == 0) {
    // sets upper right
    a2 = (((float *)(imgh.data))[w - 1]) * (((float *)(imgh.data))[w - 1]);
    a3 = (((float *)(imgg.data))[w - 1]) * (((float *)(imgg.data))[w - 1]);
    s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
    s2 += 1.0 / (a2 + lambda);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a2 = (((float *)(imgh.data + row * imgh.step))[w - 1]) *
               (((float *)(imgh.data + row * imgh.step))[w - 1]) +
           (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]) *
               (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[w - 1]) *
               (((float *)(imgg.data + row * imgg.step))[w - 1]) +
           (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]) *
               (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]);
      s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
      s2 += 1.0 / (a2 + lambda);
    }

    // sets down right
    if (h % 2 == 0) {
      a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]) *
           (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]) *
           (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]);
      s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
      s2 += 1.0 / (a2 + lambda);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a2 = (((float *)(imgh.data + row * imgh.step))[col]) *
               (((float *)(imgh.data + row * imgh.step))[col]) +
           (((float *)(imgh.data + row * imgh.step))[col + 1]) *
               (((float *)(imgh.data + row * imgh.step))[col + 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[col]) *
               (((float *)(imgg.data + row * imgg.step))[col]) +
           (((float *)(imgg.data + row * imgg.step))[col + 1]) *
               (((float *)(imgg.data + row * imgg.step))[col + 1]);
      s1 += 1.0 * a3 / ((a2 + lambda) * (a2 + lambda));
      s2 += 1.0 / (a2 + lambda);
    }
  }
  sum = s1 / (s2 * s2);
  return (sum);
}

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix,
 * regularized by a given operator
 */
void FMatInv(cv::Mat &imgw, cv::Mat &imgq, float gamma) {
  float a1, a2, a3, a4, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;           // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] =
      ((float *)(imgw.data))[0] /
      (((float *)(imgw.data))[0] * ((float *)(imgw.data))[0] +
       gamma * ((float *)(imgq.data))[0] * ((float *)(imgq.data))[0]);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    a3 = ((float *)(imgq.data + row * imgq.step))[0];
    a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[0];
    sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
    ((float *)(imgw.data + row * imgw.step))[0] = a1 / sum;
    ((float *)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float *)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float *)(imgw.data + (h - 1) * imgw.step))[0] *
             ((float *)(imgw.data + (h - 1) * imgw.step))[0] +
         gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[0] *
             ((float *)(imgq.data + (h - 1) * imgq.step))[0]);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float *)(imgw.data))[w - 1] =
        ((float *)(imgw.data))[w - 1] /
        (((float *)(imgw.data))[w - 1] * ((float *)(imgw.data))[w - 1] +
         gamma * ((float *)(imgq.data))[w - 1] * ((float *)(imgq.data))[w - 1]);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      a3 = ((float *)(imgq.data + row * imgq.step))[w - 1];
      a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[w - 1];
      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
      ((float *)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] /
          (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] *
               ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] +
           gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1] *
               ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1]);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      a3 = ((float *)(imgq.data + row * imgq.step))[col];
      a4 = ((float *)(imgq.data + row * imgq.step))[col + 1];
      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
      ((float *)(imgw.data + row * imgw.step))[col] = a1 / sum;
      ((float *)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
    }
  }
}

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * with regularization
 * Note: Periodic Border Conditions!
 */

void FDFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc, cv::Mat &imgq,
              float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);

  // inverts Fourier transformed blurred
  FMatInv(imgb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgb, 0);

  // Backward Fourier Transform
  cv::dft(imgb, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

int main(int argc, char **argv) {
  cv::Mat imgi, img0;  // initial images

  int m = 320, n = 240;   // test image dimensions (QVGA)
  float x1 = 10, y1 = 0;  // motion blurr vector
  int ksize, kw, kh;      // kernel width and height
  float alpha;            // this is alpha for Tikhonov regularization
  cv::Mat kernel;         // kernel for blurring
  cv::Mat img, imgl;
  // blurr and deblurred image
  cv::Mat img1, imgb;
  // images for the initial DFT and DCT transformations
  cv::Mat imgDFTi;
  // images for the DFT and DCT transformations
  cv::Mat imgDFT;
  // images for the DFT and DCT regularization operator
  cv::Mat imgDFTq, imgDFTqt;

  cv::Mat tmp;  // for border removal
  cv::Scalar ksum;

  // Norms declaration and solving variables
  float lambd, f1, f2, step;
  int iter;
  float lambd1, lambd2, lambd3, lambd4, lambd5;

  // regularized images
  cv::Mat img2, img3, img4, img5, img6;

  // cut off variables
  float re, im, stdev;

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
    img0 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img0, CV_32F);
    img0 = img0 / 255.0;
  } else {
    img0 = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img0);
  }

  img2 = cv::Mat(cv::Size(img0.cols, img0.rows), CV_8UC1);
  img3 = cv::Mat(cv::Size(img0.cols, img0.rows), CV_8UC1);

  // create blurr kernel
  ksize = max(abs(x1) + 1, abs(y1) + 1);
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);
  ksetMoveXY(kernel, x1, y1);

  // declare blurred image
  img = cv::Mat(
      cv::Size(img0.cols + 2 * kernel.cols, img0.rows + 2 * kernel.rows),
      img0.depth(), 1);
  // temporal image
  imgl = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  // create border
  cv::copyMakeBorder(img0, img, kernel.rows, kernel.rows, kernel.cols,
                     kernel.cols, cv::BORDER_REPLICATE);
  // supress ringing from boundaries
  EdgeTaper(img, kernel.cols, kernel.cols);

  // creates Fourier smearing image
  imgb = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  Blurrset(imgb, kernel);
  FCFilter(img, imgb, img);

  // declare the output image
  img1 = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1);

  // iteration starts here
  stdev = 0.00;
  step = 0.001;
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("|x|", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("|Grad x|", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("|Lapl x|", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("|HLapl x|", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("TV x", cv::WINDOW_AUTOSIZE);

  do {
    // declaring initial images
    imgDFTi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFT = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFTq = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFTqt = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);

    /*
    //use this in standalone deblurring
    //create border
    cv::copyMakeBorder(imgi,img,kernel.rows, kernel.rows, kernel.cols,
    kernel.cols, cv::BORDER_REPLICATE);
    //supress ringing from boundaries
    EdgeTaper(img, kernel.cols, kernel.cols);
     */

    img.copyTo(imgDFTi);

    // add some salt and pepper noise
    // SPnoise(imgDFTi, stdev);

    // add some gauss noise
    Gnoise(imgDFTi, 0.0, stdev);

    // add some poisson noise
    // Pnoise(imgDFTi, stdev*50000);

    // add some speckle noise
    // SPEnoise(imgDFTi, stdev);

    // add some uniform noise
    // Unoise(imgDFTi, 1.00, stdev);

    cv::imshow("Initial Image", imgDFTi);

    // deblurring starts here

    //|x| norm

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    imgDFTi.copyTo(imgDFTq);
    // GradNorm(imgDFTi, imgDFTq);
    // cvLaplace(imgDFTi,imgDFTq);
    // HLNorm(imgDFTi, imgDFTq);
    // TotVar(imgDFTi, imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it via the secant mtehod
    lambd = 1e-3;
    step = 1e-4;
    f1 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
    f2 = GCV(lambd * lambd + step, imgDFTqt, imgDFT, imgDFTq);

    iter = 0;
    do {
      if (f2 != f1)
        step = -f2 * step / (f2 - f1);
      else
        break;
      lambd += step;
      f1 = f2;
      f2 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
      iter++;
      // cout << lambd << '\n';
    } while ((abs(f2 - f1) > 1e-8) &&
             (iter < 1000));  // while(abs(f2-f1)>1e-6*step);

    lambd1 = lambd;

    /*
   for (lambd =0; lambd<1e-2; lambd+=1e-4)
   {
           //cout << lambd << "  " << GCV(lambd, imgDFTqt, imgDFT, imgDFTq) <<
   '\n';
           //cout << lambd << "  " << GCV(lambd, imgDFTqt, imgDFT) << '\n';
   //this is a simplified transform for |x|
   }
    */
    cv::imshow("|x|", imgDFT);
    cout << stdev << "  " << lambd * lambd << "  ";
    // deblurring with regularization
    FDFilter(imgDFTi, imgb, imgDFT, imgDFTq, lambd * lambd);
    cv::imshow("|x|", imgDFT);

    //|Grad x| norm

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    // imgDFTi.copyTo(imgDFTq);
    GradNorm(imgDFTi, imgDFTq);
    // cvLaplace(imgDFTi,imgDFTq);
    // HLNorm(imgDFTi, imgDFTq);
    // TotVar(imgDFTi, imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it via the secant mtehod
    lambd = 1e-3;
    step = 1e-4;
    f1 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
    f2 = GCV(lambd * lambd + step, imgDFTqt, imgDFT, imgDFTq);

    iter = 0;
    do {
      if (f2 != f1)
        step = -f2 * step / (f2 - f1);
      else
        break;
      lambd += step;
      f1 = f2;
      f2 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
      iter++;
      // cout << lambd << '\n';
    } while ((abs(f2 - f1) > 1e-8) &&
             (iter < 1000));  // while(abs(f2-f1)>1e-6*step);

    lambd2 = lambd;

    // deblurring with regularization
    FDFilter(imgDFTi, imgb, imgDFT, imgDFTq, lambd * lambd);
    cv::imshow("|Grad x|", imgDFT);
    cout << lambd * lambd << "  ";

    //|Lapl x| norm

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    // imgDFTi.copyTo(imgDFTq);
    // GradNorm(imgDFTi, imgDFTq);
    cv::Laplacian(imgDFTi, imgDFTq, -1);
    // HLNorm(imgDFTi, imgDFTq);
    // TotVar(imgDFTi, imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it via the secant mtehod
    lambd = 1e-3;
    step = 1e-4;
    f1 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
    f2 = GCV(lambd * lambd + step, imgDFTqt, imgDFT, imgDFTq);

    iter = 0;
    do {
      if (f2 != f1)
        step = -f2 * step / (f2 - f1);
      else
        break;
      lambd += step;
      f1 = f2;
      f2 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
      iter++;
      // cout << lambd << '\n';
    } while ((abs(f2 - f1) > 1e-8) &&
             (iter < 1000));  // while(abs(f2-f1)>1e-6*step);

    lambd3 = lambd;

    // deblurring with regularization
    FDFilter(imgDFTi, imgb, imgDFT, imgDFTq, lambd * lambd);
    cv::imshow("|Lapl x|", imgDFT);
    cout << lambd * lambd << "  ";

    //|HLapl x| norm

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    // imgDFTi.copyTo(imgDFTq);
    // GradNorm(imgDFTi, imgDFTq);
    // cvLaplace(imgDFTi,imgDFTq);
    HLNorm(imgDFTi, imgDFTq);
    // TotVar(imgDFTi, imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it via the secant mtehod
    lambd = 1e-3;
    step = 1e-4;
    f1 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
    f2 = GCV(lambd * lambd + step, imgDFTqt, imgDFT, imgDFTq);

    iter = 0;
    do {
      if (f2 != f1)
        step = -f2 * step / (f2 - f1);
      else
        break;
      lambd += step;
      f1 = f2;
      f2 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
      iter++;
      // cout << lambd << '\n';
    } while ((abs(f2 - f1) > 1e-8) &&
             (iter < 1000));  // while(abs(f2-f1)>1e-6*step);

    lambd4 = lambd;

    // deblurring with regularization
    FDFilter(imgDFTi, imgb, imgDFT, imgDFTq, lambd * lambd);
    cv::imshow("|HLapl x|", imgDFT);
    cout << lambd * lambd << "  ";

    // TV x norm

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    // imgDFTi.copyTo(imgDFTq);
    // GradNorm(imgDFTi, imgDFTq);
    // cvLaplace(imgDFTi,imgDFTq);
    // HLNorm(imgDFTi, imgDFTq);
    TotVar(imgDFTi, imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it via the secant mtehod
    lambd = 1e-3;
    step = 1e-4;
    f1 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
    f2 = GCV(lambd * lambd + step, imgDFTqt, imgDFT, imgDFTq);

    iter = 0;
    do {
      if (f2 != f1)
        step = -f2 * step / (f2 - f1);
      else
        break;
      lambd += step;
      f1 = f2;
      f2 = GCV(lambd * lambd, imgDFTqt, imgDFT, imgDFTq);
      iter++;
      // cout << lambd << '\n';
    } while ((abs(f2 - f1) > 1e-8) &&
             (iter < 1000));  // while(abs(f2-f1)>1e-6*step);

    lambd5 = lambd;

    // deblurring with regularization
    FDFilter(imgDFTi, imgb, imgDFT, imgDFTq, lambd * lambd);
    cv::imshow("TV x", imgDFT);
    cout << lambd * lambd << '\n';

    char c = cv::waitKey(10);
    if (c == 27) stdev = 1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    // cvScale(imgDFT,img2,255);
    // borders for boundary conditions
    // cv::Rect(img1,
    // cv::Point(kw,kh),cv::Point(img0.cols-kw,img0.rows-kh),cv::Scalar(255));

    stdev += 0.00005;
  } while (stdev < 0.01);

  // remove borders
  // cvGetSubRect(img1, &tmp, cvRect(kw, kh, img2.cols, img2.rows));
  // tmp.copyTo(img2);

  // borders for partially restored zone
  // cv::Rect(img2,
  // cv::Point(kw/2,kh/2),cv::Point(img0.cols-kw/2,img0.rows-kh/2),cv::Scalar(255));

  cout << "\n Press 'Esc' Key to Exit \n";
  while (1) {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }
  cv::destroyWindow("Initial Image");
  cv::destroyWindow("|x|");
  cv::destroyWindow("|Grad x|");
  cv::destroyWindow("|Lapl x|");
  cv::destroyWindow("|HLapl x|");
  cv::destroyWindow("TV x");

  return (0);
}