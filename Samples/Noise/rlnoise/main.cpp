/*
 * Simple program to create
 * space variant motion blurr
 * Richardson Lucy algorithm
 * with partial sums
 * and noise is added
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "noise.hpp"

using namespace std;

// parameters of the noise
//  Gaussian: st. dev., mean; Uniform noise deviation; Salt'n'Pepper part;
//  Poisson noise coefficient; Specle noise coefficient
int pos_n1 = 0, pos_n2 = 0, pos_n3 = 0, pos_n4 = 0, pos_n5 = 0, pos_n6 = 0;

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

void FGet2D(const cv::Mat &Y, int k, int i, float *re, float *im) {
  int x, y;                       // pixel coordinates of Re Y(i,k).
  float *Yptr = (float *)Y.data;  // pointer to Re Y(i,k)
  int stride = Y.step / sizeof(float);

  if (k == 0 || k * 2 == Y.cols) {
    x = (k == 0 ? 0 : Y.cols - 1);
    if (i == 0 || i * 2 == Y.rows) {
      y = i == 0 ? 0 : Y.rows - 1;
      *re = Yptr[y * stride + x];
      *im = 0;
    } else if (i * 2 < Y.rows) {
      y = i * 2 - 1;
      *re = Yptr[y * stride + x];
      *im = Yptr[(y + 1) * stride + x];
    } else {
      y = (Y.rows - i) * 2 - 1;
      *re = Yptr[y * stride + x];
      *im = Yptr[(y + 1) * stride + x];
    }
  } else if (k * 2 < Y.cols) {
    x = k * 2 - 1;
    y = i;
    *re = Yptr[y * stride + x];
    *im = Yptr[y * stride + x + 1];
  } else {
    x = (Y.cols - k) * 2 - 1;
    y = i;
    *re = Yptr[y * stride + x];
    *im = Yptr[y * stride + x + 1];
  }
}

void ksetMoveXY(cv::Mat &kernel, int x, int y, int x0 = 0, int y0 = 0) {
  cv::Scalar ksum;

  kernel = 0;
  cv::line(kernel, cv::Point(x0 + kernel.cols / 2, y0 + kernel.rows / 2),
           cv::Point(x + kernel.cols / 2, y + kernel.rows / 2), cv::Scalar(1.0),
           1, cv::LINE_AA);
  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

/*
 * Function which introduces
 * blurring kernel on image
 * with periodic boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(cv::Mat &imga, cv::Mat &imgb, cv::Mat &imgc, int fl = 0) {
  float s1, s2, s3;
  int i, j;

  for (int row = 0; row < imga.rows; row++) {
    // here the kernel is changed
    imgb = 0;
    if (fl == 0)
      ksetMoveXY(imgb, 5.0, 5.0);
    else
      ksetMoveXY(imgb, -5.0, -5.0);

    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;

      for (int row1 = 0; row1 < imgb.rows; row1++) {
        for (int col1 = 0; col1 < imgb.cols; col1++) {
          s1 = ((float *)(imgb.data + row1 * imgb.step))[col1];

          if ((row - row1 + imgb.rows / 2) >= 0) {
            if ((row - row1 + imgb.rows / 2) < (imga.rows)) {
              i = row - row1 + imgb.rows / 2;
            } else {
              i = row - row1 + imgb.rows / 2 - imga.rows + 1;
            }

          } else {
            i = (row - row1 + imgb.rows / 2) + imga.rows - 1;
          }

          if ((col - col1 + imgb.cols / 2) >= 0) {
            if ((col - col1 + imgb.cols / 2) < (imga.cols)) {
              j = col - col1 + imgb.cols / 2;
            } else {
              j = col - col1 + imgb.cols / 2 - imga.cols + 1;
            }

          } else {
            j = (col - col1 + imgb.cols / 2) + imga.cols - 1;
          }

          s3 = ((float *)(imga.data + i * imga.step))[j] * s1;
          s2 += s3;
        }
      }
      ((float *)(imgc.data + row * imgc.step))[col] = s2;
    }
  }
}

/*
 * Total variation calculation
 */

float TotVar(cv::Mat &imga) {
  float s1, s2, s = 0;
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
      s += abs(s1) + abs(s2);
    }
  }
  return (s);
}

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

// initial declaration of trackbar reading function
void GetNois(int pos, void *);

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, img1, img2, img3, img4, img5,
      imgl;  // initial, blurred, kernel, deblurred and noise image, laplace
  int m = 320, n = 240, r = 40,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int it;          // iteration counter
  const int x1 = 5, y1 = 5;                 // Define vector of motion blurr
  float norm, norm1, norm2, oldnorm, delt;  // norm of the images
  float lambda, gi;                         // regularization parameters

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img, CV_32F);
    img = img / 255.0;
  } else {
    img = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img);
  }

  img1 = img.clone();
  imgl = img.clone();

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);

  // motion blurr
  // ksetMoveXY(kernel,x1,y1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, img1);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);

  cv::namedWindow("Motion Blurr", cv::WINDOW_NORMAL);
  cv::resizeWindow("Motion Blurr", 600, 1);
  cv::createTrackbar("Gaussian noise dev., % ", "Motion Blurr", &pos_n1, 100,
                     GetNois);
  cv::createTrackbar("Gaussian noise mean, % ", "Motion Blurr", &pos_n2, 100,
                     GetNois);
  cv::createTrackbar("Uniform noise dev, % ", "Motion Blurr", &pos_n3, 100,
                     GetNois);
  cv::createTrackbar("Bad Pixels, o/oo ", "Motion Blurr", &pos_n4, 100,
                     GetNois);
  cv::createTrackbar("Poisson noise coeff.", "Motion Blurr", &pos_n5, 500,
                     GetNois);
  cv::createTrackbar("Speckle noise coeff.", "Motion Blurr", &pos_n6, 100,
                     GetNois);

  // set the noise
  do {
    img1.copyTo(imgl);
    Gnoise(imgl, pos_n2 / 100.0, pos_n1 / 100.0);
    Unoise(imgl, pos_n3 / 100.0, 1.0);
    SPnoise(imgl, pos_n4 / 2000.0);
    if (pos_n5 != 0) Pnoise(imgl, pos_n5);
    SPEnoise(imgl, pos_n6 / 100.0);
    cv::imshow("Blurred", imgl);

    char c = cv::waitKey(10);
    if (c == 27) break;  // press Enter to continue
  } while (1);

  imgl.copyTo(img1);

  // initial parameter of the norm
  gi = 2 * cv::norm(img1, cv::NORM_L2);

  // displays image
  cv::imshow("Blurred", img1);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);

  // Fourier derivation of the kernel
  img2 = img1.clone();  // old image
  img3 = img1.clone();  // new image
  img4 = img1.clone();  // blurred old image
  img5 = img1.clone();

  BlurrPBCsv(img2, kernel, img4);
  // norm2=TotVar(img2)/((img2.cols)*(img2.rows));
  cv::Laplacian(img2, imgl, -1);
  norm2 = cv::norm(imgl, cv::NORM_L2) / ((img2.cols) * (img2.rows));
  cv::subtract(img4, img1, img5);
  norm1 = cv::norm(img5, cv::NORM_L2) / ((img2.cols) * (img2.rows));
  lambda = norm1 / (gi - norm2);
  cout << "Initial lambda= " << lambda << '\n';
  oldnorm = norm1 + lambda * norm2 + 1;

  // Richardson-Lucy starts here
  it = 0;
  do {
    // Mk=H*Ik
    BlurrPBCsv(img2, kernel, img4);

    // norm2=TotVar(img2)/((img2.cols)*(img2.rows));
    cv::Laplacian(img2, imgl, -1);
    norm2 = cv::norm(imgl, cv::NORM_L2) / ((img2.cols) * (img2.rows));
    cv::subtract(img4, img1, img5);  // r=Ax-b
    norm1 = cv::norm(img5, cv::NORM_L2) / ((img2.cols) * (img2.rows));
    lambda = norm1 / (gi - norm2);

    // D/Mk
    cv::divide(img1, img4, img4);
    // Ht*(D/Mk)
    BlurrPBCsv(img4, kernel, img5, 1);
    // pixel by pixel multiply
    cv::multiply(img5, img2, img3);

    // additional part to add coefficient
    cv::subtract(img3, img2, img3);
    cv::addWeighted(img2, 1.0, img3, 1.0, 0.0, img3);

    norm = norm1 + lambda * norm2;
    delt = oldnorm - norm;
    oldnorm = norm;

    img2 = img3.clone();

    cv::imshow("Deblurred", img3);
    char c = cv::waitKey(10);

    if (c == 27) break;

    it++;
    cout << it << "  " << delt << "  " << norm << "  " << norm1 << "  " << norm2
         << "  " << lambda << '\n';
  } while (abs(delt) > 1e-7);

  cv::waitKey(0);

  return 0;
}

void GetNois(int pos, void *) {
  pos_n1 = cv::getTrackbarPos("Gaussian noise dev., % ", "Motion Blurr");
  pos_n2 = cv::getTrackbarPos("Gaussian noise mean, % ", "Motion Blurr");
  pos_n3 = cv::getTrackbarPos("Uniform noise dev, % ", "Motion Blurr");
  pos_n4 = cv::getTrackbarPos("Bad Pixels, o/oo ", "Motion Blurr");
  pos_n5 = cv::getTrackbarPos("Poisson noise coeff.", "Motion Blurr");
  pos_n6 = cv::getTrackbarPos("Speckle noise coeff.", "Motion Blurr");
}
