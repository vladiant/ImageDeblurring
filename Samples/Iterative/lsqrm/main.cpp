//============================================================================
// Name        : lsqr.cpp
// Author      : Vladislav Antonov
// Version     :
// Description : Program to test modified LSQR motion deblurring
//             : Algorithm based on paper:
//             : RESTORATION OF BLURRED IMAGES BY GLOBAL LEAST SQUARES METHOD
//             : Sei-young Chung, SeYoung Oh, and SunJoo Kwon,
//             : JOURNAL OF THE CHUNGCHEONG MATHEMATICAL SOCIETY
//             : Volume 22, No. 2, June 2009, pp. 177-186
// Created on  : April 5, 2012
//============================================================================

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
 * Function which introduces
 * blurring kernel on image
 * with periodic boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(cv::Mat &imga, cv::Mat &imgb, cv::Mat &imgc, int fl = 0) {
  float s1, s2, s3, s4, s5;
  int i, j;

  cv::namedWindow("test", 0);

  for (int row = 0; row < imga.rows; row++) {
    // here the kernel is changed
    imgb = 0;
    if (fl == 0) {
      // ksetMoveXY(imgb,5,0);

      imgb = 0;
      s4 = 0;
      for (int col = 0; col < imgb.cols / 2; col++) {
        // classic
        // s5=(abs(col)>5)?0:1;
        // Gaussian
        // s5=exp(-col*col/(2.0*5*5));
        // erfc
        s5 = 0.5 * erfc((col - 10) / 2.0);
        s4 += s5;
        ((float *)(imgb.data + 0 * imgb.step))[col + imgb.cols / 2] = s5;
      }
      imgb = imgb / s4;
    } else {
      // ksetMoveXY(imgb,-5,0);
      imgb = 0;
      s4 = 0;
      for (int col = 0; col > -(imgb.cols / 2); col--) {
        // classic
        // s5=(abs(col)>5)?0:1;
        // Gaussian
        // s5=exp(-col*col/(2.0*5*5));
        // erfc
        s5 = 0.5 * erfc((abs(col) - 10) / 2.0);
        s4 += s5;
        // cout << s5 << '\n';
        ((float *)(imgb.data + 0 * imgb.step))[col + imgb.cols / 2] = s5;
      }
      imgb = imgb / s4;
    }

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

int main(int argc, char *argv[]) {
  // initial two images, blurred, deblurred, residual and image, blurred
  // preconditioned image, temp vector
  cv::Mat img, imgi, imgb, imgx;
  int m = 800, n = 600,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int it;          // iteration counter
  const int x1 = 13, y1 = 0;     // Define vector of motion blurr
  double norm;                   // norm for stopping criteria
  clock_t time_start, time_end;  // marks for start and end time

  // LSQR specific variables
  cv::Mat imgu, imgv, imgw, imgav;
  float alpha, betha, phi_dash, rho_dash;

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

  imgb = img.clone();

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 1), CV_32FC1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, imgb);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", imgb);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);

  // initial approximation of the restored image
  imgx = imgb.clone();
  imgx = 0;  // as used in the LSQR

  // LSQR initialization
  imgu = imgb.clone();
  betha = sqrt(cv::norm(imgu, cv::NORM_L2));
  imgu = imgu / betha;

  imgv = imgu.clone();
  BlurrPBCsv(imgu, kernel, imgv, 1);
  alpha = sqrt(cv::norm(imgv, cv::NORM_L2));
  imgv = imgv / alpha;

  imgw = imgv.clone();

  phi_dash = betha;
  rho_dash = alpha;

  // one temporal image
  imgav = imgb.clone();

  // BlurrPBCsv(imgx, kernel, imgr);
  // BlurrPBCsv(imgr, kernel, imgr,1);

  // reset iteration counter
  it = 0;

  do {
    // time starts
    time_start = clock();

    BlurrPBCsv(imgv, kernel, imgav);
    cv::addWeighted(imgav, 1.0, imgu, -1.0 * alpha, 0.0, imgav);
    betha = sqrt(cv::norm(imgav, cv::NORM_L2));
    imgu = imgav / betha;

    BlurrPBCsv(imgu, kernel, imgav, 1);
    cv::addWeighted(imgav, 1.0, imgv, -1.0 * betha, 0.0, imgav);
    alpha = sqrt(cv::norm(imgav, cv::NORM_L2));
    imgv = imgav / alpha;

    float rho = sqrt(rho_dash * rho_dash + betha * betha);
    float c = rho_dash / rho;
    float s = betha / rho;
    float Theta = s * alpha;
    rho_dash = c * alpha;
    float phi = c * phi_dash;
    phi_dash = -s * phi_dash;

    // cout << alpha << "  " << betha << "  " << phi_dash << "  " << rho_dash <<
    // "  " << rho << "  " << c << "  " << s << "  " << Theta << "  " << phi <<
    // endl;

    // update the images
    cv::addWeighted(imgx, 1.0, imgw, 1.0 * phi / rho, 0.0, imgx);
    cv::addWeighted(imgv, 1.0, imgw, -1.0 * Theta / rho, 0.0, imgw);

    // time end
    time_end = clock();

    // stopping coefficient

    norm = abs(phi_dash);

    cv::imshow("Deblurred", imgx);

    char cc = cv::waitKey(10);
    if (cc == 27) break;

    it++;
    cout << it << "  " << norm << "  "
         << (time_end - time_start) * 1.0 / CLOCKS_PER_SEC << endl;
  } while ((norm > 1.5e-2));

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred");

  return 0;
}
