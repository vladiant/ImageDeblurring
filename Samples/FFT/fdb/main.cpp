/*
 * This code performs Fourier defocusing blurring and deblurring
 * However it does now work with all sizes of images
 * This program will be used to investigate the dependence
 *
 * if r<6 some deblurring problems appear;
 * they seem to be due to truncation errors for the circle
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "fdb.hpp"

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

int main(int argc, char *argv[]) {
  // initial, blurred, kernel and deblurred image
  cv::Mat imgi, img, img1, imgb, img2;
  cv::Mat tmp, tmp1;
  int m = 320, n = 240, r = 6;  // image dimensions and radius of blurring
  cv::Mat kernel(2 * r + 1, 2 * r + 1, CV_32FC1);

  int m1, n1;
  cv::Scalar ksum;

  if (argc != 3) {
    cout << "\n Program use: \n";
    cout << argv[0] << "  m n \n";
    cout << " If undefined, then m=320 n=240 \n";
  } else {
    m1 = atoi(argv[1]);
    if (m1 > 0) {
      m = m1;
    } else {
      cout << "m defaulted to 320 \n";
    }

    n1 = atoi(argv[2]);
    if (n1 > 0) {
      n = n1;
    } else {
      cout << "n defaulted to 240 \n";
    }

    cout << "\n n= " << n << " , m=" << m << '\n';
  }

  cv::Mat img3(2 * r + 1 + n, 2 * r + 1 + m, CV_32FC1);

  // creates initial image
  img = cv::Mat(cv::Size(m, n), CV_32FC1);
  Ximgset(img);

  img1 = img.clone();

  // creates defocus kernel
  kernel = 0;
  cv::circle(kernel, cv::Point(r, r), r, cv::Scalar(1.0), -1);
  ksum = cv::sum(kernel);
  kernel /= ksum.val[0];

  // creates smearing image
  imgb = cv::Mat(img1.rows, img1.cols, CV_32FC1);
  Blurrset(imgb, kernel);

  // blurring
  FCFilter(img1, imgb, img1);

  // deblurring starts here

  img2 = img1.clone();
  FDFilter(img2, imgb, img2);

  // apply mirror border
  MirrorBorder(img2, img3, kernel.cols / 2, kernel.rows / 2);
  // EdgeTaper(img2, kernel.cols, kernel.rows);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred Defocus", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred Defocus", img1);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Deblurred", img2);
  cv::waitKey(0);
  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred Defocus");
  cv::destroyWindow("Deblurred");

  return 0;
}