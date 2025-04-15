/*
 * Simple program to create
 * the Fourier spectra of an image
 */

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

int main(int argc, char *argv[]) {
  // initial two images
  cv::Mat img, imgi;
  // image for Fourier spectrum, real and imaginary part, absolute values
  cv::Mat imgDFT, imgDFTr, imgDFTi, imgDFTa;
  int m = 320, n = 240;  // image dimensions and radius of blurring
  double mn, mx;         // minimal and maximal values

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

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Real DFT", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Imaginary DFT", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("|DFT|", cv::WINDOW_AUTOSIZE);

  // Fourier transformed image, real and imaginary
  imgDFT = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC2);
  imgDFTr = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  imgDFTi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  imgDFTa = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);

  // unoptimized Fourier Transform
  cv::dft(img, imgDFT, cv::DFT_COMPLEX_OUTPUT);

  // split it
  cv::split(imgDFT, std::vector{imgDFTr, imgDFTi});

  /*
  // mappping procedure
  cv::minMaxLoc(imgDFTr,&mn,&mx);
  cout << mn << "  " << mx << '\n';
  cvConvertScale(imgDFTr,imgDFTr,1.0/(mx-mn),-mn);
  cv::minMaxLoc(imgDFTi,&mn,&mx);
  cout << mn << "  " << mx << '\n';
  cvConvertScale(imgDFTi,imgDFTi,1.0/(mx-mn),-mn);
  */

  // calculate absolute values
  for (int row = 0; row < imgDFTa.rows; row++) {
    for (int col = 0; col < imgDFTa.cols / 2; col++) {
      float a1 = ((float *)(imgDFTr.data + row * imgDFTr.step))[col];
      float a2 = ((float *)(imgDFTi.data + row * imgDFTi.step))[col];
      int a3 = row + imgDFTa.rows / 2;
      if (a3 > (imgDFTa.rows - 1)) a3 -= imgDFTa.rows - 1;
      int a4 = col + imgDFTa.cols / 2;
      if (a4 > (imgDFTa.cols - 1)) a3 -= imgDFTa.cols - 1;
      ((float *)(imgDFTa.data + a3 * imgDFTa.step))[a4] =
          log(1 + sqrt(a1 * a1 + a2 * a2));
      ((float *)(imgDFTa.data + a3 * imgDFTa.step))[(imgDFTa.cols - 1) - a4] =
          log(1 + sqrt(a1 * a1 + a2 * a2));
      //((float*)(imgDFTa.data +
      // a3*imgDFTa.step))[(imgDFTa.cols-1)-a4]=(sqrt(a1*a1+a2*a2));
    }
  }

  // print the values
  for (int row = imgDFTa.rows / 2; row < imgDFTa.rows; row++) {
    for (int col = imgDFTa.cols / 2; col < imgDFTa.cols; col++) {
      float a3 = ((float *)(imgDFTa.data + row * imgDFTa.step))[col];

      // 3D case
      // cout << (row-imgDFTa.rows/2)*1.0/(imgDFTa.rows/2) << "  " <<
      // (col-imgDFTa.cols/2)*1.0/(imgDFTa.cols/2) << "  " << exp(a3)-1 <<
      // '\n';

      // 1D case + 3D
      float a1 = (row - imgDFTa.rows / 2) * 1.0 / (imgDFTa.rows / 2);
      float a2 = (col - imgDFTa.cols / 2) * 1.0 / (imgDFTa.cols / 2);
      cout << a1 << "  " << a2 << "  " << sqrt(a1 * a1 + a2 * a2) << "  "
           << exp(a3) - 1 << '\n';
    }
  }

  // mappping procedure
  cv::minMaxLoc(imgDFTa, &mn, &mx);
  // cout << mn << "  " << mx << '\n';
  imgDFTa = imgDFTa / ((mx - mn) - mn);

  // show them
  cv::imshow("Real DFT", imgDFTr);
  cv::imshow("Imaginary DFT", imgDFTi);
  cv::imshow("|DFT|", imgDFTa);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Real DFT");
  cv::destroyWindow("Imaginary DFT");
  cv::destroyWindow("|DFT|");

  return 0;
}