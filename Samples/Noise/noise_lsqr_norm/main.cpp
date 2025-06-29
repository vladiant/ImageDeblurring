/*
 * This code performs Fourier motion deblurring
 * and calculates the norm of the residual
 * |A.x-b|
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

void ksetMoveXY(cv::Mat &kernel, int x, int y, int x0 = 0, int y0 = 0) {
  cv::Scalar ksum;

  kernel = 0;
  cv::line(kernel, cv::Point(x0 + kernel.cols / 2, y0 + kernel.rows / 2),
           cv::Point(x + kernel.cols / 2, y + kernel.rows / 2), cv::Scalar(1.0),
           1, cv::LINE_AA);
  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

int main(int argc, char **argv) {
  cv::Mat imgi, img0, img2;  // initial images, end image

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

  cv::Mat tmp;  // for border removal
  cv::Scalar ksum;

  // Norms declaration
  float norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7;

  // cut off variables
  float re, im, stdev, step;

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
    img0 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img0, CV_32F);
    img0 = imgi / 255.0;
  } else {
    img0 = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img0);
  }

  img2 = cv::Mat(cv::Size(img0.cols, img0.rows), CV_8UC1);

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
  stdev = 0;
  step = 0.001;
  cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);

  do {
    // declaring initial images
    imgDFTi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFT = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);

    /*
    //use this in standalone deblurring
    //create border
    cv::copyMakeBorder(imgi,img,cvPoint(kernel.cols,kernel.rows),
    cv::BORDER_REPLICATE);
    //supress ringing from boundaries
    EdgeTaper(img, kernel.cols, kernel.cols);
     */

    img.copyTo(imgDFTi);

    // add some salt and pepper noise
    // SPnoise(imgDFTi, stdev);

    // add some gauss noise
    Gnoise(imgDFTi, 0.0, stdev);

    // add some poisson noise
    // Pnoise(imgDFTi, stdev * 50000);

    // add some speckle noise
    // SPEnoise(imgDFTi, stdev);

    // add some uniform noise
    // Unoise(imgDFTi, 1.00, stdev);

    // No regularization
    FDFilter(imgDFTi, imgb, imgDFT, 0);

    // norms calculation
    // FCFilter(imgDFT,imgb,imgl);
    // cv::subtract(imgl,imgDFTi,imgl);
    norm0 = cv::norm(imgl, cv::NORM_L2);  //  |Ax-b|
    norm0 /= (imgl.cols) * (imgl.rows);

    cv::imshow("Initial Image", imgDFTi);
    cv::imshow("test", imgDFT);

    cout << stdev << "   " << norm0 << '\n';

    char c = cv::waitKey(10);
    if (c == 27) stdev = 1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    stdev += 0.00005;

    img1 = imgDFT * 255;
  } while (stdev < 0.01);

  // remove borders and clear the memory
  img2 = img1(cv::Rect(kw, kh, img2.cols, img2.rows)).clone();

  // borders for partially restored zone
  cv::rectangle(img2, cv::Point(kw / 2, kh / 2),
                cv::Point(img0.cols - kw / 2, img0.rows - kh / 2),
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