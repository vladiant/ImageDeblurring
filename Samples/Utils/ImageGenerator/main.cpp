/*
 * This code generates files
 * with simple test images
 * with different resolutions
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void Ximgset(cv::Mat& imga)  // generates the image
{
  imga = 0;
  // rectangle coordinates
  int i1 = round((imga.rows) / 5), i2 = round(3 * (imga.cols) / 5),
      j1 = round((imga.rows) / 5), j2 = round(3 * (imga.cols) / 5);

  // circle radius
  int r = round(max(imga.rows, imga.cols) / 5);

  // draws rectangle
  cv::rectangle(imga, cv::Point(i1, j1), cv::Point(i2, j2), cv::Scalar(128),
                -1);

  // draws circle
  cv::circle(imga,
             cv::Point(round(5 * (imga.cols) / 8), round(3 * (imga.rows) / 5)),
             r, cv::Scalar(255), -1);
}

void Ximgset1(cv::Mat& imga)  // generates the image
{
  // triangle coordinates
  cv::Point pt1 =
      cv::Point(round(0.5 * (imga.cols) / 5), round(4.25 * (imga.rows) / 5));
  cv::Point pt2 =
      cv::Point(round(2 * (imga.cols) / 5), round(1.25 * (imga.rows) / 5));
  cv::Point pt3 =
      cv::Point(round(3.5 * (imga.cols) / 5), round(4.25 * (imga.rows) / 5));
  int numPts = 4;
  cv::Point pts[] = {pt1, pt2, pt3, pt1};

  imga = 0;
  // rectangle coordinates
  int i1 = round(0.7 * (imga.cols) / 5), i2 = round(3.7 * (imga.cols) / 5),
      j1 = round(0.6 * (imga.rows) / 5), j2 = round(3.5 * (imga.rows) / 5);

  // circle radius
  int r = round(1.1 * max(imga.rows, imga.cols) / 5);

  // draws rectangle
  cv::rectangle(imga, cv::Point(i1, j1), cv::Point(i2, j2), cv::Scalar(85), -1);

  // draw triangle
  cv::fillConvexPoly(imga, pts, numPts, cv::Scalar(170));

  // draws circle
  cv::circle(
      imga, cv::Point(round(5.5 * (imga.cols) / 8), round(3 * (imga.rows) / 5)),
      r, cv::Scalar(255), -1);
}

int main(int argc, char** argv) {
  cv::Mat img1;  // initial images, end image

  // sets the images

  // QVGA
  img1 = cv::Mat(cv::Size(320, 240), CV_8UC1);
  Ximgset(img1);
  cv::imwrite("testQVGA.png", img1);

  // VGA
  img1 = cv::Mat(cv::Size(640, 480), CV_8UC1);
  Ximgset(img1);
  cv::imwrite("testVGA.png", img1);

  // SVGA
  img1 = cv::Mat(cv::Size(800, 600), CV_8UC1);
  Ximgset(img1);
  cv::imwrite("testSVGA.png", img1);

  // HD720
  img1 = cv::Mat(cv::Size(1280, 720), CV_8UC1);
  Ximgset(img1);
  cv::imwrite("testHD720.png", img1);

  // HD1080
  img1 = cv::Mat(cv::Size(1920, 1080), CV_8UC1);
  Ximgset(img1);
  cv::imwrite("testHD1080.png", img1);

  // QVGA
  img1 = cv::Mat(cv::Size(320, 240), CV_8UC1);
  Ximgset1(img1);
  cv::imwrite("test1QVGA.png", img1);

  // VGA
  img1 = cv::Mat(cv::Size(640, 480), CV_8UC1);
  Ximgset1(img1);
  cv::imwrite("test1VGA.png", img1);

  // SVGA
  img1 = cv::Mat(cv::Size(800, 600), CV_8UC1);
  Ximgset1(img1);
  cv::imwrite("test1SVGA.png", img1);

  // HD720
  img1 = cv::Mat(cv::Size(1280, 720), CV_8UC1);
  Ximgset1(img1);
  cv::imwrite("test1HD720.png", img1);

  // HD1080
  img1 = cv::Mat(cv::Size(1920, 1080), CV_8UC1);
  Ximgset1(img1);
  cv::imwrite("test1HD1080.png", img1);

  return (0);
}