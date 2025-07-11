/*
 * Simple program to create
 * to test edge detection
 * with Prewitt and
 * Frei & Chen filters
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

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

void Ximgset1(cv::Mat &imga)  // generates the image
{
  // triangle coordinates
  cv::Point pt1 =
      cv::Point(round(0.5 * (imga.cols) / 5), round(4.25 * (imga.rows) / 5));
  cv::Point pt2 =
      cv::Point(round(2 * (imga.cols) / 5), round(1.25 * (imga.rows) / 5));
  cv::Point pt3 =
      cv::Point(round(3.5 * (imga.cols) / 5), round(4.25 * (imga.rows) / 5));
  int numPts = 5;
  cv::Point pts[] = {pt1, pt2, pt3, pt1};

  imga = 0;
  // rectangle coordinates
  int i1 = round(0.7 * (imga.cols) / 5), i2 = round(3.7 * (imga.cols) / 5),
      j1 = round(0.6 * (imga.rows) / 5), j2 = round(3.5 * (imga.rows) / 5);

  // circle radius
  int r = round(1.1 * max(imga.rows, imga.cols) / 5);

  // draws rectangle
  cv::rectangle(imga, cv::Point(i1, j1), cv::Point(i2, j2),
                cv::Scalar(1.0 / 3.0), -1);

  // draw triangle
  cv::fillConvexPoly(imga, pts, numPts, cv::Scalar(2.0 / 3.0));

  // draws circle
  cv::circle(
      imga, cv::Point(round(5.5 * (imga.cols) / 8), round(3 * (imga.rows) / 5)),
      r, cv::Scalar(1.0), -1);
}

/*
 * Prewitt filter for edge detection
 */

void Prewitt(cv::Mat &imga, cv::Mat &imgf) {
  int i1, i2, j1, j2;
  float p1, p2, p3, p4, p6, p7, p8, p9, s1, s2;
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      // replicate boundary conditions
      i1 = (row == 0 ? 0 : row - 1);
      i2 = (row == imga.cols - 1 ? imga.cols - 1 : row + 1);
      j1 = (col == 0 ? 0 : col - 1);
      j2 = (col == imga.cols - 1 ? imga.cols - 1 : col + 1);
      p1 = ((float *)(imga.data + i1 * imga.step))[j1];
      p2 = ((float *)(imga.data + row * imga.step))[j1];
      p3 = ((float *)(imga.data + i2 * imga.step))[j1];
      p4 = ((float *)(imga.data + i1 * imga.step))[col];
      p6 = ((float *)(imga.data + i2 * imga.step))[col];
      p7 = ((float *)(imga.data + i1 * imga.step))[j2];
      p8 = ((float *)(imga.data + row * imga.step))[j2];
      p9 = ((float *)(imga.data + i2 * imga.step))[j2];
      s1 = p1 + p2 + p3 - p7 - p8 - p9;
      s2 = p3 + p6 + p9 - p1 - p4 - p7;
      ((float *)(imgf.data + row * imgf.step))[col] = sqrt(s1 * s1 + s2 * s2);
    }
  }
}

/*
 * Frei & Chen filter
 * edges & lines
 */

void FrChn(cv::Mat &imga, cv::Mat &imgf) {
  int i1, i2, j1, j2;
  float p1, p2, p3, p4, p5, p6, p7, p8, p9;
  float s = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0,
        s9 = 0;
  const float sq2 = sqrt(2);
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      // replicate boundary conditions
      i1 = (row == 0 ? 0 : row - 1);
      i2 = (row == imga.cols - 1 ? imga.cols - 1 : row + 1);
      j1 = (col == 0 ? 0 : col - 1);
      j2 = (col == imga.cols - 1 ? imga.cols - 1 : col + 1);
      p1 = ((float *)(imga.data + i1 * imga.step))[j1];
      p2 = ((float *)(imga.data + row * imga.step))[j1];
      p3 = ((float *)(imga.data + i2 * imga.step))[j1];
      p4 = ((float *)(imga.data + i1 * imga.step))[col];
      p5 = ((float *)(imga.data + row * imga.step))[col];
      p6 = ((float *)(imga.data + i2 * imga.step))[col];
      p7 = ((float *)(imga.data + i1 * imga.step))[j2];
      p8 = ((float *)(imga.data + row * imga.step))[j2];
      p9 = ((float *)(imga.data + i2 * imga.step))[j2];
      // edges
      s1 = (p1 + p2 * sq2 + p3 - p7 - p9 * sq2 - p9) / (2 * sq2);
      s2 = (p1 - p3 + p4 * sq2 - p6 * sq2 + p7 - p9) / (2 * sq2);
      s3 = (-p2 + p3 * sq2 + p4 - p6 - p7 * sq2 + p8) / (2 * sq2);
      s4 = (p1 * sq2 - p2 - p4 + p6 + p8 - p9 * sq2) / (2 * sq2);
      // lines
      s5 = (p2 - p4 - p6 + p8) / 2;
      s6 = (-p1 + p3 + p7 - p9) / 2;
      s7 = (p1 - p2 * 2 + p3 - p4 * 2 + p5 * 4 - p6 * 2 + p7 - p8 * 2 + p9) / 6;
      s8 =
          (-p1 * 2 + p2 - p3 * 2 + p4 + p5 * 4 + p6 - p7 * 2 + p8 - p9 * 2) / 6;
      // averaging
      s9 = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
      // sum squares
      s = s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6 + s7 * s7 +
          s8 * s8 + s9 * s9;
      // edges
      // s=(s1*s1+s2*s2+s3*s3+s4*s4)/s;
      // lines
      // s=(s5*s5+s6*s6+s7*s7+s8*s8)/s;
      // edges & lines
      if (s != 0)
        s = (s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6 +
             s7 * s7 + s8 * s8) /
            s;
      ((float *)(imgf.data + row * imgf.step))[col] = sqrt(s);
    }
  }
}

/*
 * Frei & Chen filter
 * edges
 */

void FrChn1(cv::Mat &imga, cv::Mat &imgf) {
  int i1, i2, j1, j2;
  float p1, p2, p3, p4, p5, p6, p7, p8, p9;
  float s, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0,
           s9 = 0;
  const float sq2 = sqrt(2);
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      // replicate boundary conditions
      i1 = (row == 0 ? 0 : row - 1);
      i2 = (row == imga.cols - 1 ? imga.cols - 1 : row + 1);
      j1 = (col == 0 ? 0 : col - 1);
      j2 = (col == imga.cols - 1 ? imga.cols - 1 : col + 1);
      p1 = ((float *)(imga.data + i1 * imga.step))[j1];
      p2 = ((float *)(imga.data + row * imga.step))[j1];
      p3 = ((float *)(imga.data + i2 * imga.step))[j1];
      p4 = ((float *)(imga.data + i1 * imga.step))[col];
      p5 = ((float *)(imga.data + row * imga.step))[col];
      p6 = ((float *)(imga.data + i2 * imga.step))[col];
      p7 = ((float *)(imga.data + i1 * imga.step))[j2];
      p8 = ((float *)(imga.data + row * imga.step))[j2];
      p9 = ((float *)(imga.data + i2 * imga.step))[j2];
      // edges
      s1 = (p1 + p2 * sq2 + p3 - p7 - p9 * sq2 - p9) / (2 * sq2);
      s2 = (p1 - p3 + p4 * sq2 - p6 * sq2 + p7 - p9) / (2 * sq2);
      s3 = (-p2 + p3 * sq2 + p4 - p6 - p7 * sq2 + p8) / (2 * sq2);
      s4 = (p1 * sq2 - p2 - p4 + p6 + p8 - p9 * sq2) / (2 * sq2);
      // lines
      s5 = (p2 - p4 - p6 + p8) / 2;
      s6 = (-p1 + p3 + p7 - p9) / 2;
      s7 = (p1 - p2 * 2 + p3 - p4 * 2 + p5 * 4 - p6 * 2 + p7 - p8 * 2 + p9) / 6;
      s8 =
          (-p1 * 2 + p2 - p3 * 2 + p4 + p5 * 4 + p6 - p7 * 2 + p8 - p9 * 2) / 6;
      // averaging
      s9 = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
      // sum squares
      s = s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6 + s7 * s7 +
          s8 * s8 + s9 * s9;
      // edges
      if (s != 0) s = (s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4) / s;
      ((float *)(imgf.data + row * imgf.step))[col] = sqrt(s);
    }
  }
}

/*
 * Frei & Chen filter
 * lines
 */

void FrChn2(cv::Mat &imga, cv::Mat &imgf) {
  int i1, i2, j1, j2;
  float p1, p2, p3, p4, p5, p6, p7, p8, p9;
  float s, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0,
           s9 = 0;
  const float sq2 = sqrt(2);
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      // replicate boundary conditions
      i1 = (row == 0 ? 0 : row - 1);
      i2 = (row == imga.cols - 1 ? imga.cols - 1 : row + 1);
      j1 = (col == 0 ? 0 : col - 1);
      j2 = (col == imga.cols - 1 ? imga.cols - 1 : col + 1);
      p1 = ((float *)(imga.data + i1 * imga.step))[j1];
      p2 = ((float *)(imga.data + row * imga.step))[j1];
      p3 = ((float *)(imga.data + i2 * imga.step))[j1];
      p4 = ((float *)(imga.data + i1 * imga.step))[col];
      p5 = ((float *)(imga.data + row * imga.step))[col];
      p6 = ((float *)(imga.data + i2 * imga.step))[col];
      p7 = ((float *)(imga.data + i1 * imga.step))[j2];
      p8 = ((float *)(imga.data + row * imga.step))[j2];
      p9 = ((float *)(imga.data + i2 * imga.step))[j2];
      // edges
      s1 = (p1 + p2 * sq2 + p3 - p7 - p9 * sq2 - p9) / (2 * sq2);
      s2 = (p1 - p3 + p4 * sq2 - p6 * sq2 + p7 - p9) / (2 * sq2);
      s3 = (-p2 + p3 * sq2 + p4 - p6 - p7 * sq2 + p8) / (2 * sq2);
      s4 = (p1 * sq2 - p2 - p4 + p6 + p8 - p9 * sq2) / (2 * sq2);
      // lines
      s5 = (p2 - p4 - p6 + p8) / 2;
      s6 = (-p1 + p3 + p7 - p9) / 2;
      s7 = (p1 - p2 * 2 + p3 - p4 * 2 + p5 * 4 - p6 * 2 + p7 - p8 * 2 + p9) / 6;
      s8 =
          (-p1 * 2 + p2 - p3 * 2 + p4 + p5 * 4 + p6 - p7 * 2 + p8 - p9 * 2) / 6;
      // averaging
      s9 = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
      // sum squares
      s = s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6 + s7 * s7 +
          s8 * s8 + s9 * s9;
      // lines
      if (s != 0) s = (s5 * s5 + s6 * s6 + s7 * s7 + s8 * s8) / s;
      ((float *)(imgf.data + row * imgf.step))[col] = sqrt(s);
    }
  }
}

/*
 * Frei & Chen filter, new implementation
 * it seems not to work OK
 * http://www.roborealm.com/help/Frei_Chen.php
 */

void FrChn0(cv::Mat &imga, cv::Mat &imgf) {
  int i1, i2, j1, j2;
  float p1, p2, p3, p4, p5, p6, p7, p8, p9;
  float s, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      // replicate boundary conditions
      i1 = (row == 0 ? 0 : row - 1);
      i2 = (row == imga.cols - 1 ? imga.cols - 1 : row + 1);
      j1 = (col == 0 ? 0 : col - 1);
      j2 = (col == imga.cols - 1 ? imga.cols - 1 : col + 1);
      p1 = ((float *)(imga.data + i1 * imga.step))[j1];
      p2 = ((float *)(imga.data + row * imga.step))[j1];
      p3 = ((float *)(imga.data + i2 * imga.step))[j1];
      p4 = ((float *)(imga.data + i1 * imga.step))[col];
      p5 = ((float *)(imga.data + row * imga.step))[col];
      p6 = ((float *)(imga.data + i2 * imga.step))[col];
      p7 = ((float *)(imga.data + i1 * imga.step))[j2];
      p8 = ((float *)(imga.data + row * imga.step))[j2];
      p9 = ((float *)(imga.data + i2 * imga.step))[j2];
      // edges
      s1 = (2 * p1 + 3 * p2 + 2 * p3 - 2 * p7 - 3 * p8 - 2 * p9);
      s2 = (2 * p1 - 2 * p3 - 2 * p5 + 3 * p6 + 3 * p7 - 2 * p8);
      s3 = (3 * p1 - 3 * p3 + 2 * p4 - 2 * p6 - 2 * p7 + 2 * p9);
      s4 = (2 * p1 - 2 * p3 - 3 * p4 + 2 * p5 + 2 * p8 + 3 * p9);
      s = s1 + s2 + s3 + s4;
      ((float *)(imgf.data + row * imgf.step))[col] = sqrt(s);
    }
  }
}

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, img1, img2, img3,
      img4;              // initial, blurred, kernel, deblurred and noise image
  int m = 320, n = 240;  // image dimensions

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img, CV_32F, 1.0 / 255.0);
  } else {
    img = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img);
  }

  // img1 = cv::Mat(cv::Size(img.cols, img.rows),IPL_DEPTH_32F,1);
  img1 = img.clone();
  img1 = 0;
  Prewitt(img, img1);

  img2 = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  img2 = 0;
  FrChn(img, img2);

  img3 = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  img3 = 0;
  FrChn1(img, img3);

  img4 = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  img4 = 0;
  FrChn2(img, img4);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Prewitt", cv::WINDOW_AUTOSIZE);
  cv::imshow("Prewitt", img1);
  cv::namedWindow("Frei&Chen", cv::WINDOW_AUTOSIZE);
  cv::imshow("Frei&Chen", img2);
  cv::namedWindow("Frei&Chen edges", cv::WINDOW_AUTOSIZE);
  cv::imshow("Frei&Chen edges", img3);
  cv::namedWindow("Frei&Chen lines", cv::WINDOW_AUTOSIZE);
  cv::imshow("Frei&Chen lines", img4);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Prewitt");
  cv::destroyWindow("Frei&Chen");
  cv::destroyWindow("Frei&Chen edges");
  cv::destroyWindow("Frei&Chen lines");

  return 0;
}
