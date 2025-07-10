/*
 * Simple program to create
 * to split image color channels
 * and to test edge detection
 * with Prewitt and
 * Frei & Chen filters
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;

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
  cv::Mat img, imgi, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;
  // filter temporal images
  cv::Mat imgc1f, imgc2f, imgc3f;
  // filter display images
  cv::Mat imgfp, imgffc, imgffc1, imgffc2;
  double sca = 1.0;
  char *nmP, *nmFC, *nmFC1, *nmFC2;
  int i;

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
            .empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

    switch (imgi.depth()) {
      case CV_8U:
        sca = 255;
        break;

      case CV_16U:
        sca = 65535;
        break;

      case CV_32S:
        sca = 4294967295;
        break;

      default:  // unknown depth, program should go on
        sca = 1.0;
    }

    // displays image
    cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
    cv::imshow("Initial", imgi);

    imgc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgc1f = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

    if (imgi.channels() != 1) {
      imgc1i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc2i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc3i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      imgc3 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      imgc2f = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      imgc3f = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

      imgfp = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);
      imgffc = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);
      imgffc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);
      imgffc2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);

      cv::split(imgi, std::vector{imgc1i, imgc2i, imgc3i});
      imgc1i.convertTo(imgc1, CV_32F, 1.0 / sca);
      imgc2i.convertTo(imgc2, CV_32F, 1.0 / sca);
      imgc3i.convertTo(imgc3, CV_32F, 1.0 / sca);

      // filtering
      Prewitt(imgc1, imgc1f);
      Prewitt(imgc2, imgc2f);
      Prewitt(imgc3, imgc3f);
      cv::merge(std::vector{imgc1f, imgc2f, imgc3f}, imgfp);

      FrChn(imgc1, imgc1f);
      FrChn(imgc2, imgc2f);
      FrChn(imgc3, imgc3f);
      cv::merge(std::vector{imgc1f, imgc2f, imgc3f}, imgffc);

      FrChn1(imgc1, imgc1f);
      FrChn1(imgc2, imgc2f);
      FrChn1(imgc3, imgc3f);
      cv::merge(std::vector{imgc1f, imgc2f, imgc3f}, imgffc1);

      FrChn2(imgc1, imgc1f);
      FrChn2(imgc2, imgc2f);
      FrChn2(imgc3, imgc3f);
      cv::merge(std::vector{imgc1f, imgc2f, imgc3f}, imgffc2);

    } else {
      imgi.convertTo(imgc1, CV_32F, 1.0 / sca);

      imgfp = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      Prewitt(imgc1, imgfp);

      imgffc = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      FrChn(imgc1, imgffc);

      imgffc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      FrChn1(imgc1, imgffc1);

      imgffc2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      FrChn2(imgc1, imgffc2);
    }

    // cout << "Image Channels:  " << imgi.channels() << '\n';

    nmP = new char[strlen(argv[1]) + 2];
    nmFC = new char[strlen(argv[1]) + 2];
    nmFC1 = new char[strlen(argv[1]) + 2];
    nmFC2 = new char[strlen(argv[1]) + 2];

    i = 0;

    do {
      *(nmP + i) = *(argv[1] + i);
      *(nmFC + i) = *(argv[1] + i);
      *(nmFC1 + i) = *(argv[1] + i);
      *(nmFC2 + i) = *(argv[1] + i);
      i++;
    } while (*(argv[1] + i) != '.');

    *(nmP + i) = '_';
    *(nmFC + i) = '_';
    *(nmFC1 + i) = '_';
    *(nmFC2 + i) = '_';

    *(nmP + i + 1) = 'P';
    *(nmFC + i + 1) = 'F';
    *(nmFC1 + i + 1) = 'F';
    *(nmFC2 + i + 1) = 'F';

    *(nmP + i + 2) = 'R';
    *(nmFC + i + 2) = '0';
    *(nmFC1 + i + 2) = '1';
    *(nmFC2 + i + 2) = '2';

    do {
      *(nmP + i + 3) = *(argv[1] + i);
      *(nmFC + i + 3) = *(argv[1] + i);
      *(nmFC1 + i + 3) = *(argv[1] + i);
      *(nmFC2 + i + 3) = *(argv[1] + i);
      i++;
    } while (*(argv[1] + i - 1) != '\0');

    /*
    cout <<'"' << nmP << '"'<< '\n';
    cout <<'"' << nmFC << '"'<< '\n';
    cout <<'"' << nmFC1 << '"'<< '\n';
    cout <<'"' << nmFC2 << '"'<< '\n';
    */

    cv::namedWindow("Prewitt Filter", cv::WINDOW_AUTOSIZE);
    cv::imshow("Prewitt Filter", imgfp);
    cv::namedWindow("Frei&Chen", cv::WINDOW_AUTOSIZE);
    cv::imshow("Frei&Chen", imgffc);
    cv::namedWindow("Frei&Chen edges", cv::WINDOW_AUTOSIZE);
    cv::imshow("Frei&Chen edges", imgffc1);
    cv::namedWindow("Frei&Chen lines", cv::WINDOW_AUTOSIZE);
    cv::imshow("Frei&Chen lines", imgffc2);

    cv::waitKey(0);

    cv::destroyWindow("Initial");
    cv::destroyWindow("Prewitt Filter");
    cv::destroyWindow("Frei&Chen");
    cv::destroyWindow("Frei&Chen edges");
    cv::destroyWindow("Frei&Chen lines");

    /*
    //save images
    cvScale(imgfp,imgfp,255);
    cv::imwrite(nmP,imgfp);
    cvScale(imgffc,imgffc,255);
    cv::imwrite(nmFC,imgffc);
    cvScale(imgffc1,imgffc1,255);
    cv::imwrite(nmFC1,imgffc1);
    cvScale(imgffc2,imgffc2,255);
    cv::imwrite(nmFC2,imgffc2);
    */

    delete[] nmP;
    delete[] nmFC;
    delete[] nmFC1;
    delete[] nmFC2;
  }

  // img1 = cvCloneImage(img);

  return 0;
}