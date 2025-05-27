/*
 * Simple program to create
 * space variant motion blurr
 * Richardson Lucy algorithm
 * with deringing
 * Method from
 * Hatanaka et al.
 * IEICE Trans. Inf. & Syst. vol. E94-D, No 5, May 2011, pp. 1082-1089
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
 * Custom filter for edge detection
 */

void MHfilter(cv::Mat &imga, cv::Mat &imgf) {
  double mi1, mx1, mi2, mx2;
  cv::Mat imgb = imga.clone();
  Prewitt(imga, imgb);

  // original method
  // cvMinMaxLoc(imgb,&mi1,&mx1);
  // cv::subtractS(imgb, cvScalar(mi1),imgb);
  // cvScale(imgb,imgb,1.0/(mx1-mi1));

  // only the filter is better
  imgb.copyTo(imgf);

  // modified - seems to work better
  // cvThreshold(imgb,imgf,0.1,1.0,CV_THRESH_BINARY);
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

void BlurrPBCsv(cv::Mat &imga, cv::Mat &imgb, cv::Mat &imgc, int fl = 0,
                const cv::Mat &mask = cv::Mat()) {
  float s1, s2, s3;
  int i, j, fl2;

  for (int row = 0; row < imga.rows; row++) {
    // here the kernel is changed
    imgb = 0;
    if (fl == 0)
      ksetMoveXY(imgb, 5.0, 5.0);
    else
      ksetMoveXY(imgb, -5.0, -5.0);

    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;
      if (!mask.empty())
        fl2 = ((((uchar *)(mask.data + row * mask.step))[col]) > 0);
      else
        fl2 = 1;

      if (fl2) {
        for (int row1 = 0; row1 < imgb.rows; row1++) {
          for (int col1 = 0; col1 < imgb.cols; col1++) {
            s1 = ((float *)(imgb.data + row1 * imgb.step))[col1];

            // if (s1==0) continue;

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
      } else {
        ((float *)(imgc.data + row * imgc.step))[col] =
            ((float *)(imga.data + row * imga.step))[col];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, img1, img2, img3, img4, img5, img6,
      img7;  // initial, blurred, kernel, deblurred and noise image
  int m = 320, n = 240, r = 40,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int it;          // iteration counter
  const int x1 = 5, y1 = 5;                 // Define vector of motion blurr
  float norm, norm1, norm2, oldnorm, delt;  // norm of the images
  float oldnorm1, oldnorm2;
  float lambda;  // suspected image noise

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

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);

  // motion blurr
  // ksetMoveXY(kernel,x1,y1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, img1);

  // add some uniform noise
  // Unoise(img1, 1.00, 0.01);
  // Gnoise(img1, 0.0, 0.01);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", img1);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);

  // cvScale(img1,img1,255);
  // cv::imwrite( "Blurred.tif", img1);
  // cvScale(img1,img1,1.0/255.0);

  // Fourier derivation of the kernel
  img2 = img1.clone();
  img3 = img1.clone();
  img4 = img1.clone();
  img5 = img1.clone();
  // Laplacian regularization image
  img6 = img1.clone();
  img7 = img1.clone();

  // norm2=TotVar(img2)/((img2.cols)*(img2.rows));
  // norm2=cv::norm(img2, cv::NORM_L2)/((img2.cols)*(img2.rows));

  MHfilter(img4, img6);
  cv::imshow("Weights", img6);

  // cvScale(img6,img6,255);
  // cv::imwrite( "Edges.tif", img6);
  // cvScale(img6,img6,1.0/255.0);

  cv::subtract(img4, img1, img5);
  norm1 = cv::norm(img5, cv::NORM_L2) / ((img5.cols) * (img5.rows));

  // regularization parameter
  lambda = 0.01;

  oldnorm1 = norm1;
  oldnorm2 = norm2;
  oldnorm = norm1;

  // Richardson-Lucy starts here
  it = 0;
  do {
    // Mk=H*Ik
    BlurrPBCsv(img2, kernel, img4);
    // norm2=TotVar(img2)/((img2.cols)*(img2.rows));
    cv::subtract(img4, img1, img5);
    norm1 = cv::norm(img5, cv::NORM_L2) / ((img5.cols) * (img5.rows));

    // D/Mk
    cv::divide(img1, img4, img4);
    // Ht*(D/Mk)
    BlurrPBCsv(img4, kernel, img5, 1);

    // pixel by pixel multiply
    cv::multiply(img5, img2, img3);

    norm = norm1;
    delt = oldnorm - norm;

    oldnorm = norm;
    delt = ((it == 0) ? 1 : delt);
    delt = ((delt < 0) ? 0 : delt);

    img2 = img3.clone();

    // restoration
    for (int row = 0; row < img6.rows; row++) {
      for (int col = 0; col < img6.cols; col++) {
        float a1 = ((float *)(img6.data + row * img6.step))[col];
        float a2 = ((float *)(img3.data + row * img3.step))[col];
        float a3 = ((float *)(img1.data + row * img1.step))[col];
        ((float *)(img7.data + row * img7.step))[col] = (1 - a1) * a3 + a1 * a2;
      }
    }

    cv::imshow("Deblurred", img3);
    cv::imshow("Restored", img7);
    char c = cv::waitKey(10);

    if (c == 27) break;

    it++;
    cout << it << "  " << delt << "  " << norm << '\n';
  } while (abs(delt) > 1e-9);
  // while(it<2000);

  cv::waitKey(0);

  // cvScale(img3,img3,255);
  // cv::imwrite( "Deblurred.tif", img3);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred");
  cv::destroyWindow("Deblurred");

  return 0;
}
