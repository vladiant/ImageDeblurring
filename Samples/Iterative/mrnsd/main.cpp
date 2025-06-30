/*
 * Simple program to create
 * space variant motion blurr
 * Iteration algorithm
 * Modified Residual Norm Steepest Descent
 * MRNSD
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
 * This is procedure for setting of the move kernel
 * (x-x0,y-y0) defines the vector of movement
 * move was assumed with constant speed.
 */
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
      ksetMoveXY(imgb, 5, 5);
    else
      ksetMoveXY(imgb, -5, -5);

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
  // preconditioned images, temp vector
  cv::Mat img, imgi, imgb, imgx, imgr, imgp, imgap, imgz, imgr0;
  // regularization, weights image
  cv::Mat imgl, imgls;
  int m = 320, n = 240,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int it;          // iteration counter
  const int x1 = 5, y1 = 5;  // Define vector of motion blurr
  double gk, bk, norm, minp,
      alpha;  // Temp for dot product, regularization coefficient

  // test the median filter against ringing
  // cv::Mat& imgmf, *imgmf1;

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

  /*
  //test images for median filter
  imgmf=cv::Mat(cv::Size(img.cols, img.rows),IPL_DEPTH_8U,1);
  imgmf1=cv::Mat(cv::Size(img.cols, img.rows),IPL_DEPTH_8U,1);
  */

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, imgb);

  // add some noise
  // Unoise(imgb, 1.00, 0.1);
  // Gnoise(imgb, 0.0, 0.01);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", imgb);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);

  // initial approximation of the restored image
  imgx = imgb.clone();

  // regularization parameters
  imgl = imgb.clone();  // remove it

  // initial approximation of the residual
  imgr = imgb.clone();
  BlurrPBCsv(imgx, kernel, imgr);
  cv::subtract(imgb, imgr, imgr);
  BlurrPBCsv(imgr, kernel, imgr, 1);

  gk = 0;
  for (int row = 0; row < imgr.rows; row++) {
    for (int col = 0; col < imgr.cols; col++) {
      float a1 = ((float *)(imgr.data + row * imgr.step))[col];
      float a2 = ((float *)(imgx.data + row * imgx.step))[col];
      gk += a1 * a1 * a2;
    }
  }

  // residual change
  imgr0 = imgr.clone();
  imgr0 = 0;

  // initial approximation of preconditioner
  imgp = imgr.clone();

  // initial approximation of preconditioned blurred image
  imgap = imgb.clone();

  // initial approximation of preconditioned residual
  imgz = imgb.clone();

  // reset iteration counter
  it = 0;

  do {
    // p_k
    for (int row = 0; row < imgp.rows; row++) {
      for (int col = 0; col < imgp.cols; col++) {
        float a1 = ((float *)(imgr.data + row * imgr.step))[col];
        float a2 = ((float *)(imgx.data + row * imgx.step))[col];
        ((float *)(imgp.data + row * imgp.step))[col] = -a1 * a2;
      }
    }

    // u_k
    BlurrPBCsv(imgp, kernel, imgap);

    // b_k
    minp = 1e22;
    for (int row = 0; row < imgp.rows; row++) {
      for (int col = 0; col < imgp.cols; col++) {
        float a1 = ((float *)(imgp.data + row * imgp.step))[col];
        float a2 = ((float *)(imgr.data + row * imgr.step))[col];
        // if ((row==1)&&(col==1)) minp=a2;
        if (a1 < 0)
          if (1.0 < (a2 * minp)) minp = a2;
      }
    }
    bk = min(gk / imgap.dot(imgap), minp);
    // bk=gk/imgap.dot(imgap);

    // x_k
    cv::addWeighted(imgx, 1.0, imgp, bk, 0.0, imgx);

    // z_k
    BlurrPBCsv(imgap, kernel, imgz, 1);

    // r_k
    // cvCopy(imgr,imgr0);
    cv::addWeighted(imgr, 1.0, imgz, bk, 0.0, imgr);

    // g_k
    // cv::subtract(imgr,imgr0,imgr0);
    gk = 0;
    for (int row = 0; row < imgr.rows; row++) {
      for (int col = 0; col < imgr.cols; col++) {
        float a1 = ((float *)(imgr.data + row * imgr.step))[col];
        // float a1=((float*)(imgr0.data + row*imgr0.step))[col];
        float a2 = ((float *)(imgx.data + row * imgx.step))[col];
        gk += a1 * a1 * a2;
      }
    }

    // here check if |imgr| is sufficiently small
    // norm calculation
    norm = sqrt(cv::norm(imgr, cv::NORM_L2)) / ((imgr.cols) * (imgr.rows));

    cv::imshow("Deblurred", imgx);

    /*
    //test for median filtering
    cvScale(imgx,imgmf,255);
    cvSmooth(imgmf,imgmf1,CV_MEDIAN, 3, 3);
    cv::imshow("Deblurred MF", imgmf1);
    */

    char c = cv::waitKey(10);
    if (c == 27) break;

    it++;
    cout << it << "  " << norm << "  "
         << cv::norm(imgr, cv::NORM_L2) / cv::norm(imgx, cv::NORM_L2) << '\n';
  } while (norm > 1e-6);  // 5e-6 without noise; value depends on noise
                          // deviation; typical 1.5e-5

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred");
  cv::destroyWindow("Deblurred");

  return 0;
}