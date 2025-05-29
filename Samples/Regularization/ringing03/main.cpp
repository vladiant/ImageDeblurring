/*
 * Simple program to create
 * space variant motion blurr
 * Richardson Lucy algorithm
 * with deringing
 * Method from
 * Post-Processing Algorithm for Reducing Ringing Artefacts in Deblurred Images
 * Sergey Chalkov, Natalie Meshalkina, Chnag-Su Kim
 * ITC-CSCC 2008, pp. 1193-1196
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
 * Function which creates the masks
 * Mask 1 - Edges detection & Dilation with kernel size
 * Mask 2 - Mask 1 without edges
 * Mask 3 - Mask 2 without
 */

void BlurMask(const cv::Mat &imga, const cv::Mat &kernel, cv::Mat &imgm,
              cv::Mat &imgm1, cv::Mat &imgm2) {
  int N = 7;               // kernel size of Canny edge detector
  double lowThresh = 200;  // to be adjusted if needed
  double highThresh = 300;
  cv::Mat kmat =
      getStructuringElement(cv::MORPH_RECT, cv::Size(kernel.cols, kernel.rows));
  cv::Mat tmp, tmp1;
  float vals[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
  cv::Mat kern(3, 3, CV_32FC1, vals);
  cv::Mat imgb = cv::Mat(cv::Size(imga.cols, imga.rows), CV_8UC1);
  cv::Mat imgc = cv::Mat(cv::Size(imga.cols + N, imga.rows + N), CV_8UC1);
  cv::Mat imgd = cv::Mat(cv::Size(imga.cols + N, imga.rows + N), CV_8UC1);

  cv::Mat imge = cv::Mat(cv::Size(imga.cols + N, imga.rows + N), CV_32FC1);

  imga.convertTo(imgb, CV_8U);
  imgb = imgb * 255;
  // add convolution borders
  cv::copyMakeBorder(imgb, imgc, N / 2, N / 2, N / 2, N / 2,
                     cv::BORDER_REPLICATE);
  cv::Canny(imgc, imgc, lowThresh * N * N, highThresh * N * N, N);

  // distance transform
  imge = imgc / 255;
  imgc = ~imgc;
  cv::distanceTransform(imgc, imge, cv::DIST_L1, 3);
  imge = imge / 128.0;
  cv::imshow("Dist", imge);
  imgc = ~imgc;

  // for the second mask
  imgc.copyTo(imgd);

  // increase the edge area
  cv::dilate(imgc, imgc, kmat);

  // remove borders and clear the memory
  imgm = imgc(cv::Rect(N / 2, N / 2, imgm.cols, imgm.rows)).clone();

  // prepare second mask
  cv::subtract(imgc, imgd, imgd);
  imgm1 = imgd(cv::Rect(N / 2, N / 2, imgm1.cols, imgm1.rows)).clone();

  // prepare third mask
  // cvSmooth(imgb,imgb, CV_GAUSSIAN, N, N);
  cv::filter2D(imgb, imgb, -1, kern);
  cv::threshold(imgb, imgb, 2, 255, cv::THRESH_BINARY);
  for (int row = 0; row < imgb.rows; row++) {
    for (int col = 0; col < imgb.cols; col++) {
      char a1 = ((uchar *)(imgm.data + row * imgm.step))[col];
      if (a1 != 0) ((uchar *)(imgb.data + row * imgb.step))[col] = 0;
    }
  }
  imgb.copyTo(imgm2);
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
  cv::Mat img, imgi, img1, img2, img3, img4, img5, img6, img7, img8,
      img9;  // initial, blurred, kernel, deblurred and noise image
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
  img9 = img1.clone();
  img9 = 0;
  // Laplacian regularization image
  img6 = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC1);
  img7 = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC1);
  img8 = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC1);

  // norm2=TotVar(img2)/((img2.cols)*(img2.rows));
  // norm2=cv::norm( img2, 0, cv::NORM_L2 )/((img2.cols)*(img2.rows));

  BlurMask(img4, kernel, img6, img7, img8);
  cv::imshow("Mask1", img6);
  cv::imshow("Mask2", img7);
  cv::imshow("Mask3", img8);

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
        float a1 = ((uchar *)(img6.data + row * img6.step))[col] / 255;
        float a2 = ((float *)(img3.data + row * img3.step))[col];
        float a3 = ((float *)(img1.data + row * img1.step))[col];
        float a4 = ((uchar *)(img8.data + row * img8.step))[col] / 255;
        ((float *)(img9.data + row * img9.step))[col] = (1 - a1) * a3 + a1 * a2;
        if (a4 != 0) {
          float a5 = ((float *)(img9.data + row * img9.step))[col];
          // select the coefficient for fusion
          ((float *)(img9.data + row * img9.step))[col] = a2 * 0.9 + a3 * 0.1;
        }
      }
    }

    cv::imshow("Deblurred", img3);
    cv::imshow("Restored", img9);
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
