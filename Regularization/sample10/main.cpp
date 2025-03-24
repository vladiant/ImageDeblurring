/*
 * Simple program to create
 * space variant motion blurr
 * Iteration algorithm
 * flexible Conjugate gradient
 * Polak–Ribière formula
 * with preconditioning,
 * reduction of ringing and noise
 * based on:
 * R. L Lagendijk, PhD thesis
 * Delft, Netherlands, 26 April 1990
 * Iterative Identification and Resoration of Images
 *
 * Laplacian regularized weighted CG iterations, modified
 * based on paper:
 * Adaptive iterative image restoration with reduced computational load
 * Serafim N. Efstratiadis, Aggelos K. Katsaggelos
 * OPTICAL ENGINEERING 29 (1990) 1458-1468
 *
 * - NVF V / NVF g (c & d)
 * - Non weighted (e & f)
 * - Variable Window
 * - Constant/Updated R and S (a & b)
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

/*
 * Gaussian distribution, taken from GSL
 * sigma i the standard deviation
 * mean is zero - add value if required
 */
double gaussian(const double sigma) {
  double x, y, r2;
  do {
    /* choose x,y in uniform square (-1,-1) to (+1,+1) */
    x = rand();
    y = rand();
    x /= RAND_MAX;
    y /= RAND_MAX;
    x = -1.0 + 2.0 * x;
    y = -1.0 + 2.0 * x;

    /* see if it is in the unit circle */
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0);

  /* Box-Muller transform */
  return sigma * y * sqrt(-2.0 * log(r2) / r2);
}

/*
 * Adds Gaussian noise with:
 * mean - mean value
 * stdev - standard deviation
 */
void Gnoise(cv::Mat &imgc, float mean = 0, float stdev = 0.01) {
  int row, col;  // row and column

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      ((float *)(imgc.data + row * imgc.step))[col] += gaussian(stdev) + mean;
    }
  }
}

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
 * S matrix set up
 */

void Sset(cv::Mat &imga, cv::Mat &imgf, float so2 = 1e-6) {
  int i, j, its, l;
  const int Nmin = 1, Nmax = 3;
  int x1 = 2,
      y1 = 2;        // size of the scan area - variable!!! initial values given
  bool fl1 = false;  // flag which controls window size change
  const float T1 = 0.2, T2 = 0.5;
  float s1, s2, s3, p1, smax = 0;
  const float mu = 0.5;  // tunable constant

  // initial calculations of the maximal values
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      s1 = 0;
      s2 = 0;
      s3 = 0;
      its = 0;

      for (int row1 = -y1; row1 < y1 + 1; row1++) {
        for (int col1 = -x1; col1 < x1 + 1; col1++, its++) {
          // replicate boundary conditions
          if ((row + row1) < 0)
            i = 0;
          else if ((row + row1) > (imga.rows - 1))
            i = imga.rows - 1;
          else
            i = row + row1;

          if ((col + col1) < 0)
            j = 0;
          else if ((col + col1) > (imga.cols - 1))
            j = imga.cols - 1;
          else
            j = col + col1;
          // weights
          s3 += exp(-0.5 * (row1 * row1 + col1 * col1) /
                    ((x1 + 1) * (x1 + 1) + (y1 + 1) * (y1 + 1)));
          s1 += ((float *)(imga.data + i * imga.step))[j];
        }
      }

      s1 /= its;

      for (int row1 = -y1; row1 < y1 + 1; row1++) {
        for (int col1 = -x1; col1 < x1 + 1; col1++) {
          // replicate boundary conditions
          if ((row + row1) < 0)
            i = 0;
          else if ((row + row1) > (imga.rows - 1))
            i = imga.rows - 1;
          else
            i = row + row1;

          if ((col + col1) < 0)
            j = 0;
          else if ((col + col1) > (imga.cols - 1))
            j = imga.cols - 1;
          else
            j = col + col1;
          p1 = ((float *)(imga.data + i * imga.step))[j];
          // weights
          float a1 = exp(-0.5 * (row1 * row1 + col1 * col1) /
                         ((x1 + 1) * (x1 + 1) + (y1 + 1) * (y1 + 1)));
          s2 += (p1 - s1) * (p1 - s1) * a1;  //*a1; //weights applied
        }
      }

      // NVF G
      // s2=((s2-so2)>0)?s2-so2:0;
      s2 = ((s2 / s3 - so2) > 0) ? s2 - so2 : 0;

      if (s2 > smax) smax = s2;

      ((float *)(imgf.data + row * imgf.step))[col] = s2;
    }
  }

  // variable window sizes
  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      s1 = 0;
      s2 = 0;
      s3 = 0;
      its = 0;

      // varying the size of the scan area
      x1 = 2;
      y1 = 2;

      do {
        fl1 = false;

        for (int row1 = -y1; row1 < y1 + 1; row1++) {
          for (int col1 = -x1; col1 < x1 + 1; col1++, its++) {
            // replicate boundary conditions
            if ((row + row1) < 0)
              i = 0;
            else if ((row + row1) > (imga.rows - 1))
              i = imga.rows - 1;
            else
              i = row + row1;

            if ((col + col1) < 0)
              j = 0;
            else if ((col + col1) > (imga.cols - 1))
              j = imga.cols - 1;
            else
              j = col + col1;
            // weights
            s3 += exp(-0.5 * (row1 * row1 + col1 * col1) /
                      ((x1 + 1) * (x1 + 1) + (y1 + 1) * (y1 + 1)));
            s1 += ((float *)(imga.data + i * imga.step))[j];
          }
        }

        s1 /= its;

        for (int row1 = -y1; row1 < y1 + 1; row1++) {
          for (int col1 = -x1; col1 < x1 + 1; col1++) {
            // replicate boundary conditions
            if ((row + row1) < 0)
              i = 0;
            else if ((row + row1) > (imga.rows - 1))
              i = imga.rows - 1;
            else
              i = row + row1;

            if ((col + col1) < 0)
              j = 0;
            else if ((col + col1) > (imga.cols - 1))
              j = imga.cols - 1;
            else
              j = col + col1;
            p1 = ((float *)(imga.data + i * imga.step))[j];
            // weights
            float a1 = exp(-0.5 * (row1 * row1 + col1 * col1) /
                           ((x1 + 1) * (x1 + 1) + (y1 + 1) * (y1 + 1)));
            s2 += (p1 - s1) * (p1 - s1) * a1;  //*a1; //weights applied
          }
        }

        // NVF G
        // s2=((s2-so2)>0)?s2-so2:0;
        s2 = ((s2 / s3 - so2) > 0) ? s2 - so2 : 0;

        ((float *)(imgf.data + row * imgf.step))[col] = s2;

        if (s2 > (smax * T1))
          if (s2 > (smax * T2))
            l = 3;
          else
            l = 2;
        else
          l = 1;

        if (x1 != (Nmax + 1 - l)) {
          x1 = Nmax + 1 - l;
          fl1 = true;
        }

        if (y1 != (Nmax + 1 - l)) {
          y1 = Nmax + 1 - l;
          fl1 = true;
        }

      } while (not(fl1));
    }
  }

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      float a1 = ((float *)(imgf.data + row * imgf.step))[col];
      // a1=(a1>so2)?a1:so2;
      ((float *)(imgf.data + row * imgf.step))[col] = 1.0 / (1.0 + mu * a1);
    }
  }
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
        s5 = 0.5 * erfc((col - 20) / 2.0);
        // cout << s5 << '\n';
        s4 += s5;
        ((float *)(imgb.data + 0 * imgb.step))[col + imgb.cols / 2] = s5;
      }
      imgb = imgb * 1.0 / s4;
      // cv::imshow("test", imgb);
      // cv::waitKey(0);
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
        s5 = 0.5 * erfc((abs(col) - 20) / 2.0);
        s4 += s5;
        // cout << s5 << '\n';
        ((float *)(imgb.data + 0 * imgb.step))[col + imgb.cols / 2] = s5;
      }
      imgb = imgb * 1.0 / s4;
      cv::imshow("test", imgb);
      // cv::waitKey(0);
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

/*
 * Function to calculate
 * the standard deviation
 * of the noise
 */

float NoiseDev(const cv::Mat &imga) {
  cv::Mat imgd = cv::Mat(imga.rows, imga.cols, CV_32FC1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  imgd = imga * 1.0 / 255.0;
  cv::dft(imgd, imgd, cv::DFT_COMPLEX_OUTPUT);
  for (int row = int(shar * imgd.rows / 2); row < (imgd.rows / 2) - 1; row++) {
    for (int col = int(shar * imgd.cols / 2); col < (imgd.cols / 2) - 1;
         col++) {
      re = imgd.at<std::complex<float> >(row, col).real();
      im = imgd.at<std::complex<float> >(row, col).imag();
      sum += sqrt(re * re + im * im);
    }
  }
  sum /= ((imgd.rows / 2) - int(shar * imgd.rows / 2)) *
         ((imgd.cols / 2) - int(shar * imgd.cols / 2)) *
         sqrt((imgd.rows) * (imgd.cols));
  return (sum);
}

int main(int argc, char *argv[]) {
  // initial two images, blurred, deblurred, residual and image, blurred
  // preconditioned image, temp vector, regularization, weights
  cv::Mat img, imgi, imgb, imgx, imgr, imgp, imgap, imgr0, imgl, imgls, imglr;
  int m = 320, n = 240,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int it;          // iteration counter
  const int x1 = 25, y1 = 0;  // Define vector of motion blurr
  double gk, bk, norm, rs,
      alpha;                 // Temp for dot product, regularization coefficient
  const float regul = 1500;  // regularization constant
  float stdev, isnr, isnr0;  // Noise standard deviation, ISNR
  float s1 = 0, gam = 0;     // Temporal variable, alpha determination constant

  // creates initial image
  if ((argc >= 2) &&
      !(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty()) {
    imgi.convertTo(img, CV_32F);
    img = img * 1.0 / 255.0;
  } else {
    img = cv::Mat(n, m, CV_32FC1);
    Ximgset(img);
  }

  imgb = img.clone();

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel = cv::Mat(1, 2 * ksize + 1, CV_32FC1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, imgb);

  // average value of the blurred image pixels
  s1 = 0;
  for (int row = 0; row < imgb.rows; row++)
    for (int col = 0; col < imgb.cols; col++) {
      float a1 = ((float *)(imgb.data + row * imgb.step))[col];
      s1 += a1;
    }
  s1 /= (imgb.cols) * (imgb.rows);

  cout << "Average image value:  " << s1 << '\n';

  if (argc == 3) {
    stdev = atof(argv[2]);
    // add some salt and pepper noise
    // SPnoise(imgb, stdev);

    // add some gauss noise
    Gnoise(imgb, 0.0, stdev);

    // add some poisson noise
    // Pnoise(imgb, 0.01*50000);

    // add some speckle noise
    // SPEnoise(imgb, 0.1);

    // add some uniform noise
    // Unoise(imgb, 1.00, 0.01);
  }

  stdev = NoiseDev(imgb) * regul / 3.12;

  // blurred SNR
  s1 = s1 / stdev;

  // alpha=(6.67533*stdev*stdev+0.00089085*stdev+2e-6);
  // cout << "Alpha:  " << alpha << '\n';

  s1 = 10 * log(s1) / 2.302585093;
  cout << "BSNR:  " << s1 << '\n';

  // upper part of ISNR
  isnr0 = cv::norm(imgb, img, cv::NORM_L2);

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", imgb);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);

  // initial approximation of the restored image
  imgx = imgb.clone();

  // regularization parameters
  imgl = imgb.clone();
  // weights
  imgls = imgb.clone();
  imglr = imgb.clone();
  Sset(imgb, imgls, stdev * stdev);

  // set R matrix based on paper
  for (int row = 0; row < imglr.rows; row++)
    for (int col = 0; col < imglr.cols; col++) {
      ((float *)(imglr.data + row * imglr.step))[col] =
          1.0;  //-((float*)(imgls.data + row*imgls.step))[col];
    }

  cv::namedWindow("Weight S", cv::WINDOW_AUTOSIZE);
  cv::imshow("Weight S", imgls);
  cv::namedWindow("Weight R", cv::WINDOW_AUTOSIZE);
  cv::imshow("Weight R", imglr);

  // initial approximation of the residual
  imgr = imgb.clone();
  BlurrPBCsv(imgx, kernel, imgr);
  cv::subtract(imgb, imgr, imgr);

  BlurrPBCsv(imgr, kernel, imgr, 1);

  // add regularization
  cv::Laplacian(imgb, imgl, -1);
  // weights
  cv::Laplacian(imgl, imgl, -1);

  // alpha as proposed in this paper
  alpha = 0.00002;  // regularization causes ringing here!!!

  imgl = imgl * alpha;
  cv::subtract(imgr, imgl, imgr);

  imgr0 = imgr.clone();
  imgr0 = 0;

  // initial approximation of preconditioner
  imgp = imgr.clone();

  // initial approximation of preconditioned blurred image
  imgap = imgb.clone();

  // reset iteration counter
  it = 0;

  do {
    // gamma_k initial
    gk = 1.0 / imgr.dot(imgr);

    // modification - improves convergence
    imgr.copyTo(imgr0);

    // r_k
    imgr = imgb.clone();

    BlurrPBCsv(imgx, kernel, imgr);
    cv::subtract(imgb, imgr, imgr);

    // weights
    cv::multiply(imgr, imglr, imgr);

    BlurrPBCsv(imgr, kernel, imgr, 1);

    // add regularization
    cv::Laplacian(imgx, imgl, -1);

    // weights
    cv::multiply(imgl, imgls, imgl);

    cv::Laplacian(imgl, imgl, -1);
    imgl = imgl * alpha;
    cv::subtract(imgr, imgl, imgr);

    // modification - improves convergence
    cv::subtract(imgr, imgr0, imgr0);

    // gamma_k final - modified to improve convergence
    gk *= imgr.dot(imgr0);  // original gk*=imgr.dot(imgr);

    // p_k
    cv::addWeighted(imgr, 1.0, imgp, gk, 0.0, imgp);

    // beta_k
    BlurrPBCsv(imgp, kernel, imgap);

    // regularization
    cv::Laplacian(imgp, imgl, -1);
    // bk = 1.0 / (imgap.dot(imgap) + alpha * imgl.dot(imgl));

    bk = 0;
    for (int row = 0; row < imgx.rows; row++) {
      for (int col = 0; col < imgx.cols; col++) {
        float a1 = ((float *)(imgl.data + row * imgl.step))[col];
        float a2 = ((float *)(imgls.data + row * imgls.step))[col];
        float a3 = ((float *)(imgap.data + row * imgap.step))[col];
        float a4 = ((float *)(imglr.data + row * imglr.step))[col];
        // weighted method
        bk += alpha * a1 * a1 * a2 + a3 * a3 * a4;
      }
    }
    bk = 1.0 / bk;

    bk *= imgr.dot(imgp);

    // stopping coefficient
    float s1 = 0, s2 = 0;
    for (int row = 0; row < imgx.rows; row++)
      for (int col = 0; col < imgx.cols; col++) {
        float a1 = ((float *)(imgx.data + row * imgx.step))[col];
        float a2 = ((float *)(imgp.data + row * imgp.step))[col];
        s1 += a1 * (bk * a2) * a1 * (bk * a2);
        s2 += a1 * a1;
      }
    norm = (s2 != 0) ? s1 / s2 : 0;

    // ISNR
    isnr = 10 * log(isnr0 / cv::norm(img, imgx, cv::NORM_L2)) / 2.302585093;

    // x_k
    cv::addWeighted(imgx, 1.0, imgp, bk, 0.0, imgx);

    // S matrix updated
    Sset(imgx, imgls, stdev * stdev);
    cv::imshow("Weight S", imgls);

    /*
    //R matrix updated
    for(int row=0;row<imglr.rows;row++)
            for(int col=0;col<imglr.cols;col++)
                    {
                            ((float*)(imglr.data +
    row*imglr.step))[col]=1.0-((float*)(imgls.data +
    row*imgls.step))[col];
                    }
    cvShowImage("Weight R", imglr);
    */

    cv::imshow("Deblurred", imgx);

    char c = cv::waitKey(10);
    if (c == 27) break;

    it++;
    cout << it << " norm " << norm << " isnr " << isnr << " alpha " << alpha
         << '\n';
  } while ((norm > 1e-8));  //&&(it<301)

  imgx = imgx * 255;
  // cvSaveImage("Deblurred.tif",imgx);
  imgb = imgb * 255;
  // cvSaveImage("Blurred.tif",imgb);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred");
  cv::destroyWindow("Deblurred");

  return 0;
}