#include "noise.hpp"

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

double poisson(const double lambda) {
  double L, k, p, rn;

  L = exp(-lambda);
  k = 0;
  p = 1;

  do {
    k++;
    rn = rand();
    rn /= RAND_MAX;
    p *= rn;
  } while (p > L);

  return k - 1;
}

void SPnoise(cv::Mat &imgc, float pr) {
  int row, col;   // row and column
  float rn, rn1;  // for random generator

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      rn = rand();
      rn /= RAND_MAX;
      rn1 = rand();
      rn1 /= RAND_MAX;
      if (rn < pr)
        if (rn1 < 0.5) {
          ((float *)(imgc.data + row * imgc.step))[col] = 0.0;
        } else {
          ((float *)(imgc.data + row * imgc.step))[col] = 1.0;
        }
    }
  }
}

void Gnoise(cv::Mat &imgc, float mean, float stdev) {
  int row, col;  // row and column

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      ((float *)(imgc.data + row * imgc.step))[col] += gaussian(stdev) + mean;
    }
  }
}

void LVnoise(cv::Mat &imgc, cv::Mat &imgm) {
  int row, col;  // row and column

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      {
        ((float *)(imgc.data + row * imgc.step))[col] +=
            gaussian(((float *)(imgm.data + row * imgm.step))[col]);
      }
    }
  }
}

void LV1noise(cv::Mat &imgc, float (*ptLV1f)(float)) {
  int row, col;  // row and column

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      {
        ((float *)(imgc.data + row * imgc.step))[col] +=
            gaussian((*ptLV1f)(((float *)(imgc.data + row * imgc.step))[col]));
      }
    }
  }
}

void Pnoise(cv::Mat &imgc, float scl) {
  int row, col;  // row and column

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      ((float *)(imgc.data + row * imgc.step))[col] =
          poisson(((float *)(imgc.data + row * imgc.step))[col] * scl) / scl;
    }
  }
}

void SPEnoise(cv::Mat &imgc, float stdev) {
  int row, col;  // row and column
  float rn;      // for random generator

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      rn = rand();
      rn /= RAND_MAX;
      ((float *)(imgc.data + row * imgc.step))[col] +=
          rn * stdev * ((float *)(imgc.data + row * imgc.step))[col];
    }
  }
}

void Unoise(cv::Mat &imgc, float pr, float v) {
  int row, col;   // row and column
  float rn, rn1;  // for the random generator

  srand(time(0));  // initialize random number generator

  for (int row = 0; row < imgc.rows; row++) {
    for (int col = 1; col < imgc.cols; col++) {
      rn = rand();
      rn /= RAND_MAX;
      rn1 = rand();
      rn1 /= RAND_MAX;
      if (rn < pr) {
        ((float *)(imgc.data + row * imgc.step))[col] += rn1 * v;
      }
    }
  }
}
