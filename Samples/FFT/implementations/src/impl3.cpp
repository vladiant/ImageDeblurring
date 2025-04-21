/*
 *
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#include "impl3.hpp"

#include <math.h>
#include <stdlib.h>

/*-------------------------------------------------------------------------
 Perform a 2D FFT inplace given a complex 2D array
 The direction dir, 1 for forward, -1 for reverse
 The size of the array (nx,ny)
 Return false if there are memory problems or
 the dimensions are not powers of 2
 */
bool FFT2D_(float *c, int nx, int ny, int dir) {
  int i, j;
  int m, twopm;

  /* Transform the rows */
  if (!Powerof2_(nx, &m, &twopm) || twopm != nx) return false;
  for (j = 0; j < ny; j++) {
    FFT(dir, m, &c[2 * (j * nx) + 0], &c[2 * (j * nx) + 1], 2);
  }

  /* Transform the columns */
  if (!Powerof2_(ny, &m, &twopm) || twopm != ny) return false;
  for (i = 0; i < nx; i++) {
    FFT(dir, m, &c[2 * i + 0], &c[2 * i + 1], 2 * nx);
  }

  return true;
}

/*-------------------------------------------------------------------------
 This computes an in-place complex-to-complex FFT
 x and y are the real and imaginary arrays of 2^m points.
 dir =  1 gives forward transform
 dir = -1 gives reverse transform

 *     Formula: forward
 *                 N-1
 *                 ---
 *             1   \          - j k 2 pi n / N
 *     X(n) = ---   >   x(k) e                    = forward transform
 *             N   /                                n=0..N-1
 *                 ---
 *                 k=0
 *
 *     Formula: reverse
 *                 N-1
 *                 ---
 *                 \          j k 2 pi n / N
 *     X(n) =       >   x(k) e                    = forward transform
 *                 /                                n=0..N-1
 *                 ---
 *                 k=0

 */
bool FFT(int dir, int m, float *x, float *y, int step) {
  long nn, i, i1, j, k, i2, l, l1, l2;
  float c1, c2, tx, ty, t1, t2, u1, u2, z;

  /* Calculate the number of points */
  nn = 1;
  for (i = 0; i < m; i++) nn *= 2;

  /* Do the bit reversal */
  i2 = nn >> 1;
  j = 0;
  for (i = 0; i < nn - 1; i++) {
    if (i < j) {
      tx = x[i * step];
      ty = y[i * step];
      x[i * step] = x[j * step];
      y[i * step] = y[j * step];
      x[j * step] = tx;
      y[j * step] = ty;
    }
    k = i2;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  /* Compute the FFT */
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l = 0; l < m; l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j = 0; j < l1; j++) {
      for (i = j; i < nn; i += l2) {
        i1 = i + l1;
        t1 = u1 * x[i1 * step] - u2 * y[i1 * step];
        t2 = u1 * y[i1 * step] + u2 * x[i1 * step];
        x[i1 * step] = x[i * step] - t1;
        y[i1 * step] = y[i * step] - t2;
        x[i * step] += t1;
        y[i * step] += t2;
      }
      z = u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1) c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  /* Scaling for forward transform */
  if (dir == -1) {
    for (i = 0; i < nn; i++) {
      x[i * step] /= (float)nn;
      y[i * step] /= (float)nn;
    }
  }

  return true;
}

/*-------------------------------------------------------------------------
 Calculate the closest but lower power of two of a number
 twopm = 2**m <= n
 Return true if 2**m == n
 */
bool Powerof2_(int n, int *m, int *twopm) {
  if (n <= 1) {
    *m = 0;
    *twopm = 1;
    return false;
  }

  *m = 1;
  *twopm = 2;
  do {
    (*m)++;
    (*twopm) *= 2;
  } while (2 * (*twopm) <= n);

  if (*twopm != n)
    return false;
  else
    return true;
}
