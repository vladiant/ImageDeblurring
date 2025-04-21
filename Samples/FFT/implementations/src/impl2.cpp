/*
 *
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#include "impl2.hpp"

#include <string.h>

#include <iostream>

const double PI = 3.141592653589793238460;

// Cooley–Tukey FFT (in-place)
void fft(CArray& x) {
  const size_t N = x.size();
  if (N <= 1) return;

  // divide
  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd = x[std::slice(1, N / 2, 2)];

  // conquer
  fft(even);
  fft(odd);

  // combine
  for (size_t k = 0; k < N / 2; ++k) {
    Complex t = Complex(std::polar(1.0, -2 * PI * k / N)) * odd[k];
    x[k] = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}

// inverse fft (in-place)
void ifft(CArray& x) {
  // conjugate the complex numbers
  x = x.apply(std::conj);

  // forward fft
  fft(x);

  // conjugate the complex numbers again
  x = x.apply(std::conj);

  // scale the numbers
  x /= x.size();
}

void fft2d(float* x, int width, int height) {
  // Transform the rows.
  CArray rowArray(width);
  CArray colArray(height);
  for (int row = 0; row < height; row++) {
    memcpy(&rowArray[0], &x[2 * row * width], 2 * width * sizeof(float));
    fft(rowArray);
    memcpy(&x[2 * row * width], &rowArray[0], 2 * width * sizeof(float));
  }

  // Transform the columns.
  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      memcpy(&colArray[row], &x[2 * (col + row * width)], 2 * sizeof(float));
    }
    fft(colArray);
    for (int row = 0; row < height; row++) {
      memcpy(&x[2 * (col + row * width)], &colArray[row], 2 * sizeof(float));
    }
  }
}

void ifft2d(float* x, int width, int height) {
  // Transform the rows.
  CArray rowArray(width);
  CArray colArray(height);
  for (int row = 0; row < height; row++) {
    memcpy(&rowArray[0], &x[2 * row * width], 2 * width * sizeof(float));
    ifft(rowArray);
    memcpy(&x[2 * row * width], &rowArray[0], 2 * width * sizeof(float));
  }

  // Transform the columns.
  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      memcpy(&colArray[row], &x[2 * (col + row * width)], 2 * sizeof(float));
    }
    ifft(colArray);
    for (int row = 0; row < height; row++) {
      memcpy(&x[2 * (col + row * width)], &colArray[row], 2 * sizeof(float));
    }
  }
}
