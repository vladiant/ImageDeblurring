/*
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#pragma once

struct complex_t {
  float real;
  float imag;
};

bool FFT(int dir, int m, float *x, float *y);
bool FFT2D(float *c, int nx, int ny, int dir);
bool Powerof2(int n, int *m, int *twopm);
