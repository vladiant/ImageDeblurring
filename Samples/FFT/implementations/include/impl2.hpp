/*
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#pragma once

#include <complex>
#include <valarray>

typedef std::complex<float> Complex;
typedef std::valarray<Complex> CArray;

void fft(CArray& x);

void ifft(CArray& x);

void fft2d(float* x, int width, int height);

void ifft2d(float* x, int width, int height);
