/*
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#pragma once

int fft(float *x_in, float *x_out, int n, int shift);

int fftInverse(float *x_in, float *x_out, int n, int shift);

int fft2d(float *x_in, float *x_out, int numRows, int numColls);

int fftInverse2d(float *x_in, float *x_out, int numRows, int numColls);
