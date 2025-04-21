/*
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#pragma once

bool FFT(int dir, int m, float *x, float *y, int step);
bool FFT2D_(float *c, int nx, int ny, int dir);
bool Powerof2_(int n, int *m, int *twopm);
