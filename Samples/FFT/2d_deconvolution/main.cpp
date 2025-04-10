#include <fftw3.h>

#include <complex>
#include <iomanip>
#include <iostream>

constexpr std::complex<double> convertToComplex(const fftw_complex &aValue) {
  return std::complex<double>{aValue[0], aValue[1]};
}

constexpr void convertFromComplex(const std::complex<double> &aInput,
                                  fftw_complex &aOutput) {
  aOutput[0] = aInput.real();
  aOutput[1] = aInput.imag();
}

int main(int argc, char *argv[]) {
  constexpr int Nx = 16;
  constexpr int Ny = 16;

  // row-major order
  constexpr double expectedData[Nx * Ny] = {
      5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 0
      5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 1
      5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 2
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 3
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 4
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 5
      0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 6
      0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 7
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 8
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 9
      0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0,  // row 10
      0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0,  // row 11
      0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0,  // row 12
      0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0,  // row 13
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 14
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 15
  };

  // row-major order
  double convolvedData[Nx * Ny] = {
      2.22222,
      3.33333,
      2.22222,
      1.11111,
      -1.11022e-16,
      -2.22045e-16,
      -1.11022e-16,
      2.22045e-16,
      2.22045e-16,
      8.88178e-16,
      2.22045e-16,
      1.11022e-16,
      -1.11022e-16,
      -4.44089e-16,
      -1.11022e-16,
      1.11111,  // row 0
      3.33333,
      5,
      3.33333,
      1.66667,
      -1.11022e-16,
      -4.44089e-16,
      -4.25041e-16,
      2.22045e-16,
      2.22045e-16,
      0,
      2.22045e-16,
      0,
      -1.11022e-16,
      0,
      2.02996e-16,
      1.66667,  // row 1
      2.22222,
      3.33333,
      2.22222,
      1.11111,
      -1.11022e-16,
      -3.33067e-16,
      -1.11022e-16,
      1.11022e-16,
      0,
      4.44089e-16,
      0,
      0,
      -1.11022e-16,
      -3.33067e-16,
      -1.11022e-16,
      1.11111,  // row 2
      1.11111,
      1.66667,
      1.11111,
      0.555556,
      -1.66533e-16,
      -3.33067e-16,
      -1.73268e-16,
      0,
      2.22045e-16,
      2.22045e-16,
      3.33067e-16,
      2.22045e-16,
      1.66533e-16,
      0,
      6.22458e-17,
      0.555556,  // row 3
      1.39645e-16,
      1.92291e-16,
      1.64098e-16,
      1.23588e-16,
      8.50015e-17,
      5.12915e-17,
      1.03368e-16,
      1.18549e-16,
      9.62772e-17,
      7.5001e-17,
      1.48152e-16,
      2.10219e-16,
      1.64799e-16,
      4.9178e-17,
      -1.31623e-17,
      6.80611e-17,  // row 4
      -1.52656e-16,
      -1.94289e-16,
      4.7621e-18,
      4.16334e-17,
      0.444444,
      0.888889,
      0.888889,
      0.444444,
      1.249e-16,
      2.77556e-17,
      1.61771e-16,
      2.08167e-16,
      2.77556e-16,
      3.33067e-16,
      1.11022e-16,
      -2.77556e-17,  // row 5
      -1.66533e-16,
      -1.66533e-16,
      -1.11022e-16,
      -1.11022e-16,
      0.888889,
      1.77778,
      1.77778,
      0.888889,
      5.55112e-17,
      -5.55112e-17,
      -1.11022e-16,
      -5.55112e-17,
      5.55112e-17,
      -1.11022e-16,
      -1.11022e-16,
      -5.55112e-17,  // row 6
      -5.55112e-17,
      1.11022e-16,
      3.6953e-16,
      3.33067e-16,
      0.888889,
      1.77778,
      1.77778,
      0.888889,
      -5.55112e-17,
      -2.22045e-16,
      -2.58507e-16,
      -1.66533e-16,
      0,
      1.11022e-16,
      0,
      0,  // row 7
      1.38778e-16,
      1.66533e-16,
      9.47635e-17,
      -6.93889e-17,
      0.444444,
      0.888889,
      0.888889,
      0.444444,
      -8.32667e-17,
      -1.11022e-16,
      1.62588e-17,
      1.249e-16,
      2.498e-16,
      3.33067e-16,
      2.77556e-16,
      1.38778e-16,  // row 8
      1.38778e-16,
      2.77556e-16,
      2.22045e-16,
      2.22045e-16,
      1.11022e-16,
      -1.11022e-16,
      -1.82792e-16,
      -1.38778e-16,
      0.333333,
      0.666667,
      1,
      1,
      0.666667,
      0.333333,
      -2.61297e-16,
      -8.32667e-17,  // row 9
      -2.77556e-16,
      -1.11022e-16,
      0,
      2.22045e-16,
      0,
      -1.66533e-16,
      -1.20547e-16,
      -1.66533e-16,
      0.666667,
      1.33333,
      2,
      2,
      1.33333,
      0.666667,
      -4.34565e-16,
      -3.88578e-16,  // row 10
      -5.55112e-17,
      0,
      0,
      4.44089e-16,
      1.11022e-16,
      1.66533e-16,
      -6.50354e-17,
      0,
      1,
      2,
      3,
      3,
      2,
      1,
      -3.79054e-16,
      -2.22045e-16,  // row 11
      -5.55112e-17,
      1.11022e-16,
      4.44089e-16,
      4.44089e-16,
      1.11022e-16,
      1.11022e-16,
      -1.76058e-16,
      0,
      1,
      2,
      3,
      3,
      2,
      1,
      -4.90076e-16,
      -4.44089e-16,  // row 12
      -1.11022e-16,
      0,
      0,
      1.11022e-16,
      0,
      -1.66533e-16,
      -9.52421e-18,
      -5.55112e-17,
      0.666667,
      1.33333,
      2,
      2,
      1.33333,
      0.666667,
      -3.23543e-16,
      -2.77556e-16,  // row 13
      0,
      0,
      5.55112e-17,
      0,
      0,
      -5.55112e-17,
      -5.55112e-17,
      -1.66533e-16,
      0.333333,
      0.666667,
      1,
      1,
      0.666667,
      0.333333,
      -5.55112e-17,
      -5.55112e-17,  // row 14
      1.11111,
      1.66667,
      1.11111,
      0.555556,
      -5.55112e-17,
      -1.11022e-16,
      -3.92523e-17,
      1.11022e-16,
      0,
      0,
      0,
      0,
      -5.55112e-17,
      0,
      3.92523e-17,
      0.555556,  // row 15
  };

  constexpr double w = 1.0 / 9.0;
  double kernel[Nx * Ny] = {
      w, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w,  // row 0
      w, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w,  // row 1
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 2
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 3
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 4
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 5
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 6
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 7
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 8
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 9
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 10
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 11
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 12
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 13
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // row 14
      w, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w,  // row 15
  };

  double restoredData[Nx * Ny]{};

  auto *convolvedFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));
  auto *kernelFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));
  auto *deconvolvedFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));

  auto forwardTransform =
      fftw_plan_dft_r2c_2d(Nx, Ny, convolvedData, convolvedFft, FFTW_ESTIMATE);

  fftw_execute(forwardTransform);

  auto kernelTransform =
      fftw_plan_dft_r2c_2d(Nx, Ny, kernel, kernelFft, FFTW_ESTIMATE);

  fftw_execute(kernelTransform);

  std::cout << std::setw(5) << "deconvolvedFft\n";
  for (int i = 0; i < Ny / 2 + 1; i++) {
    for (int j = 0; j < Nx; j++) {
      convertFromComplex(convertToComplex(convolvedFft[i * Nx + j]) /
                             convertToComplex(kernelFft[i * Nx + j]),
                         deconvolvedFft[i * Nx + j]);
      std::cout << convertToComplex(deconvolvedFft[i * Nx + j]) << "\t";
    }
    std::cout << '\n';
  }

  auto backardTransform =
      fftw_plan_dft_c2r_2d(Nx, Ny, deconvolvedFft, restoredData, FFTW_ESTIMATE);
  fftw_execute(backardTransform);

  std::cout.precision(1);

  std::cout << "outputData\n";
  for (int i = 0; i < Ny; i++) {
    for (int j = 0; j < Nx; j++) {
      std::cout << (restoredData[i * Nx + j] / (1.0 * Nx * Ny)) << ",\t";
    }
    std::cout << '\n';
  }

  fftw_destroy_plan(forwardTransform);
  fftw_destroy_plan(kernelTransform);
  fftw_destroy_plan(backardTransform);

  fftw_free(convolvedFft);
  fftw_free(kernelFft);
  fftw_free(deconvolvedFft);

  return EXIT_SUCCESS;
}
