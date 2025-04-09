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
  double inputData[Nx * Ny] = {
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

  double outputData[Nx * Ny]{};

  auto *inputFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));
  auto *kernelFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));
  auto *convolvedFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * (Ny / 2 + 1));

  auto forwardTransform =
      fftw_plan_dft_r2c_2d(Nx, Ny, inputData, inputFft, FFTW_ESTIMATE);

  fftw_execute(forwardTransform);

  auto kernelTransform =
      fftw_plan_dft_r2c_2d(Nx, Ny, kernel, kernelFft, FFTW_ESTIMATE);

  fftw_execute(kernelTransform);

  std::cout << std::setw(5) << "convolvedFft\n";
  for (int i = 0; i < Ny / 2 + 1; i++) {
    for (int j = 0; j < Nx; j++) {
      convertFromComplex(convertToComplex(inputFft[i * Nx + j]) *
                             convertToComplex(kernelFft[i * Nx + j]),
                         convolvedFft[i * Nx + j]);
      std::cout << convertToComplex(convolvedFft[i * Nx + j]) << "\t";
    }
    std::cout << '\n';
  }

  auto backardTransform =
      fftw_plan_dft_c2r_2d(Nx, Ny, convolvedFft, outputData, FFTW_ESTIMATE);
  fftw_execute(backardTransform);

  std::cout.precision(1);

  std::cout << "outputData\n";
  for (int i = 0; i < Ny; i++) {
    for (int j = 0; j < Nx; j++) {
      std::cout << (outputData[i * Nx + j] / (1.0 * Nx * Ny)) << "\t";
    }
    std::cout << '\n';
  }

  fftw_destroy_plan(forwardTransform);
  fftw_destroy_plan(kernelTransform);
  fftw_destroy_plan(backardTransform);

  fftw_free(inputFft);
  fftw_free(kernelFft);
  fftw_free(convolvedFft);

  return EXIT_SUCCESS;
}
