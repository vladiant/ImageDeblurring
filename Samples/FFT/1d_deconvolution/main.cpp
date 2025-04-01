#include <fftw3.h>

#include <complex>
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
  constexpr int N = 16;

  constexpr double expectedData[N] = {5, 5, 5, 0, 4, 4, 0, 0,
                                      3, 3, 3, 3, 0, 0, 0, 0};

  double convolvedData[N] = {3.33333, 5, 3.33333, 3,      2.66667, 2.66667,
                             1.33333, 1, 2,       3,      3,       2,
                             1,       0, 0,       1.66667};
  double kernel[N] = {1.0 / 3.0, 1.0 / 3.0, 0, 0, 0, 0, 0, 0,
                      0,         0,         0, 0, 0, 0, 0, 1.0 / 3.0};

  double restoredData[N]{};

  auto *convolvedFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  auto *kernelFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  auto *restoredFft =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

  auto forwardTransform =
      fftw_plan_dft_r2c_1d(N, convolvedData, convolvedFft, FFTW_ESTIMATE);

  fftw_execute(forwardTransform);

  auto kernelTransform =
      fftw_plan_dft_r2c_1d(N, kernel, kernelFft, FFTW_ESTIMATE);

  fftw_execute(kernelTransform);

  for (int i = 0; i < N / 2 + 1; i++) {
    convertFromComplex(
        convertToComplex(convolvedFft[i]) / convertToComplex(kernelFft[i]),
        restoredFft[i]);
    std::cout << "restoredFft[" << i
              << "] = " << convertToComplex(restoredFft[i]) << "\n";
  }

  auto backardTransform =
      fftw_plan_dft_c2r_1d(N, restoredFft, restoredData, FFTW_ESTIMATE);
  fftw_execute(backardTransform);

  for (int i = 0; i < N; i++) {
    std::cout << "restoredData[" << i << "] = {" << restoredData[i] / N << "} "
              << "expectedData[" << i << "] = {" << expectedData[i] << "}\n";
  }

  fftw_destroy_plan(forwardTransform);
  fftw_destroy_plan(kernelTransform);
  fftw_destroy_plan(backardTransform);

  fftw_free(convolvedFft);
  fftw_free(kernelFft);
  fftw_free(restoredFft);

  return EXIT_SUCCESS;
}
