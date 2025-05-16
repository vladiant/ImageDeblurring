#include <iostream>

#include "MonotCubicInterpolator.hpp"

int main() {
  // Input sine table
  //-----------------------------------------------------------------
  // Sine table values from  Handbook of mathematical functions
  // by M. Abramowitz and I.A. Stegun, NBS, june 1964
  // -----------------------------------------------------------------

  constexpr int N = 14;  // number of function points
  float x[N], f[N];

  x[0] = 0.000;
  f[0] = 0.00000000;
  x[1] = 0.125;
  f[1] = 0.12467473;
  x[2] = 0.217;
  f[2] = 0.21530095;
  x[3] = 0.299;
  f[3] = 0.29456472;
  x[4] = 0.376;
  f[4] = 0.36720285;
  x[5] = 0.450;
  f[5] = 0.43496553;
  x[6] = 0.520;
  f[6] = 0.49688014;
  x[7] = 0.589;
  f[7] = 0.55552980;
  x[8] = 0.656;
  f[8] = 0.60995199;
  x[9] = 0.721;
  f[9] = 0.66013615;
  x[10] = 0.7853981634;
  f[10] = 0.7071067812;
  x[11] = 0.849;
  f[11] = 0.75062005;
  x[12] = 0.911;
  f[12] = 0.79011709;
  x[13] = 0.972;
  f[13] = 0.82601466;

  MonotCubicInterpolator interpolator;

  for (int i = 0; i < N; i++) {
    interpolator.addPair(x[i], f[i]);
  }

  char choice = 'y';
  while (choice == 'y') {
    std::cout << std::endl << "Enter x  : ";

    float p;
    std::cin >> p;

    std::cout << "\nFunctional value: " << interpolator(p) << '\n';
    std::cout << "\nContinue (y/n) ? \n";
    std::cin >> choice;
  }

  return 0;
}