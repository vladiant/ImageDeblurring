/********************* Cubic Spline Interpolation **********************/
/*
 * Code taken from
 * http://ganeshtiwaridotcomdotnp.blogspot.com/2009/12/c-c-code-cubic-spline-interpolation.html
 */

#include <math.h>

#include <iostream>

using namespace std;

int main() {
  char choice = 'y';
  constexpr int N = 14;  // number of function points (limited to 14!)
  int i, j, k;           // indices
  /*
   * x[i] - array of x values initial
   * f[i] - array of y=f(x) values initial
   * F[i] - derivatives
   * h[i] - x[i+1]- x[i]
   * m[i,i] - sparse matrix, to be reduced!!!
   * s[i] - array, used to calculate functional values
   * a - a*x^3
   * b - b*x^2
   * c - c*x^1
   * d - d*x^0
   */
  float h[N], a, b, c, d, sum, s[N] = {0}, x[N], F[N], f[N], p, m[N][N] = {0},
                               temp;
  cout << "No of samples " << N << '\n';

  // Input sine table
  //-----------------------------------------------------------------
  // Sine table values from  Handbook of mathematical functions
  // by M. Abramowitz and I.A. Stegun, NBS, june 1964
  // -----------------------------------------------------------------

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

  for (i = 0; i < N; i++) {
    cout << "Points:  x[" << i << "]= " << x[i] << ", f[" << i << "]= " << f[i]
         << '\n';
  }

  // set the derivatives and differences
  for (i = N - 1; i > 0; i--) {
    F[i] = (f[i] - f[i - 1]) / (x[i] - x[i - 1]);
    h[i - 1] = x[i] - x[i - 1];
  }

  //*********** formation of h, s , f matrix **************//
  for (i = 1; i < N - 1; i++)  // values without the first and the last
  {
    m[i][i] = 2 * (h[i - 1] + h[i]);  // diagonal values
    if (i != 1) {
      m[i][i - 1] = h[i - 1];  // off-diagonal values - consider sparse matrix
      m[i - 1][i] = h[i - 1];
    }
    m[i][N - 1] = 6 * (F[i + 1] - F[i]);  // i=n-1
    cout << "*** i= " << i << '\n';
  }

  //***********  forward elimination **************//

  for (i = 1; i < N - 2; i++) {
    temp = (m[i + 1][i] / m[i][i]);
    for (j = 1; j <= N - 1; j++)
      m[i + 1][j] -= temp * m[i][j];  // upper diagonal eliminated
  }

  //*********** back ward substitution *********//
  for (i = N - 2; i > 0; i--) {
    sum = 0;
    for (j = i; j <= N - 2; j++) sum += m[i][j] * s[j];
    s[i] = (m[i][N - 1] - sum) / m[i][i];
  }

  // m[i,i] not required anymore

  while (choice == 'y') {
    cout << endl << "Enter x  : ";
    cin >> p;
    for (i = 0; i < N - 1; i++)
      if ((x[i] <= p) && (p <= x[i + 1])) {
        a = (s[i + 1] - s[i]) / (6 * h[i]);  // a*x^3
        b = s[i] / 2;                        // b*x^2
        c = (f[i + 1] - f[i]) / h[i] -
            (2 * h[i] * s[i] + s[i + 1] * h[i]) / 6;  // c*x^1
        d = f[i];                                     // d*x^0
        sum = a * (p - x[i]) * (p - x[i]) * (p - x[i]) +
              b * (p - x[i]) * (p - x[i]) + c * (p - x[i]) + d;

        cout << "First derivative= "
             << 3.0 * a * (p - x[i]) * (p - x[i]) + 2.0 * b * (p - x[i]) + c
             << '\n';
        cout << "Second derivative= " << 6.0 * a * (p - x[i]) + 2.0 * b << '\n';
        cout << "Third derivative= " << 6.0 * a << '\n';
      }
    cout << "Coefficients of sub interval : \n";
    cout << "a= " << a << '\n';
    cout << "b= " << b << '\n';
    cout << "c= " << c << '\n';
    cout << "d= " << d << '\n';

    cout << "\nFunctional value: " << sum << '\n';
    cout << "\nContinue (y/n) ? \n";
    cin >> choice;
  }
  return (0);
}
