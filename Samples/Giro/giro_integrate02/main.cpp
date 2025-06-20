/*
 * Simple program to perform
 * integration of gyro data
 * via multistep Adams–Bashforth methods
 */

#include <cmath>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
  const int N = 100;

  long tmstmp;

  // time, angular velocities, angular positions
  long *timestamp = new long[N];  // timestamp [ns]
  float *time = new float[N];     // time [s]
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]

  float *Thetax = new float[N];  // angle x
  float *Thetay = new float[N];  // angle y
  float *Thetaz = new float[N];  // angle z

  float *Theta1x = new float[N];  // angle x
  float *Theta1y = new float[N];  // angle y
  float *Theta1z = new float[N];  // angle z

  float *Theta2x = new float[N];  // angle x
  float *Theta2y = new float[N];  // angle y
  float *Theta2z = new float[N];  // angle z

  float *Theta3x = new float[N];  // angle x
  float *Theta3y = new float[N];  // angle y
  float *Theta3z = new float[N];  // angle z

  // mean g
  float gix = 0, giy = 0, giz = 0;

  for (int j = 0; j < N; j++) {
    // read the values
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];

    // cout << timestamp[j] << "  " << wx[j] << "  " << wy[j] << "  " << wz[j]
    // << '\n';

    if (j > 0) {
      float Rt[9];       // rotation matrix
      float ThetaR[3];   // angle of rotation
      float dTheta[3];   // angle of rotation change
      float dTheta1[3];  // angle of rotation change
      float dTheta2[3];  // angle of rotation change
      float dTheta3[3];  // angle of rotation change

      // calculate time interval
      // time[j]=(timestamp[j]-timestamp[0])*1e-9;

      // calculate time interval - force to 10ms
      time[j] = int((timestamp[j] - timestamp[0]) * 1e-9 * 100 + 0.5) / 100.0;

      // calculate unit vector of rotation
      float urx = Thetax[j - 1];
      float ury = Thetay[j - 1];
      float urz = Thetaz[j - 1];
      float sur = sqrt(urx * urx + ury * ury + urz * urz);

      if (sur != 0) {
        urx /= sur;
        ury /= sur;
        urz /= sur;
        float c = cos(sur), s = sin(sur), cc = 1 - c;
        // inverse of the rotation matrix
        Rt[0] = urx * urx * cc + c;
        Rt[1] = urx * ury * cc + urz * s;
        Rt[2] = urx * urz * cc - ury * s;
        Rt[3] = ury * urx * cc - urz * s;
        Rt[4] = ury * ury * cc + c;
        Rt[5] = ury * urz * cc + urx * s;
        Rt[6] = urz * urx * cc + ury * s;
        Rt[7] = urz * ury * cc - urx * s;
        Rt[8] = urz * urz * cc + c;
      } else {
        Rt[0] = 1.0;
        Rt[1] = 0.0;
        Rt[2] = 0.0;
        Rt[3] = 0.0;
        Rt[4] = 1.0;
        Rt[5] = 0.0;
        Rt[6] = 0.0;
        Rt[7] = 0.0;
        Rt[8] = 1.0;
      }

      // for (int i=0; i<9; i++) cout << Rt[i] << "  ";
      // cout << '\n';

      dTheta[0] = Rt[0] * wx[j - 1] + Rt[1] * wy[j - 1] + Rt[2] * wz[j - 1];
      dTheta[1] = Rt[3] * wx[j - 1] + Rt[4] * wy[j - 1] + Rt[5] * wz[j - 1];
      dTheta[2] = Rt[6] * wx[j - 1] + Rt[7] * wy[j - 1] + Rt[8] * wz[j - 1];

      // cout << urx << "  " << ury << "  " << urz << '\n';

      // new value of rotation
      Thetax[j] = Thetax[j - 1] + dTheta[0] * (time[j] - time[j - 1]);
      Thetay[j] = Thetay[j - 1] + dTheta[1] * (time[j] - time[j - 1]);
      Thetaz[j] = Thetaz[j - 1] + dTheta[2] * (time[j] - time[j - 1]);

      // Adams–Bashforth 2 method
      if (j > 1) {
        urx = Theta1x[j - 2];
        float ury = Theta1y[j - 2];
        float urz = Theta1z[j - 2];
        if (sur != 0) {
          urx /= sur;
          ury /= sur;
          urz /= sur;
          float c = cos(sur), s = sin(sur), cc = 1 - c;
          // inverse of the rotation matrix
          Rt[0] = urx * urx * cc + c;
          Rt[1] = urx * ury * cc + urz * s;
          Rt[2] = urx * urz * cc - ury * s;
          Rt[3] = ury * urx * cc - urz * s;
          Rt[4] = ury * ury * cc + c;
          Rt[5] = ury * urz * cc + urx * s;
          Rt[6] = urz * urx * cc + ury * s;
          Rt[7] = urz * ury * cc - urx * s;
          Rt[8] = urz * urz * cc + c;
        } else {
          Rt[0] = 1.0;
          Rt[1] = 0.0;
          Rt[2] = 0.0;
          Rt[3] = 0.0;
          Rt[4] = 1.0;
          Rt[5] = 0.0;
          Rt[6] = 0.0;
          Rt[7] = 0.0;
          Rt[8] = 1.0;
        }

        dTheta1[0] = Rt[0] * wx[j - 2] + Rt[1] * wy[j - 2] + Rt[2] * wz[j - 2];
        dTheta1[1] = Rt[3] * wx[j - 2] + Rt[4] * wy[j - 2] + Rt[5] * wz[j - 2];
        dTheta1[2] = Rt[6] * wx[j - 2] + Rt[7] * wy[j - 2] + Rt[8] * wz[j - 2];

        // new value of rotation
        Theta1x[j] = Theta1x[j - 1] + 0.5 * (3 * dTheta[0] - dTheta1[0]) *
                                          (time[j] - time[j - 1]);
        Theta1y[j] = Theta1y[j - 1] + 0.5 * (3 * dTheta[1] - dTheta1[1]) *
                                          (time[j] - time[j - 1]);
        Theta1z[j] = Theta1z[j - 1] + 0.5 * (3 * dTheta[2] - dTheta1[2]) *
                                          (time[j] - time[j - 1]);
      } else {
        Theta1x[1] = Thetax[1];
        Theta1z[1] = Thetaz[1];
        Theta1z[1] = Thetaz[1];
      }

      // Adams–Bashforth 3 method
      if (j > 2) {
        urx = Theta2x[j - 3];
        float ury = Theta2y[j - 3];
        float urz = Theta2z[j - 3];
        if (sur != 0) {
          urx /= sur;
          ury /= sur;
          urz /= sur;
          float c = cos(sur), s = sin(sur), cc = 1 - c;
          // inverse of the rotation matrix
          Rt[0] = urx * urx * cc + c;
          Rt[1] = urx * ury * cc + urz * s;
          Rt[2] = urx * urz * cc - ury * s;
          Rt[3] = ury * urx * cc - urz * s;
          Rt[4] = ury * ury * cc + c;
          Rt[5] = ury * urz * cc + urx * s;
          Rt[6] = urz * urx * cc + ury * s;
          Rt[7] = urz * ury * cc - urx * s;
          Rt[8] = urz * urz * cc + c;
        } else {
          Rt[0] = 1.0;
          Rt[1] = 0.0;
          Rt[2] = 0.0;
          Rt[3] = 0.0;
          Rt[4] = 1.0;
          Rt[5] = 0.0;
          Rt[6] = 0.0;
          Rt[7] = 0.0;
          Rt[8] = 1.0;
        }

        dTheta2[0] = Rt[0] * wx[j - 3] + Rt[1] * wy[j - 3] + Rt[2] * wz[j - 3];
        dTheta2[1] = Rt[3] * wx[j - 3] + Rt[4] * wy[j - 3] + Rt[5] * wz[j - 3];
        dTheta2[2] = Rt[6] * wx[j - 3] + Rt[7] * wy[j - 3] + Rt[8] * wz[j - 3];

        // new value of rotation
        Theta2x[j] = Theta2x[j - 1] +
                     (23 * dTheta[0] - 16 * dTheta1[0] + 5 * dTheta2[0]) *
                         (time[j] - time[j - 1]) / 12.0;
        Theta2y[j] = Theta2y[j - 1] +
                     (23 * dTheta[1] - 16 * dTheta1[1] + 5 * dTheta2[1]) *
                         (time[j] - time[j - 1]) / 12.0;
        Theta2z[j] = Theta2z[j - 1] +
                     (23 * dTheta[2] - 16 * dTheta1[2] + 5 * dTheta2[2]) *
                         (time[j] - time[j - 1]) / 12.0;
      } else {
        Theta2x[j] = Theta1x[j];
        Theta2z[j] = Theta1z[j];
        Theta2z[j] = Theta1z[j];
      }

      // Adams–Bashforth 4 method
      if (j > 3) {
        urx = Theta2x[j - 4];
        float ury = Theta2y[j - 4];
        float urz = Theta2z[j - 4];
        if (sur != 0) {
          urx /= sur;
          ury /= sur;
          urz /= sur;
          float c = cos(sur), s = sin(sur), cc = 1 - c;
          // inverse of the rotation matrix
          Rt[0] = urx * urx * cc + c;
          Rt[1] = urx * ury * cc + urz * s;
          Rt[2] = urx * urz * cc - ury * s;
          Rt[3] = ury * urx * cc - urz * s;
          Rt[4] = ury * ury * cc + c;
          Rt[5] = ury * urz * cc + urx * s;
          Rt[6] = urz * urx * cc + ury * s;
          Rt[7] = urz * ury * cc - urx * s;
          Rt[8] = urz * urz * cc + c;
        } else {
          Rt[0] = 1.0;
          Rt[1] = 0.0;
          Rt[2] = 0.0;
          Rt[3] = 0.0;
          Rt[4] = 1.0;
          Rt[5] = 0.0;
          Rt[6] = 0.0;
          Rt[7] = 0.0;
          Rt[8] = 1.0;
        }

        dTheta3[0] = Rt[0] * wx[j - 4] + Rt[1] * wy[j - 4] + Rt[2] * wz[j - 4];
        dTheta3[1] = Rt[3] * wx[j - 4] + Rt[4] * wy[j - 4] + Rt[5] * wz[j - 4];
        dTheta3[2] = Rt[6] * wx[j - 4] + Rt[7] * wy[j - 4] + Rt[8] * wz[j - 4];

        // new value of rotation
        Theta3x[j] = Theta3x[j - 1] + (55 * dTheta[0] - 59 * dTheta1[0] +
                                       37 * dTheta2[0] - 9 * dTheta3[0]) *
                                          (time[j] - time[j - 1]) / 24.0;
        Theta3y[j] = Theta3y[j - 1] + (55 * dTheta[1] - 59 * dTheta1[1] +
                                       37 * dTheta2[1] - 9 * dTheta3[1]) *
                                          (time[j] - time[j - 1]) / 24.0;
        Theta3z[j] = Theta3z[j - 1] + (55 * dTheta[2] - 59 * dTheta1[2] +
                                       37 * dTheta2[2] - 9 * dTheta3[2]) *
                                          (time[j] - time[j - 1]) / 24.0;
      } else {
        Theta3x[j] = Theta2x[j];
        Theta3z[j] = Theta2z[j];
        Theta3z[j] = Theta2z[j];
      }

    } else {
      Thetax[0] = 0.0;
      Thetay[0] = 0.0;
      Thetaz[0] = 0.0;
      Theta1x[0] = 0.0;
      Theta1y[0] = 0.0;
      Theta1z[0] = 0.0;
      Theta2x[0] = 0.0;
      Theta2y[0] = 0.0;
      Theta2z[0] = 0.0;
      Theta3x[0] = 0.0;
      Theta3y[0] = 0.0;
      Theta3z[0] = 0.0;

      time[0] = 0.0;
    }
    /*
    cout << time[j] << "  " << Thetax[j] << "  " << Thetay[j] << "  " <<
    Thetaz[j] << "  " << Theta1x[j] << "  " << Theta1y[j] << "  " << Theta1z[j]
    << "  " << Theta2x[j] << "  " << Theta2y[j] << "  " << Theta2z[j] << "  " <<
    Theta3x[j] << "  " << Theta3y[j] << "  " << Theta3z[j] << '\n';
     */
    cout << time[j] * 1e9 + timestamp[0] << "  " << Thetax[j] << "  "
         << Thetay[j] << "  " << Thetaz[j] << endl;
  }

  delete[] timestamp;
  delete[] time;
  delete[] wx;
  delete[] wy;
  delete[] wz;
  delete[] Thetax;
  delete[] Thetay;
  delete[] Thetaz;
  delete[] Theta1x;
  delete[] Theta1y;
  delete[] Theta1z;
  delete[] Theta2x;
  delete[] Theta2y;
  delete[] Theta2z;
  delete[] Theta3x;
  delete[] Theta3y;
  delete[] Theta3z;

  return (0);
}
