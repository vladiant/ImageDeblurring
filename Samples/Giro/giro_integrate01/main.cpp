/*
 * Simple program to perform
 * integration of gyro data
 * and calculate the movement
 * of a single pixel
 */

#include <cmath>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
  const int N = 100;

  // vector from accelerometer to the center of rotation [m]
  const float rpqx = 0.02;
  const float rpqy = 0.05;
  const float rpqz = 0.00;
  // its length
  float drpq = sqrt(rpqx * rpqx + rpqy * rpqy + rpqz * rpqz);

  // earth acceleration
  const float gx = 0.0, gy = -9.8, gz = 0;

  long tmstmp;

  // time, angular velocities, angular positions
  long *timestamp = new long[N];  // timestamp [ns]
  float *time = new float[N];     // time [s]
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  float *Thetax = new float[N];   // angle x
  float *Thetay = new float[N];   // angle y
  float *Thetaz = new float[N];   // angle z
  float *wix = new float[N];      // absolute angular velocity x [rad/s]
  float *wiy = new float[N];      // absolute angular velocity y [rad/s]
  float *wiz = new float[N];      // absolute angular velocity z [rad/s]

  // this should be measured - here it will be synthetized, assuming g is
  // opposite to y
  float *apx = new float[N];  // acceleration x [rad/s]
  float *apy = new float[N];  // acceleration y [rad/s]
  float *apz = new float[N];  // acceleration z [rad/s]

  float *aipx = new float[N];  // acceleration in initial frame x [rad/s]
  float *aipy = new float[N];  // acceleration in initial frame y [rad/s]
  float *aipz = new float[N];  // acceleration in initial frame z [rad/s]

  // angular accelerations - synthetized here, part of accelerometer data
  float *alphax = new float[N];  // acceleration in initial frame x [rad/s^2]
  float *alphay = new float[N];  // acceleration in initial frame y [rad/s^2]
  float *alphaz = new float[N];  // acceleration in initial frame z [rad/s^2]

  // velocities
  float *vpx = new float[N];  // velocity x [m/s]
  float *vpy = new float[N];  // velocity y [m/s]
  float *vpz = new float[N];  // velocity z [m/s]

  // accelerometer position (center of the coordinate system)
  float *xpx = new float[N];  // position x [m]
  float *xpy = new float[N];  // position y [m]
  float *xpz = new float[N];  // position z [m]

  // pixel position (defined by rpq distance)
  float *xix = new float[N];  // position x [m]
  float *xiy = new float[N];  // position y [m]
  float *xiz = new float[N];  // position z [m]

  // mean g
  float gix = 0, giy = 0, giz = 0;

  for (int j = 0; j < N; j++) {
    // read the values
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];

    // cout << timestamp[j] << "  " << wx[j] << "  " << wy[j] << "  " << wz[j]
    // << '\n';

    if (j > 0) {
      float *Rt = new float[9];      // rotation matrix
      float *ThetaR = new float[3];  // angle of rotation
      float *dTheta = new float[3];  // angle of rotation change

      // calculate time interval
      time[j] = (timestamp[j] - timestamp[0]) * 1e-9;

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

      // absolute angular speed
      wix[j] = (Thetax[j] - Thetax[j - 1]) / (time[j] - time[j - 1]);
      wiy[j] = (Thetay[j] - Thetay[j - 1]) / (time[j] - time[j - 1]);
      wiz[j] = (Thetaz[j] - Thetaz[j - 1]) / (time[j] - time[j - 1]);

      // synthetic angular acceleration
      alphax[j] = dTheta[0];
      alphay[j] = dTheta[1];
      alphaz[j] = dTheta[2];

      // measured accelerations - synthetized
      float *Rti = new float[9];  // inverse of Rt
      float dtRt = Rt[0] * (Rt[4] * Rt[8] - Rt[5] * Rt[7]) -
                   Rt[1] * (Rt[3] * Rt[8] - Rt[5] * Rt[6]) +
                   Rt[2] * (Rt[3] * Rt[7] - Rt[4] * Rt[6]);

      // cout << "dtRt " << dtRt << '\n';

      if (dtRt != 0) {
        Rti[0] = (Rt[4] * Rt[8] - Rt[5] * Rt[7]) / dtRt;
        Rti[1] = (Rt[2] * Rt[7] - Rt[1] * Rt[8]) / dtRt;
        Rti[2] = (Rt[2] * Rt[6] - Rt[2] * Rt[4]) / dtRt;
        Rti[3] = (Rt[5] * Rt[6] - Rt[3] * Rt[8]) / dtRt;
        Rti[4] = (Rt[0] * Rt[8] - Rt[2] * Rt[6]) / dtRt;
        Rti[5] = (Rt[2] * Rt[3] - Rt[0] * Rt[5]) / dtRt;
        Rti[6] = (Rt[3] * Rt[7] - Rt[4] * Rt[6]) / dtRt;
        Rti[7] = (Rt[1] * Rt[6] - Rt[0] * Rt[7]) / dtRt;
        Rti[8] = (Rt[0] * Rt[4] - Rt[1] * Rt[3]) / dtRt;
      } else {
        Rti[0] = 0;
        Rti[1] = 0;
        Rti[2] = 0;
        Rti[3] = 0;
        Rti[4] = 0;
        Rti[5] = 0;
        Rti[6] = 0;
        Rti[7] = 0;
        Rti[8] = 0;
      }

      // for (int i=0; i<9; i++) cout << Rt[i] << "  ";
      // cout << '\n';

      apx[j] = Rti[0] * (wx[j] * drpq - rpqx * sur * sur + alphay[j] * rpqz -
                         alphaz[j] * rpqy) +
               Rti[1] * (wy[j] * drpq - rpqy * sur * sur + alphaz[j] * rpqx -
                         alphax[j] * rpqz) +
               Rti[2] * (wz[j] * drpq - rpqz * sur * sur + alphax[j] * rpqy -
                         alphay[j] * rpqx) +
               gx;

      apy[j] = Rti[3] * (wx[j] * drpq - rpqx * sur * sur + alphay[j] * rpqz -
                         alphaz[j] * rpqy) +
               Rti[4] * (wy[j] * drpq - rpqy * sur * sur + alphaz[j] * rpqx -
                         alphax[j] * rpqz) +
               Rti[5] * (wz[j] * drpq - rpqz * sur * sur + alphax[j] * rpqy -
                         alphay[j] * rpqx) +
               gy;

      apz[j] = Rti[6] * (wx[j] * drpq - rpqx * sur * sur + alphay[j] * rpqz -
                         alphaz[j] * rpqy) +
               Rti[7] * (wy[j] * drpq - rpqy * sur * sur + alphaz[j] * rpqx -
                         alphax[j] * rpqz) +
               Rti[8] * (wz[j] * drpq - rpqz * sur * sur + alphax[j] * rpqy -
                         alphay[j] * rpqx) +
               gz;

      // test the mean g
      gix += apx[j];
      giy += apy[j];
      giz += apz[j];

      // accelerations
      aipx[j] = Rt[0] * apx[j] + Rt[1] * apy[j] + Rt[2] * apz[j];
      aipy[j] = Rt[3] * apx[j] + Rt[4] * apy[j] + Rt[5] * apz[j];
      aipz[j] = Rt[6] * apx[j] + Rt[7] * apy[j] + Rt[8] * apz[j];

      // velocities
      vpx[j] = (aipx[j - 1] - gx) * (time[j] - time[j - 1]) + vpx[j - 1];
      vpy[j] = (aipy[j - 1] - gy) * (time[j] - time[j - 1]) + vpy[j - 1];
      vpz[j] = (aipz[j - 1] - gz) * (time[j] - time[j - 1]) + vpz[j - 1];

      // position of accelerometer
      xpx[j] = 0.5 * (aipx[j - 1] - gx) * (time[j] - time[j - 1]) *
                   (time[j] - time[j - 1]) +
               vpx[j - 1] * (time[j] - time[j - 1]) + xpx[j - 1];
      xpy[j] = 0.5 * (aipy[j - 1] - gy) * (time[j] - time[j - 1]) *
                   (time[j] - time[j - 1]) +
               vpy[j - 1] * (time[j] - time[j - 1]) + xpy[j - 1];
      xpz[j] = 0.5 * (aipz[j - 1] - gz) * (time[j] - time[j - 1]) *
                   (time[j] - time[j - 1]) +
               vpz[j - 1] * (time[j] - time[j - 1]) + xpz[j - 1];

      // position of pixel
      xix[j] = Rt[0] * rpqx + Rt[1] * rpqy + Rt[2] * rpqz - xpx[j];
      xiy[j] = Rt[3] * rpqx + Rt[4] * rpqy + Rt[5] * rpqz - xpy[j];
      xiz[j] = Rt[6] * rpqx + Rt[7] * rpqy + Rt[8] * rpqz - xpz[j];

      delete[] Rti;
      delete[] Rt;
      delete[] ThetaR;
      delete[] dTheta;

    } else {
      Thetax[0] = 0.0;
      Thetay[0] = 0.0;
      Thetaz[0] = 0.0;

      time[0] = 0.0;

      wix[0] = wx[0];
      wiy[0] = wy[0];
      wiz[0] = wz[0];

      float sur = sqrt(wx[0] * wx[0] + wy[0] * wy[0] + wz[0] * wz[0]);
      // measured accelerations - synthetized
      apx[0] =
          wx[0] * drpq - rpqx * sur + alphay[0] * rpqz - alphaz[0] * rpqy + gx;
      apy[0] =
          wy[0] * drpq - rpqy * sur + alphaz[0] * rpqx - alphax[0] * rpqz + gy;
      apz[0] =
          wz[0] * drpq - rpqz * sur + alphax[0] * rpqy - alphay[0] * rpqx + gz;
      // test the mean g
      gix += apx[0];
      giy += apy[0];
      giz += apz[0];

      // synthetic angular acceleration
      alphax[0] = 0;
      alphay[0] = 0;
      alphaz[0] = 0;

      // accelerations
      aipx[0] = apx[0];
      aipy[0] = apy[0];
      aipz[0] = apz[0];

      // velocities
      vpx[0] = 0;
      vpy[0] = 0;
      vpz[0] = 0;

      // position of accelerometer
      xpx[0] = 0;
      xpy[0] = 0;
      xpz[0] = 0;

      // position of pixel
      xix[0] = rpqx;
      xiy[0] = rpqy;
      xiz[0] = rpqz;
    }

    // cout << time[j] << "  " << Thetax[j] << "  " << Thetay[j] << "  " <<
    // Thetaz[j] << '\n'; cout << time[j] << "  " << wix[j] << "  " << wiy[j] <<
    // "  " << wiz[j] << '\n'; cout << time[j] << "  " << apx[j] << "  " <<
    // apy[j] << "  " << apz[j] << '\n'; cout << time[j] << "  " << aipx[j] << "
    // " << aipy[j] << "  " << aipz[j] << '\n'; cout << time[j] << "  " <<
    // alphax[j] << "  " << alphay[j] << "  " << alphaz[j] << '\n'; cout <<
    // time[j] << "  " << vpx[j] << "  " << vpy[j] << "  " << vpz[j] << '\n';
    // cout << time[j] << "  " << xpx[j] << "  " << xpy[j] << "  " << xpz[j] <<
    // '\n'; cout << time[j] << "  " << xix[j] << "  " << xiy[j] << "  " <<
    // xiz[j] << '\n';
  }

  cout << "\nMean g: (" << gix / N << ", " << giy / N << ", " << giz / N
       << ") = " << sqrt(gix * gix + giy * giy + giz * giz) / N << '\n';

  delete[] timestamp;
  delete[] time;
  delete[] wx;
  delete[] wy;
  delete[] wz;
  delete[] Thetax;
  delete[] Thetay;
  delete[] Thetaz;
  delete[] wix;
  delete[] wiy;
  delete[] wiz;
  delete[] apx;
  delete[] apy;
  delete[] apz;
  delete[] aipx;
  delete[] aipy;
  delete[] aipz;
  delete[] alphax;
  delete[] alphay;
  delete[] alphaz;
  delete[] vpx;
  delete[] vpy;
  delete[] vpz;
  delete[] xpx;
  delete[] xpy;
  delete[] xpz;
  delete[] xix;
  delete[] xiy;
  delete[] xiz;

  return (0);
}
