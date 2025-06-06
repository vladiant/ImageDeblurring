//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Test for the integration of video stabilization
//             : in deblurring. Demonstration program.
//============================================================================

#include <iostream>

#include "DiscreteAngularVelocitiesIntegrator.h"
#include "GyroDataCorrection.h"
#include "Versor.h"

using namespace std;

using Test::Math::Vector;
using Test::Math::Versor;
using Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator;
using Test::OpticalFlow::GyroDataCorrection;
using Test::OpticalFlow::TimeOrientation;

// container for the gyro data - for 200 samples
DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
    angularVelocityIntegrator(200);

// this procedure corrects the entered gyro data
void CalcAngularVelocity(GyroDataCorrection<float> &gyroCorrector,
                         GyroDataCorrection<float>::GyroSample_t gyroSample) {
  if (gyroCorrector.startup(gyroSample)) {
    // GyroDataCorrection<float>::GyroSample_t gyroSampleSaved = gyroSample;
    while (gyroCorrector.sweepForInsertOrCorrect(gyroSample)) {
      GyroDataCorrection<float>::GyroSample_t gyroSample2 = gyroCorrector.get();
      // recursive call of this function
      CalcAngularVelocity(gyroCorrector, gyroSample2);
    }
    // this prints the corrected gyro data
    // cout << gyroSample.first << " " << gyroSample.second.x() <<	" " <<
    // gyroSample.second.y() << " " << gyroSample.second.z() << endl;

    // corrected angular velocity added to container
    angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }
}

int main(int, char *[]) {
  // set number of samples for further treatment
  int N;

  long capture_timestamp;  // capture end timestamp

  // read the values without parser
  cin >> N;
  cin >> capture_timestamp;

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  long *timestamp = new long[N];  // timestamp [ns]

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  GyroDataCorrection<float> gyroCorrector(1000000, 20);

  // read the values
  for (int j = 0; j < N; j++) {
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];
  }

  // read the values
  for (int j = 0; j < N * 1.01; j++) {
    GyroDataCorrection<float>::GyroSample_t gyroSample(
        timestamp[j], Vector<float, 3>(wx[j], wy[j], wz[j]));

    // correct the gyro data
    CalcAngularVelocity(gyroCorrector, gyroSample);

    // uncomment this to add the samples without correction and comment the
    // upper line angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }

  // first timestamp
  TimeOrientation<Versor<float>> TEST1 =
      angularVelocityIntegrator.orientations().orientations()[0];
  long int time_zero = TEST1.time;

  // received from calibration data
  Versor<float> gyroSpacetoCameraQuaternion =
      Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);

  for (int j = 0; j < N - 20 - 1; j++) {
    TimeOrientation<Versor<float>> TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    // cout << j << "  ";

    /*
    cout << TEST.time << endl;    // timestamp
    cout << TEST.orientation << endl;   // rotation as versor

    //versor components
    cout << TEST.orientation.x() << endl;
    cout << TEST.orientation.y() << endl;
    cout << TEST.orientation.z() << endl;
    cout << TEST.orientation.w() << endl;

    //total rotation angle
    cout << TEST.orientation.rotationAngle() << endl;
    */

    // cout << ((TEST.time-time_zero)*1e-9)+0.21 << "  ";
    // cout << TEST.time << " ";
    cout << ((TEST.time - time_zero) * 1e-9) << "  ";

    // multiplication of versors
    // cout << gyroSpacetoCameraQuaternion*TEST.orientation << "  " <<
    // TEST.orientation << "  ";
    TEST.orientation = gyroSpacetoCameraQuaternion * TEST.orientation;

    // custom created rotation angle calculation routine
    float w, x, y, z, rotAxis[3];
    float halfedAngle, angle, sinHalfedAngle, mult;
    // versor components
    x = TEST.orientation.x();
    y = TEST.orientation.y();
    z = TEST.orientation.z();
    w = TEST.orientation.w();
    if (w > 0.9) {
      halfedAngle = asin(sqrt(x * x + y * y + z * z));
      angle = halfedAngle * 2;
      sinHalfedAngle = sin(halfedAngle);
      mult = (sinHalfedAngle == 0 ? 1 : angle / sinHalfedAngle);
      rotAxis[0] = mult * x;
      rotAxis[1] = mult * y;
      rotAxis[2] = mult * z;
    } else {
      halfedAngle = acos(w);
      angle = halfedAngle * 2;
      sinHalfedAngle = sin(halfedAngle);
      mult = angle / sinHalfedAngle;
      rotAxis[0] = mult * x;
      rotAxis[1] = mult * y;
      rotAxis[2] = mult * z;
    }
    cout << rotAxis[0] << "  " << rotAxis[1] << "  " << rotAxis[2] << endl;

    /*
    // x, y, z - rotation angles
    Vector<float,3> AxRot;
    TEST.orientation.toRoatationAxis(AxRot);
    //cout << AxRot << endl;
    cout << AxRot.x() << "  " << AxRot.y() << "  " << AxRot.z() << endl;
    */

    /*
    //division of versors
    cout << TEST.orientation << "  " << TEST1.second << endl;
    //invert the versor
    cout << TEST1.second.invert() << endl;
    //multiplies the versors - can not be divided directly
    cout << TEST.orientation*TEST1.second.invert()  << endl;
    cout << TEST.orientation << "  " << TEST1.second << endl;
    */

    /*
    //rotation matrix
    Matrix<float,3,3> MatRot;
    TEST.orientation.toRotationMatrix(MatRot);
    cout << MatRot[0][0] << "  " << MatRot[0][1] << "  " << MatRot[0][2] <<
    "  " << MatRot[1][0] << "  " << MatRot[1][1] << "  " << MatRot[1][2] <<
    "  " << MatRot[2][0] << "  " << MatRot[2][1] << "  " << MatRot[2][2] <<
    endl;
    */
  }

  // release the memory
  delete[] timestamp;
  delete[] wx;
  delete[] wy;
  delete[] wz;

  return 0;
}
