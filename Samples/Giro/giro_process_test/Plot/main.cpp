//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Gnuplot drawing utility for gyro data extracted
//             : by parsegyro program
// Created on  : Mar 27, 2012
//============================================================================

// video stabilization libraries
#include <stdint.h>
#include <string.h>

#include <fstream>
#include <iostream>

#include "DiscreteAngularVelocitiesIntegrator.h"
#include "GyroDataCorrection.h"
#include "Versor.h"

using namespace std;

using Test::Time;
using Test::timeFromSeconds;
using Test::timeToSeconds;
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

    // corrected angular velocity added to container
    angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }
}

int main(int argc, char *argv[]) {
  // string constants to be searched in data file
  const char CAPTURE_MARK[] = "capture timestamp";
  const char GYRO_SAMPLES_MARK[] = "total gyro samples";

  // samples to be used for averaging
  const int SAMPL_AVER = 20;

  // set number of samples for further treatment
  int N{};

  // number of rows in data file
  int dfile_rows;

  Time capture_timestamp{};                       // capture end timestamp
  Time time_delay = timeFromSeconds<float>(0.0);  // delay time
  Time time_exposure = timeFromSeconds<float>(0.05999);  // exposure time

  // time for readout in seconds
  const Time time_readout = timeFromSeconds<float>(0.03283);

  string line;

  if (argc < 2) {
    cout << "\nProgram to extract gyro data from parsegyro data file." << endl;
    cout << "Usage: " << argv[0] << " parsegyro_data_file";
    cout << "  [exposure_time  delay_time]\n" << endl;
    return 0;
  }

  cout << "\nExtracting gyrodata from " << argv[0] << endl;

  if (argc > 2) {
    float expos_time = atof(argv[2]);
    if ((expos_time <= 0) || (expos_time == HUGE_VAL)) {
      cout << "\nEntered exposure time defaulted to 0.05999 s" << endl;
    } else {
      time_exposure = timeFromSeconds<float>(expos_time);
    }
  }

  if (argc > 3) {
    float dela_time = atof(argv[3]);
    if ((dela_time <= 0) || (dela_time == HUGE_VAL)) {
      cout << "\nEntered delay time defaulted to 0.0 s" << endl;
    } else {
      time_delay = timeFromSeconds<float>(dela_time);
    }
  }

  cout << "\nExposure time:   " << timeToSeconds<float>(time_exposure) << endl;
  cout << "Delay time   :   " << timeToSeconds<float>(time_delay) << endl;

  dfile_rows = 0;

  ifstream datafile(argv[1]);
  if (datafile.is_open()) {
    while (datafile.good()) {
      char *pcs;  // pointer to string position
      dfile_rows++;
      getline(datafile, line);

      // find the capture timestamp
      pcs = strstr(&line[0], CAPTURE_MARK);
      if (pcs != NULL) {
        char *pEnd;
        pcs += strlen(CAPTURE_MARK);  //
        capture_timestamp = strtol(pcs, &pEnd, 10);
      }

      // find the gyro samples number
      pcs = strstr(&line[0], GYRO_SAMPLES_MARK);
      if (pcs != NULL) {
        pcs += strlen(GYRO_SAMPLES_MARK);  //
        N = atoi(pcs);
      }

      // cout << dfile_rows << "  " << line << endl;
    }
    datafile.close();
  } else {
    cout << '\n' << argv[1] << " couldn't be opened.\n";
    return 0;
  }

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  Time *timestamp = new Time[N];  // timestamp [ns]

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  GyroDataCorrection<float> gyroCorrector(1000000, SAMPL_AVER);

  // read the values from file
  ifstream datafile1(argv[1]);
  for (int j = 0; j < N; j++)  // first row is empty
  {
    datafile1 >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];  // skip the first row
    // cout << j << "  " << timestamp[j] << "  " << wx[j] << "  " << wy[j] << "
    // " << wz[j] << endl;
  }
  datafile1.close();

  // put the values to container and correct them
  for (int j = 0; j < N; j++) {
    GyroDataCorrection<float>::GyroSample_t gyroSample(
        timestamp[j], Vector<float, 3>(wx[j], wy[j], wz[j]));

    // correct the gyro data
    CalcAngularVelocity(gyroCorrector, gyroSample);
  }

  // search for the time interval of exposure
  int j_i{}, j_f{};
  bool flag1 = 1, flag2 = 1;  // initial point, final point, control flags
  for (int j = 0; j < N - 20 - 1; j++) {
    TimeOrientation<Versor<float>> TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    if ((TEST.time >
         capture_timestamp - time_exposure - time_readout + time_delay) &&
        (flag1)) {
      j_i = j - 1;
      flag1 = 0;
    }

    if ((TEST.time > capture_timestamp - time_readout + time_delay) &&
        (flag2)) {
      j_f = j;
      flag2 = 0;
    }
  }

  cout << "\nFirst sample: " << j_i << endl;
  cout << "Last sample : " << j_f << '\n' << endl;

  // first timestamp
  TimeOrientation<Versor<float>> TEST1 =
      angularVelocityIntegrator.orientations().orientations()[0];
  Time time_zero = TEST1.time;  // gyro capture time
  float capture_rel_time = timeToSeconds<float>(capture_timestamp - time_zero);

  cout << "Relative capture time: " << capture_rel_time << '\n' << endl;

  // open the output file for writing
  const char gyro_suffix[] = "_gyro.txt";
  char *gyro_data_file = new char[strlen(argv[1]) + strlen(gyro_suffix)];

  strcpy(gyro_data_file, argv[1]);
  char *pch = strchr(gyro_data_file, '.');
  strncpy(pch, gyro_suffix, strlen(gyro_suffix) + 1);

  /*
  strcpy(gyro_data_file,argv[1]);
  strcat(gyro_data_file,gyro_suffix);
  */

  ofstream output_file(gyro_data_file);

  if (output_file.is_open()) {
    cout << "Data written in:  " << gyro_data_file << '\n' << endl;
  } else {
    cout << "\nUnable to open " << gyro_data_file << endl;
    cout << "Printing on screen instead\n" << endl;
  }

  for (int j = 0; j < N - SAMPL_AVER - 1; j++) {
    TimeOrientation<Test::Math::Versor<float>> TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    // custom created rotation angle calculation routine
    float w, x, y, z, rotAxis[3];
    float halfedAngle, angle, sinHalfedAngle, mult;
    // versor components
    x = TEST.orientation.x();
    y = TEST.orientation.y();
    z = TEST.orientation.z();
    w = TEST.orientation.w();
    if (w > 0.9) {
      sinHalfedAngle = sqrt(x * x + y * y + z * z);
      halfedAngle = asin(sinHalfedAngle);
      angle = halfedAngle * 2;
    } else {
      halfedAngle = acos(w);
      angle = halfedAngle * 2;
      sinHalfedAngle = sin(halfedAngle);
    }

    mult = (sinHalfedAngle == 0 ? 1 : angle / sinHalfedAngle);
    rotAxis[0] = mult * x;
    rotAxis[1] = mult * y;
    rotAxis[2] = mult * z;

    if (output_file.is_open()) {
      output_file << timeToSeconds<float>(TEST.time - time_zero) << "  ";
      // print the values
      output_file << rotAxis[0] << "  " << rotAxis[1] << "  " << rotAxis[2];
      // print the values between exposure points
      if ((j >= j_i) && (j <= j_f))
        output_file << "  " << rotAxis[0] << "  " << rotAxis[1] << "  "
                    << rotAxis[2];
      output_file << endl;
    } else {
      cout << timeToSeconds<float>(TEST.time - time_zero) << "  ";
      // print the values
      cout << rotAxis[0] << "  " << rotAxis[1] << "  " << rotAxis[2];
      // print the values between exposure points
      if ((j >= j_i) && (j <= j_f))
        cout << "  " << rotAxis[0] << "  " << rotAxis[1] << "  " << rotAxis[2];
      cout << endl;
    }
  }

  // close the output file
  output_file.close();

  // Gnuplot file graph creation
  FILE *gp;
#ifdef WIN32
  gp = _popen("gnuplot -persist", "w");
#else
  gp = popen("gnuplot -persist", "w");
#endif

  if (gp == NULL) {
    cout << "\nNo access to Gnuplot!\n" << endl;
    return -1;
  }

  // open the output ps file for writing
  const char ps_suffix[] = "_gyro.ps";
  char *ps_data_file = new char[strlen(argv[1]) + strlen(ps_suffix)];

  strcpy(ps_data_file, argv[1]);
  pch = strchr(ps_data_file, '.');
  strncpy(pch, ps_suffix, strlen(ps_suffix) + 1);

  fprintf(gp, "set term postscript enhanced solid lw 3 color\n");
  fprintf(gp, "set output \"%s\"\n", ps_data_file);
  fprintf(gp, "set xlabel \"Relative time, s\"\n");
  fprintf(gp, "set ylabel  \"Rotation, degrees\"\n");
  fprintf(gp,
          "plot \"%s\" using 1:2 title \"X rotation\" with line 1,  \"%s\" "
          "using 1:5 title \"exposure\" with line 4\n",
          gyro_data_file, gyro_data_file);
  fprintf(gp,
          "plot \"%s\" using 1:3 title \"Y rotation\" with line 2,  \"%s\" "
          "using 1:6 title \"exposure\" with line 4\n",
          gyro_data_file, gyro_data_file);
  fprintf(gp,
          "plot \"%s\" using 1:4 title \"Z rotation\" with line 3,  \"%s\" "
          "using 1:7 title \"exposure\" with line 4\n",
          gyro_data_file, gyro_data_file);
  fprintf(gp,
          "plot \"%s\" using 1:2 title \"X rotation\" with line 1,  \"%s\" "
          "using 1:3 title \"Y rotation\" with line 2",
          gyro_data_file, gyro_data_file);
  fprintf(gp,
          " , \"%s\" using 1:4 title \"Z rotation\" with line 3, \"%s\" using "
          "1:5 title \"exposure\" with line 4",
          gyro_data_file, gyro_data_file);
  fprintf(gp,
          " , \"%s\" using 1:6 title \"\" with line 4, \"%s\" using 1:7 title "
          "\"\" with line 4\n",
          gyro_data_file, gyro_data_file);

  cout << "Post Script written in:  " << ps_data_file << '\n' << endl;

#ifdef WIN32
  _pclose(gp);
#else
  pclose(gp);
#endif

  // release the memory
  delete[] gyro_data_file;
  delete[] ps_data_file;
  delete[] timestamp;
  delete[] wx;
  delete[] wy;
  delete[] wz;

  return 0;
}
