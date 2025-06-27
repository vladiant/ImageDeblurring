/*
 * Simple program to perform
 * integration of gyro data
 * and calculate the kernel
 * by direct reading of the file
 * sparse matrix used for kernel storage
 *
 * Code taken from parsergyro.cpp program
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// time for readout in seconds
const float time_readout = 0.03283;

// exposure time in seconds - to be read from EXIF Data!!!
const float time_exposure = 0.0599;

// Field of view for intrinsic camera matrix K
const float FOV = 2592 * 195.5 / 189;

// time delay for the timestamps - to be calibrated
const float time_delay = 0.02;

// parse_gyro_data segment
struct DataSample {
  unsigned long long int timestamp;
  float x;
  float y;
  float z;
  int extra;
};
typedef struct DataSample DataSample;

struct MovementDataHeader {
  char md_identifier[16];
  long long int md_capture_timestamp;
  unsigned int total_gyro_samples;
  unsigned int last_sample_index;
  long long int capture_command_timestamp;
};

typedef struct MovementDataHeader MovementDataHeader;

unsigned int segment_len = 16384;

DataSample *gyroSamples;
MovementDataHeader *gyroDataHeader;

int main(int argc, char *argv[]) {
  // time for capture end in seconds
  float time_capture_end;

  // time for capture start in seconds
  float time_capture_start;

  // parse_gyro_data segment
  FILE *pFile;
  int filesize;
  unsigned int total_samples, last_sample;
  char *buf;

  if (argc < 2) {
    cout << "\nUsage: parsegyro <jpeg file> \n";
    return 0;
  }

  pFile = fopen(argv[1], "r");

  if (pFile == NULL) {
    cout << "\nFile couldn't be opened.\n";
    return 0;
  }

  buf = new char[segment_len];
  gyroDataHeader = (MovementDataHeader *)malloc(sizeof(MovementDataHeader));

  fseek(pFile, 0, SEEK_END);
  long size = ftell(pFile);
  size -= segment_len;
  fseek(pFile, size, SEEK_SET);
  fread(buf, sizeof(char), segment_len, pFile);

  memcpy(gyroDataHeader, buf, sizeof(MovementDataHeader));

  total_samples = gyroDataHeader->total_gyro_samples;

  // set number of samples for further treatment
  int N = total_samples;
  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  long *timestamp = new long[N];  // timestamp [ns]
  int j = 0;                      // position for data transfer

  last_sample = gyroDataHeader->last_sample_index;

  gyroSamples = (DataSample *)malloc(sizeof(DataSample) * total_samples);

  memcpy(gyroSamples, buf + sizeof(MovementDataHeader),
         sizeof(DataSample) * total_samples);

  // print the gyro data parameters
  cout << "\n# Capture timestamp: "
       << gyroDataHeader->md_capture_timestamp -
              gyroSamples[last_sample].timestamp
       << "\n";
  cout << "# User press timestamp: "
       << gyroDataHeader->capture_command_timestamp -
              gyroSamples[last_sample].timestamp
       << "\n";
  cout << "# Total gyro samples: " << gyroDataHeader->total_gyro_samples
       << "\n\n";

  time_capture_end = (gyroDataHeader->md_capture_timestamp -
                      gyroSamples[last_sample].timestamp) /
                     1e9;
  time_capture_start = time_capture_end - time_exposure - time_readout;
  cout << "# Capture time start " << time_capture_start << "\n";
  cout << "# Capture time end " << time_capture_end << "\n\n";

  for (int i = last_sample; i < total_samples; i++, j++) {
    timestamp[j] =
        gyroSamples[i].timestamp - gyroSamples[last_sample].timestamp;
    wx[j] = gyroSamples[i].x;
    wy[j] = gyroSamples[i].y;
    wz[j] = gyroSamples[i].z;
  }

  for (int i = 0; i < last_sample; i++, j++) {
    timestamp[j] =
        gyroSamples[i].timestamp - gyroSamples[last_sample].timestamp;
    wx[j] = gyroSamples[i].x;
    wy[j] = gyroSamples[i].y;
    wz[j] = gyroSamples[i].z;
  }

  // data transfer completed, release memory
  delete[] buf;
  free(gyroSamples);
  free(gyroDataHeader);
  fclose(pFile);

  // time, angular positions
  float *time = new float[N];    // time [s]
  float *Thetax = new float[N];  // angle x
  float *Thetay = new float[N];  // angle y
  float *Thetaz = new float[N];  // angle z

  for (int j = 0; j < N; j++) {
    /*
    // read the values
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];
    */
    /*
    //test
    timestamp[j]=1e7*j;
    wx[j]=CV_PI*0.01;
    wy[j]=0.0;
    wz[j]=0.0;
    */
    if (j > 0) {
      float Rt[9];      // rotation matrix
      float Rti[9];     // inverse of rotation matrix
      float ThetaR[3];  // angle of rotation
      float dTheta[3];  // angle of rotation change

      // calculate time interval
      time[j] = (timestamp[j] - timestamp[0]) * 1e-9;

      // calculate unit vector of rotation
      float urx = Thetax[j - 1];
      float ury = Thetay[j - 1];
      float urz = Thetaz[j - 1];
      float sur = sqrt(urx * urx + ury * ury + urz * urz);
      float c, s, cc;

      if (sur != 0) {
        urx /= sur;
        ury /= sur;
        urz /= sur;
        c = cos(sur), s = sin(sur), cc = 1 - c;
        // inverse of the rotation matrix
        Rti[0] = urx * urx * cc + c;
        Rti[1] = urx * ury * cc + urz * s;
        Rti[2] = urx * urz * cc - ury * s;
        Rti[3] = ury * urx * cc - urz * s;
        Rti[4] = ury * ury * cc + c;
        Rti[5] = ury * urz * cc + urx * s;
        Rti[6] = urz * urx * cc + ury * s;
        Rti[7] = urz * ury * cc - urx * s;
        Rti[8] = urz * urz * cc + c;
      } else {
        Rti[0] = 1.0;
        Rti[1] = 0.0;
        Rti[2] = 0.0;
        Rti[3] = 0.0;
        Rti[4] = 1.0;
        Rti[5] = 0.0;
        Rti[6] = 0.0;
        Rti[7] = 0.0;
        Rti[8] = 1.0;
      }

      // for (int i=0; i<9; i++) cout << Rti[i] << "  ";
      // cout << '\n';

      /*
      dTheta[0]=Rt[0]*wx[j-1]+Rt[1]*wy[j-1]+Rt[2]*wz[j-1];
      dTheta[1]=Rt[3]*wx[j-1]+Rt[4]*wy[j-1]+Rt[5]*wz[j-1];
      dTheta[2]=Rt[6]*wx[j-1]+Rt[7]*wy[j-1]+Rt[8]*wz[j-1];
      */

      dTheta[0] = Rti[0] * wx[j - 1] + Rti[1] * wy[j - 1] + Rti[2] * wz[j - 1];
      dTheta[1] = Rti[3] * wx[j - 1] + Rti[4] * wy[j - 1] + Rti[5] * wz[j - 1];
      dTheta[2] = Rti[6] * wx[j - 1] + Rti[7] * wy[j - 1] + Rti[8] * wz[j - 1];

      // new value of rotation
      Thetax[j] = Thetax[j - 1] + dTheta[0] * (time[j] - time[j - 1]);
      Thetay[j] = Thetay[j - 1] + dTheta[1] * (time[j] - time[j - 1]);
      Thetaz[j] = Thetaz[j - 1] + dTheta[2] * (time[j] - time[j - 1]);
      /*
      //calculation of the rotation matrix, based on actual angle,
      //to be applied to initial point
      urx=Thetax[j]; ury=Thetay[j]; urz=Thetaz[j];
      sur=sqrt(urx*urx+ury*ury+urz*urz);
      urx/=sur; ury/=sur; urz/=sur;

      c=cos(sur), s=sin(sur), cc=1-c;
      Rt[0]=urx*urx*cc+c;
      Rt[1]=urx*ury*cc-urz*s;
      Rt[2]=urx*urz*cc+ury*s;
      Rt[3]=ury*urx*cc+urz*s;
      Rt[4]=ury*ury*cc+c;
      Rt[5]=ury*urz*cc-urx*s;
      Rt[6]=urz*urx*cc-ury*s;
      Rt[7]=urz*ury*cc+urx*s;
      Rt[8]=urz*urz*cc+c;
      */
      // for (int i=0; i<9; i++) cout << Rt[i] << "  ";
      // cout << '\n';
      /*
      if (!((time[j]>time_capture_start)&&(time[j]<time_capture_end))) cout <<
      "# ";
      {
       cout << time[j] << "  " <<
      (Rt[0]*0.0+Rt[1]*0.0+Rt[2]*1.0)/(Rt[6]*0.0+Rt[7]*0.0+Rt[8]*1.0) << "  "
       << (Rt[3]*0.0+Rt[4]*0.0+Rt[5]*1.0)/(Rt[6]*0.0+Rt[7]*0.0+Rt[8]*1.0) <<
      '\n';
      }
      */

    } else {
      Thetax[0] = 0.0;
      Thetay[0] = 0.0;
      Thetaz[0] = 0.0;

      time[0] = 0.0;
    }

    // cout << time[j] << "  " << wx[j] << "  " << wy[j] << "  " << wz[j] << "
    // "; cout  << "  " << Thetax[j] << "  " << Thetay[j] << "  " << Thetaz[j]
    // << '\n'; cout << time[j] << "  " << wix[j] << "  " << wiy[j] << "  " <<
    // wiz[j] << '\n';
  }

  // image processing
  cv::Mat imgi = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
  cv::Mat imgk = imgi.clone();

  int imw = imgi.cols, imh = imgi.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels
  int r0x = 0 / 2,
      r0y = 0 / 2;       // pixel where the movement points will be drawn
  float x1, y1, x0, y0;  // current coordinates
  char *nmK = new char[strlen(argv[1]) + 1];

  // create the name for the output file
  j = 0;
  do {
    *(nmK + j) = *(argv[1] + j);
    j++;
  } while (*(argv[1] + j) != '.');

  *(nmK + j) = '_';
  *(nmK + j + 1) = 'K';

  do {
    *(nmK + j + 2) = *(argv[1] + j);
    j++;
  } while (*(argv[1] + j - 1) != '\0');

  // Rolling shutter shift
  cout << "# Rolling shutter time shift: " << time_readout / imh << "\n\n";

  // selection of the points for kernel interpolation
  int j_i, j_f, flag1 = true,
                flag2 = true;  // initial point, final point, control flags

  for (int j = 0; j < N; j++) {
    if ((time[j] > time_capture_start + time_delay) && (flag1)) {
      j_i = j - 1;
      flag1 = false;
    }

    if ((time[j] > time_capture_start + time_exposure + time_delay) &&
        (flag2)) {
      j_f = j;
      flag2 = false;
    }
  }
  cout << "# Border samples of blurring: from  " << j_i << " to " << j_f
       << '\n';

  // extract the needed Theta and time
  int bsamples = j_f - j_i + 1;
  float *Thetabx = new float[bsamples];
  float *Thetaby = new float[bsamples];
  float *Thetabz = new float[bsamples];
  float *timeb = new float[bsamples];

  for (int j = j_i; j < j_f + 1; j++) {
    Thetabx[j - j_i] = Thetax[j];
    Thetaby[j - j_i] = Thetay[j];
    Thetabz[j - j_i] = Thetaz[j];
    timeb[j - j_i] = time[j];
  }

  // loops to draw the kernels
  for (int r0x = 50; r0x < imgk.rows; r0x += 100)
    for (int r0y = 50; r0y < imgk.cols; r0y += 100) {
      // non homogenious coordinates
      float x0 = (r0x - u0) / (FOV), y0 = (r0y - v0) / (FOV);

      // (x,y) coordinates of points, which correspond to blurring
      // alongside with angles an time
      float *xb = new float[bsamples];
      float *yb = new float[bsamples];
      float xbi, ybi, xbf, ybf, xbmax = 0, ybmax = 0;
      int *bstep = new int[bsamples - 1];  // steps for interpolation
      int bsteps = 0;

      // set the array of (x,y) points for blurring
      // point movement for the whole time interval
      for (int j = 0; j < bsamples; j++) {
        float Rt[9];  // rotation matrix

        float c, s, cc;

        // calculation of the rotation matrix, based on actual angle,
        // to be applied to initial point
        float urx = Thetabx[j], ury = Thetaby[j], urz = Thetabz[j];
        float sur = sqrt(urx * urx + ury * ury + urz * urz);
        urx /= sur;
        ury /= sur;
        urz /= sur;

        c = cos(sur), s = sin(sur), cc = 1 - c;
        Rt[0] = urx * urx * cc + c;
        Rt[1] = urx * ury * cc - urz * s;
        Rt[2] = urx * urz * cc + ury * s;
        Rt[3] = ury * urx * cc + urz * s;
        Rt[4] = ury * ury * cc + c;
        Rt[5] = ury * urz * cc - urx * s;
        Rt[6] = urz * urx * cc - ury * s;
        Rt[7] = urz * ury * cc + urx * s;
        Rt[8] = urz * urz * cc + c;

        x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        xb[j] = x1 * (FOV) + u0 - r0x;
        yb[j] = y1 * (FOV) + v0 - r0y;

        // cout << "#  " << xb[j] << "  " << yb[j] << "  " << timeb[j] << '\n';
      }

      // set the initial and final points
      xbi = (xb[1] - xb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            xb[0];
      ybi = (yb[1] - yb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            yb[0];

      // cout << "\n# Initial point:  " << xbi << "  " << ybi <<  '\n';

      xbf = (xb[bsamples - 1] - xb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            xb[bsamples - 2];
      ybf = (yb[bsamples - 1] - yb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            yb[bsamples - 2];

      // cout << "# Final point:    " << xbf << "  " << ybf << '\n';

      // the final set of points
      xb[0] = 0;
      yb[0] = 0;
      xb[bsamples - 1] = xbf - xbi;
      yb[bsamples - 1] = ybf - ybi;
      for (int j = 1; j < bsamples - 1; j++) {
        xb[j] -= xbi;
        yb[j] -= ybi;
        if (abs(xb[j]) > abs(xbmax)) xbmax = xb[j];
        if (abs(yb[j]) > abs(ybmax)) ybmax = yb[j];
      }
      if (abs(xb[0]) > abs(xbmax)) xbmax = xb[0];
      if (abs(yb[0]) > abs(ybmax)) ybmax = yb[0];
      if (abs(xb[bsamples - 1]) > abs(xbmax)) xbmax = xb[bsamples - 1];
      if (abs(yb[bsamples - 1]) > abs(ybmax)) ybmax = yb[bsamples - 1];

      // print them
      // for (int j=0;j<bsamples; j++) cout << "# " << time[j] << "  " << xb[j]
      // << "  " << yb[j] << '\n';

      // calculate number of steps
      for (int j = 0; j < bsamples - 1; j++) {
        bstep[j] = int(2.0 * sqrt((xb[j + 1] - xb[j]) * (xb[j + 1] - xb[j]) +
                                  (yb[j + 1] - yb[j]) * (yb[j + 1] - yb[j])) +
                       0.5);
        bsteps += bstep[j];
      }

      // draw the kernel for this point
      // IplImage *kernel=cv::Mat(cv::Size(2*int(abs(ybmax)+0.5)+10,
      // 2*int(abs(xbmax)+0.5)+10),IPL_DEPTH_32F,1); cvSetZero(kernel);

      // sparse kernel
      int dims[] = {2 * int(abs(ybmax) + 0.5) + 10,
                    2 * int(abs(xbmax) + 0.5) + 10};
      cv::SparseMat skernel(2, dims, CV_32F);

      // set the kernel values
      for (int j = 0; j < bsamples - 1; j++) {
        float step = 1.0 / bstep[j];
        for (int jj = 0; jj < bstep[j]; jj++) {
          float xd =
              -((yb[j + 1] - yb[j]) * jj / bstep[j] + yb[j]) + dims[0] / 2;
          float yd =
              -((xb[j + 1] - xb[j]) * jj / bstep[j] + xb[j]) + dims[1] / 2;
          /*
          ((float*)(kernel.data +
          int(yd)*(kernel.step)))[int(xd)]+=(int(xd)-xd+1)*(int(yd)-yd+1)*step;
          ((float*)(kernel.data +
          int(yd+1)*(kernel.step)))[int(xd)]+=(int(xd)-xd+1)*(yd-int(yd))*step;
          ((float*)(kernel.data +
          int(yd)*(kernel.step)))[int(xd+1)]+=(xd-int(xd))*(int(yd)-yd+1)*step;
          ((float*)(kernel.data +
          int(yd+1)*(kernel.step)))[int(xd+1)]+=(xd-int(xd))*(yd-int(yd))*step;
          */
          // sparse kernel
          int idx[] = {int(xd), int(yd)};
          skernel.ref<float>(idx) +=
              (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (int(xd) - xd + 1) * (yd - int(yd)) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd);
          skernel.ref<float>(idx) += (xd - int(xd)) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (xd - int(xd)) * (yd - int(yd)) * step;
        }
      }

      float kernelsum = 0;
      /*
      for (int row=0; row<kernel.rows;row++)
              for (int col=0; col<kernel.cols;col++)
                      if (kernelsum<((float*)(kernel.data +
      row*(kernel.step)))[col]) kernelsum=((float*)(kernel.data +
      row*(kernel.step)))[col];
                      //kernelsum+=((float*)(kernel.data +
      row*(kernel.step)))[col];

      */
      // sparse kernel sum
      cv::SparseMatIterator it;
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        float val = it.value<float>();
        if (kernelsum < val) kernelsum = val;
        // kernelsum+=val;
      }
      /*
      //normalize
      for (int row=0; row<kernel.rows;row++)
              for (int col=0; col<kernel.cols;col++)
      ((float*)(kernel.data + row*(kernel.step)))[col]/=kernelsum;
      */
      // normalize sparse kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        it.value<float>() /= kernelsum;
      }
      /*
      //set the kernel from the sparse kernel
      for(CvSparseNode *node = cvInitSparseMatIterator( skernel, &it ); node !=
      0; node = cvGetNextSparseNode( &it ))
              {
                      int* idx = CV_NODE_IDX(skernel,node);
                      float val = *(float*)CV_NODE_VAL( skernel, node );
                      ((float*)((kernel).data +
      (kernel).step*(idx[1])))[(idx[0])]=val;
              }
      */
      // cout << "# Highest value:  " << kernelsum << '\n';
      /*
      //set the big kernel image
      for (int row=0; row<kernel.rows;row++)
              for (int col=0; col<kernel.cols;col++)
                      if
      ((int(r0x+row-kernel.rows/2)<imgk.rows)&&(int(r0x+row-kernel.rows/2)>=0)&&(int(r0y+col-kernel.cols/2)<imgk.cols)&&(int(r0y+col-kernel.cols/2)>=0))
                              if (((float*)(kernel.data +
      row*(kernel.step)))[col]>0)
                                      //((uchar*)(imgk.data +
      int(r0x+row)*(imgk.step)))[int(r0y+col)]=((float*)(kernel.data
      + row*(kernel.step)))[col]*255;
                                      {
                                              //red channel
                                              ((uchar*)(imgk.data +
      int(r0x+row-kernel.rows/2)*(imgk.step)))[int(r0y+col-kernel.cols/2)*imgk.channels()+2]
                                              =((float*)(kernel.data +
      row*(kernel.step)))[col]*255;
                                              ((uchar*)(imgk.data +
      int(r0x+row-kernel.rows/2)*(imgk.step)))[int(r0y+col-kernel.cols/2)*imgk.channels()+1]=0;
      //green
                                              ((uchar*)(imgk.data +
      int(r0x+row-kernel.rows/2)*(imgk.step)))[int(r0y+col-kernel.cols/2)*imgk.channels()+0]=0;
      //blue
                                      }
      */
      // set the big kernel image from sparse matrix
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();
        if ((int(r0x + idx[1] - dims[1] / 2) < imgk.rows) &&
            (int(r0x + idx[1] - dims[1] / 2) >= 0) &&
            (int(r0y + idx[0] - dims[0] / 2) < imgk.cols) &&
            (int(r0y + idx[0] - dims[0] / 2) >= 0))
          if (val > 0) {
            // red channel
            ((uchar *)(imgk.data +
                       int(r0x + idx[1] - dims[1] / 2) * (imgk.step)))
                [int(r0y + idx[0] - dims[0] / 2) * imgk.channels() + 2] =
                    val * 255;
            ((uchar *)(imgk.data +
                       int(r0x + idx[1] - dims[1] / 2) * (imgk.step)))
                [int(r0y + idx[0] - dims[0] / 2) * imgk.channels() + 1] =
                    0;  // green
            ((uchar *)(imgk.data +
                       int(r0x + idx[1] - dims[1] / 2) * (imgk.step)))
                [int(r0y + idx[0] - dims[0] / 2) * imgk.channels() + 0] =
                    0;  // blue
          }
      }

      delete[] xb;
      delete[] yb;
      delete[] bstep;
      // cv::namedWindow("Kernel", 0);
      // cv::imshow("Kernel", kernel);
      // cv::waitKey(2);
    }

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  // Kernel save image
  cout << "\n# Kernel save image: " << nmK << '\n';
  cv::imwrite(nmK, imgk);

  // displays image
  cv::namedWindow("Initial", IMG_WIN);
  cv::namedWindow("Kernels", 0);
  if (IMG_WIN == 0) cv::resizeWindow("Initial", 1080, 720);
  cv::imshow("Initial", imgi);
  cv::imshow("Kernels", imgk);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Kernels");

  delete[] Thetabx;
  delete[] Thetaby;
  delete[] Thetabz;
  delete[] timeb;
  delete[] nmK;

  delete[] timestamp;
  delete[] time;
  delete[] wx;
  delete[] wy;
  delete[] wz;
  delete[] Thetax;
  delete[] Thetay;
  delete[] Thetaz;

  return (0);
}
