/*
 * This code generates
 * motion paths
 * for kernel creation
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void Line(cv::Mat &imga, float x1, float y1, float x2,
          float y2)  // generates the image
{
  int i1, j1, i2, j2, iter = 0;  // temporal coordinates for points
  int xl1, yl1, xl2, yl2;        // start and end points
  double step;                   // step for the line increment
  float dx = x2 - x1, dy = y2 - y1, x, y, s1, s2, s3;
  double mx, mn;  //=step*sqrt(dx*dx+dy*dy)

  imga = 0;
  /*
  //first point
  ((float*)(imga.data +
  int(y1)*(imga.step)))[int(x1)]+=(int(x1)-x1+1)*(int(y1)-y1+1);
  ((float*)(imga.data +
  int(y1+1)*(imga.step)))[int(x1)]+=(int(x1)-x1+1)*(y1-int(y1));
  ((float*)(imga.data +
  int(y1)*(imga.step)))[int(x1+1)]+=(x1-int(x1))*(int(y1)-y1+1);
  ((float*)(imga.data +
  int(y1+1)*(imga.step)))[int(x1+1)]+=(x1-int(x1))*(y1-int(y1));

  //last point
  if (!((x1==x2)&&(y1==y2)))
  {
          ((float*)(imga.data +
  int(y2)*(imga.step)))[int(x2)]+=(int(x2)-x2+1)*(int(y2)-y2+1);
          ((float*)(imga.data +
  int(y2+1)*(imga.step)))[int(x2)]+=(int(x2)-x2+1)*(y2-int(y2));
          ((float*)(imga.data +
  int(y2)*(imga.step)))[int(x2+1)]+=(x2-int(x2))*(int(y2)-y2+1);
          ((float*)(imga.data +
  int(y2+1)*(imga.step)))[int(x2+1)]+=(x2-int(x2))*(y2-int(y2));
  }
  */
  if (abs(dx) > abs(dy)) {
    if (dx > 1) {
      // y derived from x
      xl2 = int(x2);
      step = dy / dx;
      /*
      // preferable treatment
      //first point
      if ((1-y1+int(y1))/(1-x1+int(x1))>=step)
      {
              ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1)]=0.5*(1-x1+int(x1))*(2*(1-y1+int(y1))-step*(1-x1+int(x1)));
              ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1)]=0.5*(1-x1+int(x1))*(2*(y1-int(y1))+step*(1-x1+int(x1)));
              //cout <<
      0.5*(1-x1+int(x1))*(2*(1-y1+int(y1))-step*(1-x1+int(x1))) << "  " <<
      0.5*(1-x1+int(x1))*(2*(y1-int(y1))+step*(1-x1+int(x1))) << '\n';
              //part 2
              if ((1-y1+int(y1))/(2-x1+int(x1))>=step)
              {
                      ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1+1)]=0.5*(2*(1-y1+int(y1))-step*(3.0-2*x1+2*int(x1)));
                      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1+1)]=0.5*(2*(y1-int(y1))+step*(3.0-2*x1+2*int(x1)));
                      cout << '=';
              }
              else
              {
                      s1=(1-step*(1-x1+int(x1)))*(1-step*(1-x1+int(x1)))/(2*step);
                      s2=(step-1+step*(1-x1+int(x1)))*(step-1+step*(1-x1+int(x1)))/(2*step);
                      ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1+1)]=s1;
                      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1+1)]=1-s1-s2;
                      ((float*)(imga.data +
      int(y1+2)*(imga.step)))[int(x1+1)]=s2; cout << '*';
              }
              //cout << 0.5*(2*(1-y1+int(y1))-step*(3-2*x1+2*int(x1))) << "  "
      << 0.5*(2*(y1-int(y1))+step*(3-2*x1+2*int(x1))) << '\n';
      }
      else
      {
              s1=(1-y1+int(y1))*(1-y1+int(y1))/(2*step);
              s2=0.5*step*(1-x1+int(x1)-(1-y1+int(y1))/step)*(1-x1+int(x1)-(1-y1+int(y1))/step);
              s3=(1-x1+int(x1))-s2-s1;
      //(1-y1+int(y1))*(1+y1-int(y1))/(2*step)+(1-x1+int(x1)+(1-(1-y1+int(y1))/step)*0.5*step);
              ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1)]=s1;
              ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1)]=s3;
              ((float*)(imga.data +
      int(y1+2)*(imga.step)))[int(x1)]=s2; cout << s1 << "  " << s3 << "
      " << s2 << '\n';

              //second part
              if ((2-y1+int(y1))/(2-x1+int(x1))>=step)
              {
                      s1=0.5*(2-step-2*(step*(1-x1+int(x1))+y1-int(y1)-1));
                      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1+1)]=s1;
                      ((float*)(imga.data +
      int(y1+2)*(imga.step)))[int(x1+1)]=1-s1; cout << '+';
              }
              else
              {
                      s1=(2+int(y1)-y1-step*(1-x1+int(x1)))*(2+int(y1)-y1-step*(1-x1+int(x1)))/(2*step);
                      s2=(y1-int(y1)-2+step*(1-x1+int(x1)))*(y1-int(y1)-2+step*(1-x1+int(x1)))/(2*step);
                      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1+1)]=s1;
                      ((float*)(imga.data +
      int(y1+2)*(imga.step)))[int(x1+1)]=s2; cout << '/';
              }
      }
      //((float*)(imga.data +
      int(y)*(imga.step)))[int(x1)+1]+=(int(y)-y+1)*0.5*(1+int(x1)-x1);
      //((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x1)+1]+=(y-int(y))*0.5*(1+int(x1)-x1);
      //((float*)(imga.data +
      int(y)*(imga.step)))[int(x1)+1]+=(int(y)-y+1)*0.5*(1+int(x1)-x1);
      //((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x1)+1]+=(y-int(y))*0.5*(1+int(x1)-x1);
      //((float*)(imga.data +
      int(y)*(imga.step)))[int(x1)]+=(int(y)-y+1)*0.5*(1+int(x1)-x1);
      //((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x1)]+=(y-int(y))*0.5*(1+int(x1)-x1);
      */
      /*
      ((float*)(imga.data + int(y1)*(imga.step)))[int(0)]=1.0;
      ((float*)(imga.data + int(y1+1)*(imga.step)))[int(0)]=1.0;
      ((float*)(imga.data + int(0)*(imga.step)))[int(x1)]=1.0;
      ((float*)(imga.data + int(0)*(imga.step)))[int(x1+1)]=1.0;
      //cout << ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1+1)] << '\n';
      y=y1+step*(int(x1)+1.5-x1);
      //((float*)(imga.data +
      int(y)*(imga.step)))[int(x1+1)]+=(int(y)-y+1)*0.5;
      //((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x1+1)]+=(y-int(y))*0.5; cout <<
      ((float*)(imga.data + int(y)*(imga.step)))[int(x1+1)] << "  "
      << ((float*)(imga.data + int(y)*(imga.step)))[int(x1+1)] <<
      '\n';

      //second point
      */
      // line drawing - works OK
      for (xl1 = int(x1) + 2; xl1 < xl2; xl1++) {
        y = y1 + step * (xl1 - x1 - 0.5);
        ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] +=
            (int(y) - y + 1) * 0.5;
        ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] +=
            (y - int(y)) * 0.5;
        y = y1 + step * (xl1 - x1 + 0.5);
        ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] +=
            (int(y) - y + 1) * 0.5;
        ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] +=
            (y - int(y)) * 0.5;
      }
      /*
      //adjustment - first point
      y=y1+step*(int(x1)-x1+0.5);
      ((float*)(imga.data +
      int(y)*(imga.step)))[int(x1)]+=(int(y)-y+1)*0.5;
      ((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x1)]+=(y-int(y))*0.5;

      ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1)]*=(int(x1)-x1+1)*(int(y1)-y1+1);
      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1)]*=(int(x1)-x1+1)*(1-(y1-int(y1)));
      ((float*)(imga.data +
      int(y1)*(imga.step)))[int(x1+1)]*=(1+(x1-int(x1)))*(int(y1)-y1+1);
      ((float*)(imga.data +
      int(y1+1)*(imga.step)))[int(x1+1)]*=(1-(x1-int(x1)))*(1-(y1-int(y1)));

      //last point
      y=y1+step*(int(x2)-x1-0.5);
      ((float*)(imga.data +
      int(y)*(imga.step)))[int(x2)]+=(int(y)-y+1)*0.5;
      ((float*)(imga.data +
      int(y+1)*(imga.step)))[int(x2)]+=(y-int(y))*0.5;

      ((float*)(imga.data +
      int(y2)*(imga.step)))[int(x2)]*=(int(x2)-x2+1)*(int(y2)-y2+1);
      ((float*)(imga.data +
      int(y2+1)*(imga.step)))[int(x2)]*=(int(x2)-x2+1)*(1-(y2-int(y2)));
      ((float*)(imga.data +
      int(y2)*(imga.step)))[int(x2+1)]*=(1-(x2-int(x2)))*(int(y2)-y2+1);
      ((float*)(imga.data +
      int(y2+1)*(imga.step)))[int(x2+1)]*=(1-(x2-int(x2)))*(1-(y2-int(y2)));
       */
    } else {
      if (dx < -1) {
        // swap start and end points
        xl2 = int(x1);
        step = dy / dx;
        for (xl1 = int(x2) + 2; xl1 < xl2; xl1++) {
          y = y2 + step * (xl1 - x2 - 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] =
              (int(y) - y + 1);
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] =
              (y - int(y));
          y = y1 + step * (xl1 - x2 + 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] +=
              (int(y) - y + 1) * 0.5;
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] +=
              (y - int(y)) * 0.5;
        }
      }
    }
  } else if (abs(dy) > abs(dx)) {
    if (dy > 1) {
      // x derived from y
      yl2 = int(y2);
      step = dx / dy;
      // line drawing
      for (yl1 = int(y1) + 2; yl1 < yl2; yl1++) {
        x = x1 + step * (yl1 - y1 - 0.5);
        ((float *)(imga.data + int(yl1) * (imga.step)))[int(x)] +=
            (int(x) - x + 1) * 0.5;
        ((float *)(imga.data + int(yl1) * (imga.step)))[int(x + 1)] +=
            (x - int(x)) * 0.5;
        x = x1 + step * (yl1 - y1 + 0.5);
        ((float *)(imga.data + int(yl1) * (imga.step)))[int(x)] +=
            (int(x) - x + 1) * 0.5;
        ((float *)(imga.data + int(yl1) * (imga.step)))[int(x + 1)] +=
            (x - int(x)) * 0.5;
      }
    } else {
      if (dy < -1) {
        // swap start and end points
        yl2 = int(y1);
        step = dx / dy;

        // line drawing
        for (yl1 = int(y2) + 2; yl1 < yl2; yl1++) {
          x = x1 + step * (yl1 - y2 - 0.5);
          ((float *)(imga.data + int(yl1) * (imga.step)))[int(x)] +=
              (int(x) - x + 1) * 0.5;
          ((float *)(imga.data + int(yl1) * (imga.step)))[int(x + 1)] +=
              (x - int(x)) * 0.5;
          x = x1 + step * (yl1 - y2 + 0.5);
          ((float *)(imga.data + int(yl1) * (imga.step)))[int(x)] +=
              (int(x) - x + 1) * 0.5;
          ((float *)(imga.data + int(yl1) * (imga.step)))[int(x + 1)] +=
              (x - int(x)) * 0.5;
        }
      }
    }
  } else {
    if (abs(dx) > 1) {
      // diagonal
      if (dx > 0) {
        xl2 = int(x2);
        step = dy / dx;
        /*
        //first point
        if ((1-y1+int(y1))/(1-x1+int(x1))>=step)
        {
                ((float*)(imga.data +
        int(y1)*(imga.step)))[int(x1)]=0.5*(1-x1+int(x1))*(2*(1-y1+int(y1))-step*(1-x1+int(x1)));
                ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1)]=0.5*(1-x1+int(x1))*(2*(y1-int(y1))+step*(1-x1+int(x1)));
                //cout <<
        0.5*(1-x1+int(x1))*(2*(1-y1+int(y1))-step*(1-x1+int(x1))) << "  " <<
        0.5*(1-x1+int(x1))*(2*(y1-int(y1))+step*(1-x1+int(x1))) << '\n';
                //part 2
                if ((1-y1+int(y1))/(2-x1+int(x1))>=step)
                {
                        ((float*)(imga.data +
        int(y1)*(imga.step)))[int(x1+1)]=0.5*(2*(1-y1+int(y1))-step*(3.0-2*x1+2*int(x1)));
                        ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1+1)]=0.5*(2*(y1-int(y1))+step*(3.0-2*x1+2*int(x1)));
                        cout << '=';
                }
                else
                {
                        s1=(1-step*(1-x1+int(x1)))*(1-step*(1-x1+int(x1)))/(2*step);
                        s2=(step-1+step*(1-x1+int(x1)))*(step-1+step*(1-x1+int(x1)))/(2*step);
                        ((float*)(imga.data +
        int(y1)*(imga.step)))[int(x1+1)]=s1;
                        ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1+1)]=1-s1-s2;
                        ((float*)(imga.data +
        int(y1+2)*(imga.step)))[int(x1+1)]=s2; cout << '*';
                }
                //cout << 0.5*(2*(1-y1+int(y1))-step*(3-2*x1+2*int(x1))) << "  "
        << 0.5*(2*(y1-int(y1))+step*(3-2*x1+2*int(x1))) << '\n';
        }
        else
        {
                s1=(1-y1+int(y1))*(1-y1+int(y1))/(2*step);
                s2=0.5*step*(1-x1+int(x1)-(1-y1+int(y1))/step)*(1-x1+int(x1)-(1-y1+int(y1))/step);
                s3=(1-x1+int(x1))-s2-s1;
        //(1-y1+int(y1))*(1+y1-int(y1))/(2*step)+(1-x1+int(x1)+(1-(1-y1+int(y1))/step)*0.5*step);
                ((float*)(imga.data +
        int(y1)*(imga.step)))[int(x1)]=s1;
                ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1)]=s3;
                ((float*)(imga.data +
        int(y1+2)*(imga.step)))[int(x1)]=s2; cout << s1 << "  " << s3 << "
        " << s2 << '\n';

                //second part
                if ((2-y1+int(y1))/(2-x1+int(x1))>=step)
                {
                        s1=0.5*(2-step-2*(step*(1-x1+int(x1))+y1-int(y1)-1));
                        ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1+1)]=s1;
                        ((float*)(imga.data +
        int(y1+2)*(imga.step)))[int(x1+1)]=1-s1; cout << '+';
                }
                else
                {
                        s1=(2+int(y1)-y1-step*(1-x1+int(x1)))*(2+int(y1)-y1-step*(1-x1+int(x1)))/(2*step);
                        s2=(y1-int(y1)-2+step*(1-x1+int(x1)))*(y1-int(y1)-2+step*(1-x1+int(x1)))/(2*step);
                        ((float*)(imga.data +
        int(y1+1)*(imga.step)))[int(x1+1)]=s1;
                        ((float*)(imga.data +
        int(y1+2)*(imga.step)))[int(x1+1)]=s2; cout << '/';
                }
        }
        */

        // line drawing
        for (xl1 = int(x1) + 2; xl1 < xl2; xl1++) {
          y = y1 + step * (xl1 - x1 - 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] =
              (int(y) - y + 1) * 0.5;
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] =
              (y - int(y)) * 0.5;
          y = y1 + step * (xl1 - x1 + 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] +=
              (int(y) - y + 1) * 0.5;
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] +=
              (y - int(y)) * 0.5;
        }

      } else {
        xl2 = int(x1);
        step = dy / dx;
        // line drawing
        for (xl1 = int(x2) + 2; xl1 < xl2; xl1++) {
          y = y1 + step * (xl1 - x1 - 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] =
              (int(y) - y + 1) * 0.5;
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] =
              (y - int(y)) * 0.5;
          y = y1 + step * (xl1 - x1 + 0.5);
          ((float *)(imga.data + int(y) * (imga.step)))[int(xl1)] +=
              (int(y) - y + 1) * 0.5;
          ((float *)(imga.data + int(y + 1) * (imga.step)))[int(xl1)] +=
              (y - int(y)) * 0.5;
        }
      }
    }
  }
  /*
          for(x=x1, y=y1; (x<x2); x+=step*dx, y+=step*dy)
          {
                  ((float*)(imga.data +
     int(y)*(imga.step)))[int(x)]+=(int(x)-x+1)*(int(y)-y+1)/2;
                  ((float*)(imga.data +
     int(y+1)*(imga.step)))[int(x)]+=(int(x)-x+1)*(y-int(y))/2;
                  ((float*)(imga.data +
     int(y)*(imga.step)))[int(x+1)]+=(x-int(x))*(int(y)-y+1)/2;
                  ((float*)(imga.data +
     int(y+1)*(imga.step)))[int(x+1)]+=(x-int(x))*(y-int(y))/2;
                  //cout << x << "  " << y << '\n';
  */
}

void Line1(cv::Mat &imga, float x1, float y1, float x2,
           float y2)  // generates the line by brute force
{
  float xl, yl, xl1, yl1, xl2, yl2, step;
  const float stp = 20;  // number of steps

  imga = 0;

  if (abs(x1 - x2) > abs(y1 - y2)) {
    step = 1 / (abs(x1 - x2) * (stp));

    if (x1 < x2) {
      xl1 = x1;
      xl2 = x2;
      yl1 = y1;
      yl2 = y2;
    } else {
      xl1 = x2;
      xl2 = x1;
      yl1 = y2;
      yl2 = y1;
    }

    for (xl = xl1, yl = yl1; xl <= xl2;
         xl += (xl2 - xl1) * step, yl += (yl2 - yl1) * step) {
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (yl - int(yl)) * step;
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (yl - int(yl)) * step;
    }
  } else {
    if (!((x1 == x2) && (y1 == y2))) {
      step = 1 / (abs(y1 - y2) * (stp));
      if (y1 < y2) {
        xl1 = x1;
        xl2 = x2;
        yl1 = y1;
        yl2 = y2;
      } else {
        xl1 = x2;
        xl2 = x1;
        yl1 = y2;
        yl2 = y1;
      }

      for (xl = xl1, yl = yl1; yl <= yl2;
           xl += (xl2 - xl1) * step, yl += (yl2 - yl1) * step) {
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (yl - int(yl)) * step;
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (yl - int(yl)) * step;
      }
    } else {
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (y1 - int(y1));
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (y1 - int(y1));
    }
  }

  imga = imga * sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
                     1);  // normalization
}

// generates the line by brute force - optimized
void Line2(cv::Mat &imga, float x1, float y1, float x2, float y2) {
  float xl, yl, xl1, yl1, xl2, yl2, step;
  const float stp = 2;  // number of steps
  float stepx, stepy;

  imga = 0;

  if (abs(x1 - x2) > abs(y1 - y2)) {
    step = 1 / stp;

    if (x1 < x2) {
      xl1 = x1;
      xl2 = x2;
      yl1 = y1;
      yl2 = y2;
    } else {
      xl1 = x2;
      xl2 = x1;
      yl1 = y2;
      yl2 = y1;
    }

    stepy = step * (yl2 - yl1) / (xl2 - xl1);

    for (xl = xl1, yl = yl1; xl <= xl2; xl += step, yl += stepy) {
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (yl - int(yl)) * step;
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (yl - int(yl)) * step;
    }
  } else {
    if (!((x1 == x2) && (y1 == y2))) {
      step = 1 / stp;
      if (y1 < y2) {
        xl1 = x1;
        xl2 = x2;
        yl1 = y1;
        yl2 = y2;
      } else {
        xl1 = x2;
        xl2 = x1;
        yl1 = y2;
        yl2 = y1;
      }

      stepx = step * (xl2 - xl1) / (yl2 - yl1);

      for (xl = xl1, yl = yl1; yl <= yl2; xl += stepx, yl += step) {
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (yl - int(yl)) * step;
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (yl - int(yl)) * step;
      }
    } else {
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (y1 - int(y1));
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (y1 - int(y1));
    }
  }
}

// generates the line by brute force - further optimized
void Line3(cv::Mat &imga, float x1, float y1, float x2, float y2) {
  float xl, yl, xl1, yl1, xl2, yl2, step;
  const float stp = 2;  // number of steps
  float stepx, stepy;

  imga = 0;

  if (abs(x1 - x2) > abs(y1 - y2)) {
    step = 1 / stp;

    if (x1 < x2) {
      xl1 = x1;
      xl2 = x2;
      yl1 = y1;
      yl2 = y2;
    } else {
      xl1 = x2;
      xl2 = x1;
      yl1 = y2;
      yl2 = y1;
    }

    stepy = step * (yl2 - yl1) / (xl2 - xl1);

    for (xl = xl1, yl = yl1; xl <= xl1 + int(xl2 - xl1);
         xl += step, yl += stepy) {
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (yl - int(yl)) * step;
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (int(yl) - yl + 1) * step;
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (yl - int(yl)) * step;
    }

    // last point
    xl = xl2;
    yl = yl2;
    ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
        (int(xl) - xl + 1) * (int(yl) - yl + 1) * step *
        (xl2 - xl1 - int(xl2 - xl1));
    ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
        (int(xl) - xl + 1) * (yl - int(yl)) * step *
        (xl2 - xl1 - int(xl2 - xl1));
    ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
        (xl - int(xl)) * (int(yl) - yl + 1) * step *
        (xl2 - xl1 - int(xl2 - xl1));
    ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
        (xl - int(xl)) * (yl - int(yl)) * step * (xl2 - xl1 - int(xl2 - xl1));
  } else {
    if (!((x1 == x2) && (y1 == y2))) {
      step = 1 / stp;
      if (y1 < y2) {
        xl1 = x1;
        xl2 = x2;
        yl1 = y1;
        yl2 = y2;
      } else {
        xl1 = x2;
        xl2 = x1;
        yl1 = y2;
        yl2 = y1;
      }

      stepx = step * (xl2 - xl1) / (yl2 - yl1);

      for (xl = xl1, yl = yl1; yl <= yl1 + int(yl2 - yl1);
           xl += stepx, yl += step) {
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
            (int(xl) - xl + 1) * (yl - int(yl)) * step;
        ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (int(yl) - yl + 1) * step;
        ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
            (xl - int(xl)) * (yl - int(yl)) * step;
      }

      // last point
      xl = xl2;
      yl = yl2;
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (int(yl) - yl + 1) * step *
          (yl2 - yl1 - int(yl2 - yl1));
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl)] +=
          (int(xl) - xl + 1) * (yl - int(yl)) * step *
          (yl2 - yl1 - int(yl2 - yl1));
      ((float *)(imga.data + int(yl) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (int(yl) - yl + 1) * step *
          (yl2 - yl1 - int(yl2 - yl1));
      ((float *)(imga.data + int(yl + 1) * (imga.step)))[int(xl + 1)] +=
          (xl - int(xl)) * (yl - int(yl)) * step * (yl2 - yl1 - int(yl2 - yl1));
    } else {
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1)] +=
          (int(x1) - x1 + 1) * (y1 - int(y1));
      ((float *)(imga.data + int(y1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (int(y1) - y1 + 1);
      ((float *)(imga.data + int(y1 + 1) * (imga.step)))[int(x1 + 1)] +=
          (x1 - int(x1)) * (y1 - int(y1));
    }
  }
}

void Circle(cv::Mat &imga, float x0, float y0, float r)  // generates the image
{
  int i1, j1, i2, j2;        // temporal coordinates for points
  const float step = 0.005;  // step for the line increment
  float x, y, alpha;
  double mx, mn;

  imga = 0;

  for (alpha = 0; (alpha < 2 * CV_PI); alpha += step) {
    x = r * cos(alpha) + x0;
    y = r * sin(alpha) + y0;
    ((float *)(imga.data + int(y) * (imga.step)))[int(x)] +=
        (int(x) - x + 1) * (int(y) - y + 1) / 2;
    ((float *)(imga.data + int(y + 1) * (imga.step)))[int(x)] +=
        (int(x) - x + 1) * (y - int(y)) / 2;
    ((float *)(imga.data + int(y) * (imga.step)))[int(x + 1)] +=
        (x - int(x)) * (int(y) - y + 1) / 2;
    ((float *)(imga.data + int(y + 1) * (imga.step)))[int(x + 1)] +=
        (x - int(x)) * (y - int(y)) / 2;
    // cout << x << "  " << y << '\n';
  }

  cv::minMaxLoc(imga, &mn, &mx);
  imga = imga / mx;
}

void Circle(cv::Mat &imga, float x0, float y0, float r,
            int fl)  // generates the image
{
  int i1, j1, i2, j2;        // temporal coordinates for points
  const float step = 0.005;  // step for the line increment
  float x, y, alpha, rd;
  double mx, mn;

  imga = 0;

  for (alpha = 0; (alpha < 2 * CV_PI); alpha += step) {
    for (rd = 0; (rd < r); rd += step) {
      x = rd * cos(alpha) + x0;
      y = rd * sin(alpha) + y0;
      ((float *)(imga.data + int(y) * (imga.step)))[int(x)] +=
          (int(x) - x + 1) * (int(y) - y + 1) / 2;
      ((float *)(imga.data + int(y + 1) * (imga.step)))[int(x)] +=
          (int(x) - x + 1) * (y - int(y)) / 2;
      ((float *)(imga.data + int(y) * (imga.step)))[int(x + 1)] +=
          (x - int(x)) * (int(y) - y + 1) / 2;
      ((float *)(imga.data + int(y + 1) * (imga.step)))[int(x + 1)] +=
          (x - int(x)) * (y - int(y)) / 2;
    }
  }

  cv::minMaxLoc(imga, &mn, &mx);
  imga = imga * 15.0 / mx;
}

int main(int argc, char **argv) {
  cv::Mat img1;  // initial images, end image

  cv::namedWindow("Initial", 0);
  img1 = cv::Mat(cv::Size(30, 30), CV_32FC1);
  /*
  Line(img1, 3, 3, 10, 14);
  cv::imshow("Initial", img1);
  cv::waitKey(0);
  */

  for (float r = 0; r < 10; r += 0.001) {
    // Circle(img1, 15, 15, r);
    Line3(img1, r, 1, 20, 10);
    cv::imshow("Initial", img1);
    cv::waitKey(2);
  }

  // img1 = img1 * 255;
  // cvSaveImage("testQVGA.tif", img1);

  return (0);
}