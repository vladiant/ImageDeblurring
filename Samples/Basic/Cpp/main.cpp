#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr auto doc = R"doc(
Wiener deconvolution.

Sample shows how DFT can be used to perform Weiner deconvolution [1]
of an image with user-defined point spread function (PSF)

Usage:
  ./BasicDeblur  [--circle]
      [--angle <degrees>]
      [--d <diameter>]
      [--snr <signal/noise ratio in db>]
      [<input image>]

  Use sliders to adjust PSF paramitiers.
  Keys:
    SPACE - switch btw linear/circular PSF
    ESC   - exit

Examples:
  ./BasicDeblur licenseplate_motion.jpg --angle=135 --d=22  
    (image source: http://www.topazlabs.com/infocus/_images/licenseplate_compare.jpg)

  ./BasicDeblur text_motion.jpg --angle=86 --d=31  
  ./BasicDeblur text_defocus.jpg --circle --d=19  
    (image source: compact digital photo camera, no artificial distortion)


[1] http://en.wikipedia.org/wiki/Wiener_deconvolution";
)doc";

constexpr auto winName = "deconvolution";

bool defocus = false;
cv::Mat img;
cv::Mat img_dft;

cv::Mat blur_edge(const cv::Mat& img, int d = 31) {
  const int w = img.cols;
  const int h = img.rows;
  cv::Mat img_pad;
  cv::copyMakeBorder(img, img_pad, d, d, d, d, cv::BORDER_WRAP);
  cv::Mat img_blur_pad;
  cv::GaussianBlur(img_pad, img_blur_pad, cv::Size(2 * d + 1, 2 * d + 1), -1);
  cv::Mat img_blur = img_blur_pad(cv::Rect(d, d, w, h));

  cv::Mat result(h, w, CV_32FC1);
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      const float dist = std::min({x, w - x - 1, y, h - y - 1});
      const float wt = std::min(dist / d, 1.0f);
      result.at<float>(y, x) =
          wt * img.at<float>(y, x) + (1.0f - wt) * img_blur.at<float>(y, x);
    }
  }

  return result;
}

cv::Mat motion_kernel(float angle, int d, int sz = 65) {
  if (0 == d) {
    d = 1;
  }
  cv::Mat kern(1, d, CV_32FC1, cv::Scalar(1.0));
  const float c = cos(angle);
  const float s = sin(angle);
  const int sz2 = sz / 2;
  float A[] = {c, -s, 0, s, c, 0};
  const float mult = (d - 1) * 0.5;
  A[2] = sz2 - mult * A[0];
  A[5] = sz2 - mult * A[3];
  cv::warpAffine(kern, kern, cv::Mat(2, 3, CV_32FC1, A), cv::Size(sz, sz),
                 cv::INTER_CUBIC);
  return kern;
}

cv::Mat defocus_kernel(int d, int sz = 65) {
  cv::Mat kern_int(sz, sz, CV_8U, cv::Scalar(0));
  cv::circle(kern_int, cv::Point(sz, sz), d, 255, -1, cv::LINE_AA, 1);
  cv::Mat kern;
  kern_int.convertTo(kern, CV_32F);
  kern /= 255.0;
  return kern;
}

void update(int, void*) {
  const float ang = cv::getTrackbarPos("angle", winName) * M_PI / 180;
  const float d = cv::getTrackbarPos("d", winName);
  const float noise =
      std::pow(10.0, -0.1 * cv::getTrackbarPos("SNR (db)", winName));

  cv::Mat psf = defocus ? defocus_kernel(d) : motion_kernel(ang, d);

  cv::imshow("psf", psf);

  psf /= cv::sum(psf);

  cv::Mat psf_pad(img.size(), img.type(), cv::Scalar(0));
  const int kh = psf.rows;
  const int kw = psf.cols;
  psf.copyTo(psf_pad(cv::Rect(0, 0, kw, kh)));

  cv::Mat psf_dft;
  cv::dft(psf_pad, psf_dft, cv::DFT_COMPLEX_OUTPUT, kh);

  std::vector<cv::Mat> channels(2);
  cv::Mat psf2 = psf_dft.mul(psf_dft);

  cv::split(psf2, channels);
  channels[0] = channels[0] + channels[1];
  channels[1] = channels[0] + noise;
  channels[0] = channels[0] + noise;
  cv::merge(channels, psf2);

  cv::Mat ipsf = psf_dft / psf2;

  cv::Mat res_dft;
  cv::mulSpectrums(img_dft, ipsf, res_dft, 0);
  cv::Mat res;
  cv::idft(res_dft, res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  cv::imshow(winName, res);
};

int main(int argc, char** argv) {
  cv::String keys =
      "{help h usage ? |      | print this message   }"
      "{circle         |      | set defocus kernel type  }"
      "{angle          | 135 | angle of linear kernel   }"
      "{d              | 22  | line length or circle radius in case of defocus "
      "kernel  }"
      "{snr            | 25  | signal to noise ratio in dB   }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(doc);

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  std::string image_name = "text_motion.jpg";
  if (argc > 1) {
    image_name = argv[1];
  }

  cv::Mat source = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
  if (source.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  source.convertTo(img, CV_32F);
  img /= 255.0;
  cv::imshow("input", img);

  img = blur_edge(img);

  cv::dft(img, img_dft, cv::DFT_COMPLEX_OUTPUT);

  defocus = parser.has("circle");

  cv::namedWindow(winName);
  cv::namedWindow("psf", cv::WINDOW_NORMAL);

  int angle = parser.get<int>("angle");
  int d = parser.get<int>("d");
  int snr = parser.get<int>("snr");
  cv::createTrackbar("angle", winName, &angle, 180, update);
  cv::createTrackbar("d", winName, &d, 50, update);
  cv::createTrackbar("SNR (db)", winName, &snr, 50, update);

  update(0, nullptr);

  while (true) {
    char ch = cv::waitKey(0);
    if (ch == 27) {
      break;
    }
    if (ch == ' ') {
      defocus = !defocus;
      update(0, nullptr);
    }
  }

  return EXIT_SUCCESS;
}
