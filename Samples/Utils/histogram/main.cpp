#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum class Channel { Blue, Green, Red, Gray };

void show_histogram(const std::string& window_title, cv::Mat& src,
                    Channel channel_id) {
  cv::Mat img;
  constexpr int bins = 256;
  int hist[bins];
  double scale;
  int i, j, channel, max = 0;

  cv::Scalar colors[] = {CV_RGB(0, 0, 255), CV_RGB(0, 255, 0),
                         CV_RGB(255, 0, 0), CV_RGB(255, 255, 255)};

  switch (channel_id) {
    case Channel::Blue:
      channel = 0;
      break;
    case Channel::Green:
      channel = 1;
      break;
    case Channel::Red:
      channel = 2;
      break;
    case Channel::Gray:
      channel = 3;
      break;
    default:
      channel = 0;
  }

  if (src.channels() == 3 && channel == 3) {
    img = cv::Mat(cv::Size(src.cols, src.rows), 8, 1);
    cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);
  } else if (channel > src.channels())
    return;
  else
    img = src.clone();

  cv::Mat canvas(cv::Size(256, 125), CV_8UC3);
  canvas = 0;

  // Reset histogram
  for (j = 0; j < bins - 1; hist[j] = 0, j++);

  // Calc histogram of the image
  for (i = 0; i < img.rows; i++) {
    uchar* ptr = (uchar*)(img.data + i * img.step);
    for (j = 0; j < img.cols; j += img.channels())
      hist[ptr[j + (channel == 3 ? 0 : channel)]]++;
  }

  // Get histogram peak
  for (i = 0; i < bins - 1; i++) max = hist[i] > max ? hist[i] : max;

  // Get scale so the histogram fit the canvas height
  scale = max > canvas.rows ? (double)canvas.rows / max : 1.;

  // Draw histogram
  for (i = 0; i < bins - 1; i++) {
    cv::Point pt1(i, canvas.rows - (hist[i] * scale));
    cv::Point pt2(i, canvas.rows);
    cv::line(canvas, pt1, pt2, colors[channel], 1, 8, 0);
  }

  cv::imshow(window_title, canvas);
  cv::waitKey(0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: %s image_filename\n", argv[0]);
    return EXIT_FAILURE;
  }

  cv::Mat img = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);

  if (img.empty()) {
    printf("Error reading %s !\n", argv[1]);
    return EXIT_FAILURE;
  }

  show_histogram("red channel", img, Channel::Red);
  show_histogram("green channel", img, Channel::Green);
  show_histogram("blue channel", img, Channel::Blue);
  show_histogram("gray hist", img, Channel::Gray);

  printf("Done.\n");
  return EXIT_SUCCESS;
}
