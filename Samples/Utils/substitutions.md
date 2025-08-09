


| C                     | C++                                     |
| --------------------- | --------------------------------------- |
IplImage                |  cv::Mat                                |
CvMat                   |  cv::Mat                                |
cvCreateMat             |  cv::Mat                                |
CvSparseMat             |  cv::SparseMat                          |
CvSparseMatIterator     |  cv::SparseMatIterator                  |
CvSparseNode            |  cv::SparseMat::Node                    |
CvHistogram             |  cv::Mat                                |
->height                |  .rows                                  |
->widthStep             |  .step                                  |
->width                 |  .cols                                  |
->imageData             |  .data                                  |
->depth                 |  .depth()                               |
->nChannels             |  .channels()                            |
cvCloneImage            |  .clone()                               |
cvDFT                   |  cv::dft                                |
CV_DXT_FORWARD          |  default , cv::dft(imga, imgb);         |
CV_DXT_INV_SCALE        |  cv::DFT_INVERSE | cv::DFT_SCALE        |
cvMulSpectrums          |  cv::mulSpectrums (imga, imgb, imgc, 0) |
cvNamedWindow           |  cv::namedWindow                        |
cvResizeWindow          |  cv::resizeWindow                       |
cvShowImage             |  cv::imshow                             |
cvWaitKey               |  cv::waitKey                            |
CV_WINDOW_AUTOSIZE      |  cv::WINDOW_AUTOSIZE                    |
CV_WINDOW_NORMAL        |  cv::WINDOW_NORMAL                      |
cvDestroyWindow         |  cv::destroyWindow                      |
cvLoadImage             |  cv::imread                             |
CV_LOAD_IMAGE_GRAYSCALE |  cv::IMREAD_GRAYSCALE                   |
CV_LOAD_IMAGE_ANYCOLOR  |  cv::IMREAD_ANYCOLOR                    |
CV_LOAD_IMAGE_ANYDEPTH  |  cv::IMREAD_ANYDEPTH                    |
cvSaveImage             |  cv::imwrite                            |
IPL_DEPTH_32F, 1        |  CV_32FC1                               |
IPL_DEPTH_32F, 2        |  CV_32FC2                               |
IPL_DEPTH_32F, 3        |  CV_32FC3                               |
IPL_DEPTH_8U, 1         |  CV_8UC1                                |
IPL_DEPTH_8U, 3         |  CV_8UC3                                |
cvSize                  |  cv::Size                               |
cvGetSize               |  cv::Size(cols, rows)                   |
cvRect                  |  cv::Rect                               |
cvRectangle             |  cv::rectangle                          |
cvFlip                  |  cv::flip                               |
cvNorm                  |  cv::norm                               |
cvCvtColor              |  cv::cvtColor                           |
CV_BGR2GRAY             |  cv::COLOR_BGR2GRAY                     |
CV_BGR2HLS              |  cv::COLOR_BGR2HLS                      |
CV_HLS2BGR              |  cv::COLOR_HLS2BGR                      |
CV_L2                   |  cv::NORM_L2                            |
cvDiv                   |  cv::divide                             |
cvMul                   |  cv::multiply                           |
cvPoint                 |  cv::Point                              |
cvScalar                |  cv::Scalar                             |
cvCircle                |  cv::circle                             |
cvLine                  |  cv::line                               |
cvFillConvexPoly        |  cv::fillConvexPoly                     |
cvCreateImage           |  cv::Mat                                |
cvSplit                 |  cv::split  std::vector                 |
cvMerge                 |  cv::merge                              |
cvCopyMakeBorder        |  cv::copyMakeBorder                     |
IPL_BORDER_REPLICATE    |  cv::BORDER_REPLICATE                   |
cvMinMaxLoc             |  cv::minMaxLoc                          |
cvSetZero               |  = 0                                    |
cvScale(a, b, c)        |  b = a * c                              |
cvAddWeighted           |  cv::addWeighted                        |
cvSub                   |  cv::subtract                           |
cvDotProduct(a, b)      |  a.dot(b)                               |
cvCopy(&a, b)           |  a.copyTo(b)                            |
cvLaplace               |  cv::Laplacian -1)                      |
cvSobel                 |  cv::Sobel -1)                          |
cvSmooth, CV_BLUR       |  cv::blur                               |
IPL_DEPTH_8U            |  CV_8U                                  |
IPL_DEPTH_16U           |  CV_16U                                 |
IPL_DEPTH_32S           |  CV_32S                                 |
cvCalcHist              |  cv::calcHist                           |
cvGetMinMaxHistValue    |  cv::minMaxLoc                          |
cvCvtColor              |  cv::cvtColor                           |
CV_GRAY2BGR             |  cv::COLOR_GRAY2BGR                     |
CV_RGBA2GRAY            |  cv::COLOR_RGBA2GRAY                    |
CV_BGR2HLS              |  cv::COLOR_BGR2HLS                      |
CV_BGR2HSV              |  cv::COLOR_BGR2HSV                      |
CV_HLS2BGR              |  cv::COLOR_HLS2BGR                      |
CV_HSV2BGR              |  cv::COLOR_HSV2BGR                      |
CV_Lab2BGR              |  cv::COLOR_Lab2BGR                      |
CV_BGR2Lab              |  cv::COLOR_BGR2Lab                      |
CV_Luv2BGR              |  cv::COLOR_Luv2BGR                      |
CV_BGR2Luv              |  cv::COLOR_BGR2Luv                      |
CV_BGR2YCrCb            |  cv::COLOR_BGR2YCrCb                    |
CV_YCrCb2BGR            |  cv::COLOR_YCrCb2BGR                    |
cvGetOptimalDFTSize     |  cv::getOptimalDFTSize	              |
cvSetMouseCallback      |  cv::setMouseCallback                   |
CV_EVENT_LBUTTONDOWN    |  cv::EVENT_LBUTTONDOWN                  |
cvPutText               |  cv::putText                            |
CV_FONT_HERSHEY_TRIPLEX |  cv::FONT_HERSHEY_TRIPLEX               |

[C structures and operations](https://docs.opencv.org/3.4/d2/df8/group__core__c.html)