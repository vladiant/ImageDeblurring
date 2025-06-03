#!/bin/bash

# sed -i 's/IplImage */cv::Mat& /g' $1

# sed -i 's/cvReleaseImage([*]);//g' $1
# cv::Mat& imgb = cvCloneImage(imga); 
# cvSetZero(imgb);  img= 0;  
# cvCopy(&a, b)             a.copyTo(b)
# cvScale(imgi, img, 1.0 / 255.0);
# cvGetSize(imgi) cv::Size(imgi.cols, imgi.rows)
# cv::norm(imgb, img, cv::NORM_L2)
# cvCloneImage(img1)  img1.clone()
# IplConvKernel* cv::Mat

sed -i 's/CvMat/cv::Mat/g' $1

sed -i 's/cvCreateImage/cv::Mat/g' $1

sed -i 's/->height/.rows/g' $1
sed -i 's/->widthStep/.step/g' $1
sed -i 's/->width/.cols/g' $1
sed -i 's/->imageData/.data/g' $1
sed -i 's/->depth/.depth()/g' $1
sed -i 's/->nChannels/.channels()/g' $1

sed -i 's/CV_WINDOW_AUTOSIZE/cv::WINDOW_AUTOSIZE/g' $1
sed -i 's/CV_WINDOW_NORMAL/cv::WINDOW_NORMAL/g' $1
sed -i 's/cvNamedWindow/cv::namedWindow/g' $1
sed -i 's/cvResizeWindow/cv::resizeWindow/g' $1
sed -i 's/cvShowImage/cv::imshow/g' $1
sed -i 's/cvWaitKey/cv::waitKey/g' $1
sed -i 's/cvDestroyWindow/cv::destroyWindow/g' $1

sed -i 's/cvLoadImage/cv::imread/g' $1
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::IMREAD_GRAYSCALE/g' $1
sed -i 's/CV_LOAD_IMAGE_ANYCOLOR/cv::IMREAD_ANYCOLOR/g' $1
sed -i 's/CV_LOAD_IMAGE_ANYDEPTH/cv::IMREAD_ANYDEPTH/g' $1

sed -i 's/cvSaveImage/cv::imwrite/g' $1

sed -i 's/IPL_DEPTH_32F, 1/CV_32FC1/g' $1
sed -i 's/IPL_DEPTH_32F, 2/CV_32FC2/g' $1
sed -i 's/IPL_DEPTH_32F, 3/CV_32FC3/g' $1
sed -i 's/IPL_DEPTH_8U, 1/CV_8UC1/g' $1
sed -i 's/IPL_DEPTH_8U, 3/CV_8UC3/g' $1

sed -i 's/cvSize/cv::Size/g' $1
sed -i 's/CvPoint/cv::Point/g' $1
sed -i 's/cvRect/cv::Rect/g' $1
sed -i 's/cvRectangle/cv::rectangle/g' $1
sed -i 's/cvScalar/cv::Scalar/g' $1

sed -i 's/cvSub/cv::subtract/g' $1
sed -i 's/cvCanny/cv::Canny/g' $1
sed -i 's/cvAddWeighted/cv::addWeighted/g' $1
sed -i 's/cvDiv/cv::divide/g' $1
sed -i 's/cvMul/cv::multiply/g' $1

# cvFilter2D(imgb, imgb, kern) -> cv::filter2D(imgb, imgb, -1, kern);
sed -i 's/cvFilter2D/cv::filter2D/g' $1

sed -i 's/CV_SHAPE_RECT/cv::MORPH_RECT/g' $1
sed -i 's/cvCreateStructuringElementEx/getStructuringElement/g' $1
sed -i 's/cvDilate/cv::dilate/g' $1

sed -i 's/cvNorm/cv::norm/g' $1
sed -i 's/CV_L2/cv::NORM_L2/g' $1

sed -i 's/cvCopyMakeBorder/cv::copyMakeBorder/g' $1
sed -i 's/IPL_BORDER_REPLICATE/cv::BORDER_REPLICATE/g' $1

sed -i 's/CV_DIST_L1/cv::DIST_L1/g' $1
sed -i 's/cvDistTransform/cv::distanceTransform/g' $1

sed -i 's/cvThreshold/cv::threshold/g' $1
sed -i 's/CV_THRESH_BINARY/cv::THRESH_BINARY/g' $1

sed -i 's/cvCreateTrackbar/cv::createTrackbar/g' $1
sed -i 's/cvGetTrackbarPos/cv::getTrackbarPos/g' $1

sed -i 's/cvCvtColor/cv::cvtColor/g' $1
sed -i 's/CV_GRAY2BGR/cv::COLOR_GRAY2BGR/g' $1