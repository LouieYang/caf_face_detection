#ifndef DataTransformer_hpp
#define DataTransformer_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <caffe/caffe.hpp>
#include <eigen3/Eigen/Eigen>
#include <string>
#include <iostream>

#endif /* DataTransformer_hpp */

using Eigen::MatrixXf;

caffe::Blob<float>* OpenCV2Blob(const std::vector<cv::Mat> imgs);
caffe::Blob<float>* Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs);
MatrixXf OpenCV2Eigen(cv::Mat image);

void Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs, boost::shared_ptr<caffe::Net<float>> net);
