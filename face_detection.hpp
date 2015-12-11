#ifndef face_detection_hpp
#define face_detection_hpp

#include <stdio.h>
#include <ctime>
#include <string>
#include <fstream>
#include <vector>
#include "DataTransformer.hpp"
#include "Timer.hpp"

const std::string root("/Users/liuyang/Desktop/Class/ML/ImageProcessing/face_detection_caffe/FaceDetection_CNN-master/");
const double red_channel_mean   = 104.00698793;
const double green_channel_mean = 116.66876762;
const double blue_channel_mean  = 122.67891434;

using boundingbox = std::pair<cv::Rect_<float>, double>;


void face_detection(std::string imgList, std::string resultList);

void read_image_list(std::vector<std::string>& imgFiles, std::string imgList);

void generate_bounding_box(Eigen::MatrixXf prob, double scale, std::vector<boundingbox>& bounding_box);

void nms_max(std::vector<boundingbox>& bd, std::vector<boundingbox>& final_bd, double overlapped_thresh = 0.3);
void nms_average(std::vector<boundingbox>& bd, std::vector<boundingbox>& final_bd, double overlapped_thresh = 0.2);

void draw_boxes(std::vector<boundingbox>& bd, cv::Mat& img);

bool sort_by_size(const boundingbox& a, const boundingbox& b);
bool sort_by_confidence_reverse(const boundingbox& a, const boundingbox& b );



#endif /* face_detection_hpp */