#ifndef face_detection_hpp
#define face_detection_hpp

#include <stdio.h>
#include <ctime>
#include <string>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include "DataTransformer.hpp"

using Eigen::MatrixXf;
using boundingbox = std::pair<cv::Rect_<float>, double>;

constexpr double ILSVRC_RED_MEAN    = 104.00698793;
constexpr double ILSVRC_GREEN_MEAN  = 116.66876762;
constexpr double ILSVRC_BLUE_MEAN   = 122.67891434;



void face_detection(std::string imgList, std::string resultList);
cv::Mat &face_detection(cv::Mat &img);



std::vector<double> scale_list(const cv::Mat &img);
void updatePrototxt(int rows, int cols);
void read_image_list(std::vector<std::string>& imgFiles, std::string imgList);
void generate_bounding_box(Eigen::MatrixXf prob,
                           double scale, std::vector<boundingbox>& bounding_box);

void nms_max(std::vector<boundingbox>& bd,
             std::vector<boundingbox>& final_bd, double overlapped_thresh = 0.3);
void nms_average(std::vector<boundingbox>& bd,
                 std::vector<boundingbox>& final_bd, double overlapped_thresh = 0.2);
void draw_boxes(std::vector<boundingbox>& bd, cv::Mat& img);

bool sort_by_size(const boundingbox& a, const boundingbox& b);
bool sort_by_confidence_reverse(const boundingbox& a, const boundingbox& b );

#endif /* face_detection_hpp */
