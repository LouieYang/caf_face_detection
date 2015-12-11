#ifndef Detector_hpp
#define Detector_hpp

#include <stdio.h>
#include "DataTransformer.hpp"
#include <tuple>

#define CPU_ONLY

#endif /* Detector_hpp */

#define WEIGHTED_AVERAGE(a, b, times) (a + b * times) / (1 + times)

typedef std::pair<cv::Rect, float> prob;
typedef std::pair<cv::Point, float> attribute;
typedef std::pair<std::vector<cv::Mat>, std::vector<attribute>> Fragments;
typedef std::vector<prob> Prediction;

const int box_size    = 227;
const int magnifying  = 5;
const int stride      = 32;
const int batch_size  = 64;
const int min_size    = 40;
const float rescaling = 1.705;
const float threshold = 0.7;
const float NMS       = 0.3;


class Detector
{
public:
    Detector(const std::string& model_file, const std::string& trained_file, const std::string& mean_file);
    
    Prediction Detect(const cv::Mat& img);
    
    void ShowResultImage(cv::Mat img);
    
private:
    void Predict(const Fragments& fra, Prediction& pre);
    
    void Segment(const cv::Mat& img, Fragments& fragments);
    
    void SetMean(const std::string& mean_file);
    
    void Deoverlapped(Prediction& pre, Prediction& final);
    
private:
    boost::shared_ptr<caffe::Net<float>> net_;
    cv::Size input_geometry_;
    cv::Mat mean_;
    int num_channels_;
    std::vector<std::string> labels_;
};

static std::vector<int> Argmax(const std::vector<float>& v, int N);

bool compare_with_rect(const prob& a, const prob& b);

template <class T> void ClearVector(std::vector<T>& vt);
