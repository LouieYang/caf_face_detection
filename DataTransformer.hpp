#ifndef DataTransformer_hpp
#define DataTransformer_hpp

#include <opencv2/opencv.hpp>
#include <vector>
#include <caffe/caffe.hpp>
#include <cstring>

#endif /* DataTransformer_hpp */

void OpenCV2Blob(const std::vector<cv::Mat> channels,
                 std::shared_ptr<caffe::Net<float>> net);
