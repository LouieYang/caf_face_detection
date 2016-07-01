#ifndef data_transformer_hpp
#define data_transformer_hpp

#ifndef CPU_ONLY
#define CPU_ONLY
#endif

#include "opencv2/opencv.hpp"

#include <caffe/caffe.hpp>

#include <vector>
#include <iostream>
#include <memory>
#include <string>

/**
 *  @Brief: The data transform from OpenCV to caffe Blob
 *
 *  @param image: OpenCV Mat data vector
 *  @Warning: Template function must be defined in the .hpp file to avoid
 *            linking error
 */
template <typename DType>
void OpenCV2Blob(const std::vector<cv::Mat> &channels,
                 std::unique_ptr<caffe::Net<DType>> &net)
{
    caffe::Blob<DType> *input_layer = net->input_blobs()[0];
    DType *input_data = input_layer->mutable_cpu_data();
    
    for (const auto &ch: channels)
    {
        for (auto i = 0; i != ch.rows; ++i)
        {
            std::memcpy(input_data, ch.ptr<DType>(i), sizeof(DType) * ch.cols);
            input_data += ch.cols;
        }
    }
}

#endif /* data_transformer_hpp */
