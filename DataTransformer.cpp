#include "DataTransformer.hpp"

void OpenCV2Blob(const std::vector<cv::Mat> channels,
                 std::shared_ptr<caffe::Net<float>> net)
{
    caffe::Blob<float> *input_layer = net->input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();
    
    for (const auto &ch: channels)
    {
        for (auto i = 0; i != ch.rows; i++)
        {
            std::memcpy(input_data, ch.ptr<float>(i), sizeof(float) * ch.cols);
            input_data += ch.cols;
        }
    }
}