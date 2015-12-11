#include "DataTransformer.hpp"

MatrixXf OpenCV2Eigen(cv::Mat image)
{
    /*
     *  Description: map the opencv image to eigen matrix(if RGB, gray it)
     */
    
    cv::Mat image_gray;
    if (image.channels() != 1)
    {
        cv::cvtColor(image, image_gray, CV_RGB2GRAY);
    }
    else
    {
        image_gray = image;
    }
    
    image_gray.convertTo(image_gray, CV_32FC1);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image_matrix(image_gray.ptr<float>(), image_gray.rows, image_gray.cols);
    
    return image_matrix;
}

void Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs, boost::shared_ptr<caffe::Net<float>> net)
{
    
    caffe::Blob<float>* input_layer = net->input_blobs()[0];
    float* input_data = input_layer->mutable_cpu_data();
    
    int img_number  = imgs.size();
    int img_channel = imgs[0].size();
    int img_height  = imgs[0][0].rows();
    int img_width   = imgs[0][0].cols();
    int index = 0;
    for (int i = 0; i < img_number; i++)
    {
        for (int c = 0; c < img_channel; c++)
        {
            for (int h = 0; h < img_height; h++)
            {
                for (int w = 0; w < img_width; w++)
                {
                    *(input_data + index) = imgs[i][c](h, w);
                    index++;
                }
            }
        }
    }    
}


caffe::Blob<float>* Eigen2Blob(const std::vector<std::vector<Eigen::MatrixXf>> imgs)
{
    int num_channels = static_cast<int>(imgs[0].size());
    
    
    caffe::Blob<float>* blob = new caffe::Blob<float>((int)imgs.size(), num_channels, imgs[0][0].rows(), imgs[0][0].cols());
    
    //get the blobproto
    caffe::BlobProto blob_proto;
    blob_proto.set_num(int(imgs.size()));
    blob_proto.set_channels(num_channels);
    blob_proto.set_height(imgs[0][0].rows());
    blob_proto.set_width(imgs[0][0].cols());
    int size_of_imgs = num_channels * imgs[0][0].cols() * imgs[0][0].rows();
    for (int i = 0; i < size_of_imgs; ++i)
    {
        blob_proto.add_data(0.);
    }

    int img_number  = imgs.size();
    int img_channel = imgs[0].size();
    int img_height  = imgs[0][0].rows();
    int img_width   = imgs[0][0].cols();
    int index       = 0;
    if (size_of_imgs) {
        for (int i = 0; i < img_number; i++)
        {
            for (int c = 0; c < img_channel; c++)
            {
                for (int h = 0; h < img_height; h++)
                {
                    for (int w = 0; w < img_width; w++)
                    {
                        blob_proto.set_data(index++, imgs[i][c](h, w));
                    }
                }
            }
        }
    }
    
    //set data into blob
    blob->FromProto(blob_proto);
    
    return blob;
}