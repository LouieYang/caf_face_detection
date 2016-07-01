/*******************************************************************
 *  Copyright(c) 2015
 *  All rights reserved.
 *
 *  Name: face detection
 *  Description: multiview face detection 
 *  Reference: Multi-view Face Detection using deep convolutional neural networks
 *  Lib: OpenCV2, Eigen3, caffe
 *  Date: 2015-12-06
 *  Author: Yang
 ******************************************************************/
#include "face_detection.hpp"

void read_image_list(std::vector<std::string>& imgFiles, std::string imgList)
{
    /*
     *  Description: Read imgfile in imgList to a vector
     */
    
    imgFiles.clear();
    std::ifstream fin(imgList, std::ios::in);
    
    for (std::string line; std::getline(fin, line);)
    {
        imgFiles.push_back(line);
    }
    fin.close();
}

void generate_bounding_box(Eigen::MatrixXf prob, double scale, std::vector<boundingbox>& bd)
{
    const int stride    = 32;
    const int cell_size = 227;
    
    for (int h = 0; h < prob.rows(); h++)
    {
        for (int w = 0; w < prob.cols(); w++)
        {
            /* threshold is set 0.85 */
            if (prob(h, w) >= 0.85)
            {
                bd.push_back(boundingbox(cv::Rect_<float>(float(w * stride) / scale, float(h * stride) / scale, (float)cell_size / scale, (float)cell_size / scale), prob(h, w)));
            }
        }
    }
}

void nms_max(std::vector<boundingbox>& bd, std::vector<boundingbox>& final_bd, double overlapped_thresh)
{
    /*
     *  Descripttion: Non-maximum suppression algorithm with maximizing
     */
    
    std::sort(bd.begin(), bd.end(), sort_by_size);
    for (int i = 0; i < bd.size(); i++)
    {
        int j = 0;
        for (; j < final_bd.size(); j++)
        {
            /* Calculate the overlapped area */
            float x11 = bd[i].first.x;
            float y11 = bd[i].first.y;
            float x12 = bd[i].first.x + bd[i].first.height;
            float y12 = bd[i].first.y + bd[i].first.width;
            
            float x21 = final_bd[j].first.x;
            float y21 = final_bd[j].first.y;
            float x22 = final_bd[j].first.x + final_bd[j].first.height;
            float y22 = final_bd[j].first.y + final_bd[j].first.width;
            
            float x_overlap = MAX(0, MIN(x12, x22) - MAX(x11, x21));
            float y_overlap = MAX(0, MIN(y12, y22) - MAX(y11, y21));
            
            if (x_overlap * y_overlap > MIN(bd[i].first.area(), final_bd[j].first.area()) * overlapped_thresh)
            {
                if (final_bd[j].second < bd[i].second)
                {
                    final_bd[j] = bd[i];
                }
                break;
            }
        }
        if (j == final_bd.size())
        {
            final_bd.push_back(bd[i]);
        }
    }
}

void nms_average(std::vector<boundingbox>& bd, std::vector<boundingbox>& final_bd, double overlapped_thresh)
{
    /*
     *  Descripttion: Non-maximum suppression algorithm with averaging
     */
    
    std::sort(bd.begin(), bd.end(), sort_by_confidence_reverse);
    while (bd.size() != 0)
    {
        std::vector<int> iddlt(1, 0);
        
        float x11 = bd[0].first.x;
        float y11 = bd[0].first.y;
        float x12 = bd[0].first.x + bd[0].first.height;
        float y12 = bd[0].first.y + bd[0].first.width;
        
        if (bd.size() > 1)
        {
            for (int j = 1; j < bd.size(); j++)
            {
                float x21 = bd[j].first.x;
                float y21 = bd[j].first.y;
                float x22 = bd[j].first.x + bd[j].first.height;
                float y22 = bd[j].first.y + bd[j].first.width;
                
                float x_overlap = MAX(0, MIN(x12, x22) - MAX(x11, x21));
                float y_overlap = MAX(0, MIN(y12, y22) - MAX(y11, y21));
                
                if (x_overlap * y_overlap > MIN(bd[0].first.area(), bd[j].first.area()) * overlapped_thresh)
                {
                    iddlt.push_back(j);
                }
            }
        }
        
        float x_average  = 0;
        float y_average  = 0;
        float width      = 0;
        float height     = 0;
        float confidence = 0;
        
        for (int i = 0; i < iddlt.size(); i++)
        {
            x_average  += bd[iddlt[i]].first.x;
            y_average  += bd[iddlt[i]].first.y;
            width      += bd[iddlt[i]].first.width;
            height     += bd[iddlt[i]].first.height;
            confidence += bd[iddlt[i]].second;
        }
        x_average /= iddlt.size();
        y_average /= iddlt.size();
        width /= iddlt.size();
        height /= iddlt.size();
        confidence /= iddlt.size();
        
        final_bd.push_back(boundingbox(cv::Rect_<float>(y_average, x_average, width, height), confidence));
        
        
        for (int i = 0; i < iddlt.size(); i++)
        {
            bd.erase(bd.begin() + iddlt[i] - i);
        }
    }
}

bool sort_by_confidence_reverse(const boundingbox& a, const boundingbox& b)
{
    return a.second > b.second;
}

bool sort_by_size(const boundingbox& a, const boundingbox& b)
{
    return a.first.width < b.first.width;
}

void draw_boxes(std::vector<boundingbox>& bd, cv::Mat& img)
{
    for (int k = 0; k < bd.size(); k++)
    {
        cv::rectangle(img, cv::Rect(int(bd[k].first.y), int(bd[k].first.x), int(bd[k].first.width), int(bd[k].first.height)), cv::Scalar(0, 0, 255), 2);
        std::stringstream ss;
        ss << bd[k].second;
        cv::putText(img, ss.str(), cv::Point(bd[k].first.y, bd[k].first.x), 1, 1, cv::Scalar(255, 0, 0));
    }
}

std::vector<double> scale_list(const cv::Mat &img)
{
    int min             = 0;
    int max             = 0;
    double delim        = 5;
    double factor       = 0.7937;
    double factor_count = 0;
    
    std::vector<double> scales;
    
    max = MAX(img.cols, img.rows);
    min = MIN(img.cols, img.rows);
    
    //        delim = 2500 / max;
    while (delim > 1 + 1e-4)
    {
        scales.push_back(delim);
        delim *= factor;
    }
    
    while (min >= 227)
    {
        scales.push_back(pow(factor, factor_count++));
        min *= factor;
    }
    
    std::cout << "Image size: " << img.cols << "(Width)" << ' ' << img.rows <<  "(Height)" <<'\n';
    std::cout << "Scaling: ";
    std::for_each(scales.begin(), scales.end(), [](double scale){ std::cout << scale << ' '; });
    std::cout << '\n';
    return scales;
}

void updatePrototxt(int rows, int cols)
{
    std::ifstream fin("face_full_conv.prototxt", std::ios::in);
    std::ofstream fout("face_full_conv2.prototxt", std::ios::out);
    int index = 0;
    for (std::string line; std::getline(fin, line); index++)
    {
        if (index == 5)
        {
            fout << "input_dim: " << rows << '\n';
        }
        else if (index == 6)
        {
            fout << "input_dim: " << cols << '\n';
        }
        else
        {
            fout << line << '\n';
        }
    }
    fin.close();
    fout.close();
}

void face_detection(std::string imgList, std::string resultList)
{
    std::vector<std::string> imgFiles;
    read_image_list(imgFiles, imgList);
    
    std::fstream output_file(resultList, std::ios::app|std::ios::out);
    output_file << "#faceID" << '\t' << "imagePath" << '\t';
    output_file << "faceRect.y" << '\t' << "faceRect.x" << '\t';
    output_file << "faceRect.w" << '\t' << "faceRect.h" << '\n';
    
    for (int i = 0; i < imgFiles.size(); i++)
    {
        cv::Mat img = cv::imread(imgFiles[i]);
        std::vector<double> scales(scale_list(img));
        std::vector<boundingbox> bd;
        for (int j = 0; j < scales.size(); j++)
        {
            cv::Mat scale_img;
            cv::resize(img, scale_img, cv::Size(img.cols * scales[j], img.rows * scales[j]));
            updatePrototxt(scale_img.rows, scale_img.cols);
            
            std::vector<cv::Mat> channels;
            scale_img.convertTo(scale_img, CV_32FC3);

            cv::split(scale_img, channels);
            channels[0] -= ILSVRC_BLUE_MEAN;
            channels[1] -= ILSVRC_GREEN_MEAN;
            channels[2] -= ILSVRC_RED_MEAN;
            
            std::unique_ptr<caffe::Net<float>>
            net(new caffe::Net<float>("face_full_conv2.prototxt",
                                      caffe::Phase::TEST));
            net->CopyTrainedLayersFrom("face_full_conv.caffemodel");
            
            OpenCV2Blob(channels, net);
            
            net->ForwardPrefilled();
            caffe::Blob<float>* output_layer = net->output_blobs()[0];
            
            float* data = const_cast<float*>(output_layer->cpu_data() + output_layer->shape(2) * output_layer->shape(3));
            
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> prob(data, output_layer->shape(2), output_layer->shape(3));
            
            generate_bounding_box(prob, scales[j], bd);
        }
        
        std::vector<boundingbox> bd1;
        std::vector<boundingbox> bdf;
        nms_max(bd, bd1);
        nms_average(bd1, bdf);
        
        for (int k = 0; k < bdf.size(); k++)
        {
            output_file << i << '\t' << imgFiles[i] << '\t';
            output_file << int(bdf[k].first.y) << '\t'
            << int(bdf[k].first.x) << '\t';
            output_file << int(bdf[k].first.width) << '\t'
            << int(bdf[k].first.height) << '\n';
        }
    }
}

cv::Mat &face_detection(cv::Mat &img)
{
    std::vector<double> scales(scale_list(img));
    std::vector<boundingbox> bd;
    for (int j = 0; j < scales.size(); j++)
    {
        cv::Mat scale_img;
        cv::resize(img, scale_img, cv::Size(img.cols * scales[j], img.rows * scales[j]));
        updatePrototxt(scale_img.rows, scale_img.cols);
        
        std::vector<cv::Mat> channels;
        scale_img.convertTo(scale_img, CV_32FC3);
        
        cv::split(scale_img, channels);
        channels[0] -= ILSVRC_BLUE_MEAN;
        channels[1] -= ILSVRC_GREEN_MEAN;
        channels[2] -= ILSVRC_RED_MEAN;
        
        std::unique_ptr<caffe::Net<float>>
        net(new caffe::Net<float>("face_full_conv2.prototxt",
                                  caffe::Phase::TEST));
        net->CopyTrainedLayersFrom("face_full_conv.caffemodel");
        
        OpenCV2Blob<float>(channels, net);
        
        net->ForwardPrefilled();
        caffe::Blob<float>* output_layer = net->output_blobs()[0];
        
        float* data = const_cast<float*>(output_layer->cpu_data() + output_layer->shape(2) * output_layer->shape(3));
        
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> prob(data, output_layer->shape(2), output_layer->shape(3));
        
        generate_bounding_box(prob, scales[j], bd);
    }
    
    std::vector<boundingbox> bd1;
    std::vector<boundingbox> bdf;
    nms_max(bd, bd1);
    nms_average(bd1, bdf);

    draw_boxes(bdf, img);
    return img;
}