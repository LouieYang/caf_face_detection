#include "Detector.hpp"

Detector::Detector(const std::string& model_file, const std::string& trained_file, const std::string& mean_file)
{
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe:set_mode(caffe::Caffe::GPU);
#endif
    net_ = boost::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(model_file, caffe::TEST));

//    net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network shold have exactly two outputs.";
    
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Detector::SetMean(const std::string& mean_file) {
    caffe::BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
    
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void Detector::Segment(const cv::Mat& img, Fragments& fragments)
{
    cv::Mat img_cpy;
    cv::resize(img, img_cpy, cv::Size(img.cols * magnifying, img.rows * magnifying));
    
    while (img_cpy.cols > box_size && img_cpy.rows > box_size)
    {
        for (int c = 0; c < img_cpy.cols - box_size; c = c + stride)
        {
            for (int r = 0; r < img_cpy.rows - box_size; r = r + stride)
            {
                fragments.first.push_back(img_cpy(cv::Rect(c, r, box_size, box_size)));
                fragments.second.push_back(attribute(cv::Point(c, r), (float)img_cpy.cols / img.cols));
            }
        }
        cv::resize(img_cpy, img_cpy,
                   cv::Size(img_cpy.cols / rescaling, img_cpy.rows / rescaling));
    }
}

void Detector::ShowResultImage(cv::Mat img)
{
    Prediction predict = Detect(img);
    
    for (int i = 0; i < predict.size(); i++)
    {
        cv::rectangle(img, predict[i].first, cv::Scalar(0, 0, 255));
        std::stringstream ss;
        ss << predict[i].second;
        cv::putText(img, ss.str(), cv::Point(predict[i].first.y, predict[i].first.x), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
        ss.str("");
    }
    cv::imshow("Figure", img);
    cv::waitKey();
}

void Detector::Predict(const Fragments &fra, Prediction &pre)
{
    int batches = fra.first.size() / batch_size;
    int rest = fra.first.size() % batch_size;
    
    for (int i = 0; i < batches; i++)
    {
        std::vector<cv::Mat> imgs;
        std::copy(fra.first.begin() + i * batch_size, fra.first.begin() + i * batch_size + batch_size, std::back_inserter(imgs));

        caffe::Blob<float>* blob = OpenCV2Blob(imgs);
        
        std::vector<caffe::Blob<float>*> bottom;
        bottom.push_back(blob);
        float type = 0.0;
        const std::vector<caffe::Blob<float>*>& result = net_->Forward(bottom, &type);
        
        for (int j = 0; j < imgs.size(); j++)
        {
            float prob_ = *(result[0]->cpu_data() + 2 * j);
            if (prob_ > threshold)
            {
                int index = i * batch_size + j;
                float scaling = fra.second[index].second;
                cv::Point point_(fra.second[index].first.y / scaling, fra.second[index].first.x / scaling);
                cv::Rect roi(point_, cv::Size(box_size / scaling, box_size / scaling));
                pre.push_back(prob(roi, prob_));
            }
        }
    }
    
    std::vector<cv::Mat> imgs;
    std::copy(fra.first.begin() + batches * batch_size, fra.first.begin() + batches * batch_size + rest, std::back_inserter(imgs));

    for (int i = 0; i < batch_size - rest; i++)
    {
        imgs.push_back(cv::Mat::zeros(box_size, box_size, CV_8UC(num_channels_)));
    }

    caffe::Blob<float>* blob = OpenCV2Blob(imgs);
    std::vector<caffe::Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;
    const std::vector<caffe::Blob<float>*>& result = net_->Forward(bottom, &type);
        
    for (int j = 0; j < rest; j++)
    {
        float prob_ = *(result[0]->cpu_data() + j * 2);
        if (prob_ > threshold)
        {
            int index = batches * batch_size + j;
            float scaling = fra.second[index].second;
            cv::Point point_(fra.second[index].first.y / scaling, fra.second[index].first.x / scaling);
            cv::Rect roi(point_, cv::Size(box_size / scaling, box_size / scaling));
            pre.push_back(prob(roi, prob_));
        }
    }
}

void Detector::Deoverlapped(Prediction &pre, Prediction &final)
{
    final.clear();

    std::sort(pre.begin(), pre.end(), compare_with_rect);
    
    std::vector<int> times;
    for (auto i = pre.begin(); i != pre.end(); i++)
    {
        int exist_number = 0;
        int index        = 0;
        
        for (int j = 0; j < final.size(); j++)
        {
            int x11 = i->first.x;
            int y11 = i->first.y;
            int x12 = i->first.x + i->first.height;
            int y12 = i->first.y + i->first.width;

            int x21 = final[j].first.x;
            int y21 = final[j].first.y;
            int x22 = final[j].first.x + final[j].first.height;
            int y22 = final[j].first.y + final[j].first.width;
            
            int x_overlap = MAX(0, MIN(x12, x22) - MAX(x11, x21));
            int y_overlap = MAX(0, MIN(y12, y22) - MAX(y11, y21));
            
            if (x_overlap * y_overlap > MIN(i->first.width * i->first.height, final[j].first.width * final[j].first.height) * NMS)
            {
                exist_number++;
                if (exist_number > 1)
                {
                    times[index]--;
                    break;
                }
                index = j;
                times[j]++;
            }
        }
        
        if (exist_number > 1)
        {
            continue;
        }
        else if (exist_number == 1)
        {
            int current_times = times[index];
            final[index].first.x      = WEIGHTED_AVERAGE(i->first.x, final[index].first.x, current_times);
            final[index].first.y      = WEIGHTED_AVERAGE(i->first.y, final[index].first.y, current_times);
            final[index].first.width  = WEIGHTED_AVERAGE(i->first.width, final[index].first.width, current_times);
            final[index].first.height = WEIGHTED_AVERAGE(i->first.height, final[index].first.height, current_times);
            
            final[index].second = WEIGHTED_AVERAGE(i->second, final[index].second, current_times);
        }
        else
        {
            final.push_back(*i);
            times.push_back(1);
        }
    }
}

Prediction Detector::Detect(const cv::Mat& img)
{
    cv::Mat img_cpy = img.clone();
    Fragments fragments;

    
    Segment(img_cpy, fragments);
    cv::Mat image = cv::Mat::ones(img.rows, img.cols, CV_8UC1);
    
    
    Prediction prediction, deoverlapped_prediction;
    Predict(fragments, prediction);
    
    ClearVector(fragments.first);
    ClearVector(fragments.second);
    
    Deoverlapped(prediction, deoverlapped_prediction);
    ClearVector(prediction);
    
    return deoverlapped_prediction;
}

bool compare_with_rect(const prob& a, const prob& b)
{
    return a.first.height < b.first.height;
}

template <class T>
void ClearVector(std::vector<T>& vt)
{
    std::vector<T> vtTmp;
    vtTmp.swap(vt);
}