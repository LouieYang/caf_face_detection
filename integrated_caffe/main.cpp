#include "face_detection.hpp"

int main()
{
    cv::Mat img = cv::imread("tmp.jpg");
    cv::imshow("test", face_detection(img));
    cv::waitKey();
}
