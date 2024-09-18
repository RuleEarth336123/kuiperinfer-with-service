#ifndef SERVICE_HANDLE_
#define SERVICE_HANDLE_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "image_util.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

#include <chrono>
#include <cstdio>
#include <httplib.h>
#include <string>

using std::vector;
using std::string;

struct ImageTask {
    string image_path;
    string param_path;
    string bin_path;
    uint32_t batch_size;
    float conf_thresh;
    float iou_thresh;
    vector<Detection>& detections;
    int index;
};

namespace Yolo{
    kuiper_infer::sftensor preProcessImage(const cv::Mat& image ,const int32_t input_h,const int32_t input_w);

    void processSingleImage(ImageTask task);

    void processImage(const std::vector<std::string>& image_paths, const std::string& param_path,
                const std::string& bin_path, const uint32_t batch_size,
                const float conf_thresh = 0.25f, const float iou_thresh = 0.25f);
}


namespace HttpServer{
    void handleProcessImage(const httplib::Request& req, httplib::Response& res);
}



#endif