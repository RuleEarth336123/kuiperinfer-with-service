#include "handle.h"
#include "json11.hpp"
#include "data/tensor.hpp"
#include <memory>

using json11::Json;
using std::string;

using namespace httplib;
std::string dump_headers(const Headers &headers) {
    std::string s;
    char buf[BUFSIZ];

    for (auto it = headers.begin(); it != headers.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
        s += buf;
    }

    return s;
}

std::string log(const Request &req, const Response &res) {
    std::string s;
    char buf[BUFSIZ];

    s += "================================\n";

    snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
            req.version.c_str(), req.path.c_str());
    s += buf;

    std::string query;
    for (auto it = req.params.begin(); it != req.params.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%c%s=%s",
                (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
                x.second.c_str());
        query += buf;
    }
    snprintf(buf, sizeof(buf), "%s\n", query.c_str());
    s += buf;

    s += dump_headers(req.headers);

    s += "--------------------------------\n";

    snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
    s += buf;
    s += dump_headers(res.headers);
    s += "\n";

    if (!res.body.empty()) { s += res.body; }

    s += "\n";

    return s;
}

kuiper_infer::sftensor Yolo::preProcessImage(const cv::Mat& image, const int32_t input_h, const int32_t input_w) {
    assert(!image.empty());
    using namespace kuiper_infer;
    const int32_t input_c = 3;

    int stride = 32;
    cv::Mat out_image;
    Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114}, true);

    cv::Mat rgb_image;
    cv::cvtColor(out_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat normalize_image;
    rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

    std::vector<cv::Mat> split_images;
    cv::split(normalize_image, split_images);
    assert(split_images.size() == input_c);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    input->Fill(0.f);

    int index = 0;
    int offset = 0;
    for (const auto& split_image : split_images) {
        assert(split_image.total() == input_w * input_h);
        const cv::Mat& split_image_t = split_image.t();
        memcpy(input->slice(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
        index += 1;
        offset += split_image.total();
    }
    return input;
}

void Yolo::processSingleImage(ImageTask task) {


}

void Yolo::processImage(const std::vector<std::string>& image_paths, const std::string& param_path,
                  const std::string& bin_path, const uint32_t batch_size, const float conf_thresh,
                  const float iou_thresh) {
    using namespace kuiper_infer;
    const int32_t input_h = 640;
    const int32_t input_w = 640;

    RuntimeGraph graph(param_path, bin_path);
    graph.Build();

    assert(batch_size == image_paths.size());
    std::vector<sftensor> inputs;
    for (uint32_t i = 0; i < batch_size; ++i) {
        const auto& input_image = cv::imread(image_paths.at(i));
        sftensor input = Yolo::preProcessImage(input_image, input_h, input_w);
        assert(input->rows() == 640);
        assert(input->cols() == 640);
        inputs.push_back(input);
    }

    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    std::cout << "begin to set inputs ..." <<std::endl;
    graph.set_inputs("pnnx_input_0", inputs);
    for (int i = 0; i < 1; ++i) {
        graph.Forward(true);
    }
    outputs = graph.get_outputs("pnnx_output_0");
    assert(outputs.size() == inputs.size());
    assert(outputs.size() == batch_size);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto& image = cv::imread(image_paths.at(i));
        const int32_t origin_input_h = image.size().height;
        const int32_t origin_input_w = image.size().width;

        const auto& output = outputs.at(i);
        assert(!output->empty());
        const auto& shapes = output->shapes();
        assert(shapes.size() == 3);

        const uint32_t elements = shapes.at(1);
        const uint32_t num_info = shapes.at(2);
        std::vector<Detection> detections;

        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> class_ids;

        const uint32_t b = 0;
        for (uint32_t e = 0; e < elements; ++e) {
        float cls_conf = output->at(b, e, 4);
        if (cls_conf >= conf_thresh) {
            int center_x = (int)(output->at(b, e, 0));
            int center_y = (int)(output->at(b, e, 1));
            int width = (int)(output->at(b, e, 2));
            int height = (int)(output->at(b, e, 3));
            int left = center_x - width / 2;
            int top = center_y - height / 2;

            int best_class_id = -1;
            float best_conf = -1.f;
            for (uint32_t j = 5; j < num_info; ++j) {
            if (output->at(b, e, j) > best_conf) {
                best_conf = output->at(b, e, j);
                best_class_id = int(j - 5);
            }
            }

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(best_conf * cls_conf);
            class_ids.emplace_back(best_class_id);
        }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);

        for (int idx : indices) {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        ScaleCoords(cv::Size{input_w, input_h}, det.box, cv::Size{origin_input_w, origin_input_h});

        det.conf = confs[idx];
        det.class_id = class_ids[idx];
        detections.emplace_back(det);
        }

        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 2;

        for (const auto& detection : detections) {
        cv::rectangle(image, detection.box, cv::Scalar(255, 255, 255), 4);
        cv::putText(image, std::to_string(detection.class_id),
                    cv::Point(detection.box.x, detection.box.y), font_face, font_scale,
                    cv::Scalar(255, 255, 0), 4);
        }
        cv::imwrite(std::string("output") + std::to_string(i) + ".jpg", image);
    }

}

void HttpServer::handleProcessImage(const httplib::Request& req, httplib::Response& res) {

    string err;

    Json requestJson = Json::parse(req.body, err);
    if (!err.empty()) {
        res.set_content("Invalid JSON", "text/plain");
        res.status = 400; // Bad Request
        return;
    }

    auto images = requestJson["images"].array_items();
    vector<string> image_paths;
    for (const auto& image : images) {
        image_paths.push_back(image.string_value());
        std::cout << "image: " << image.string_value() << std::endl;
    }

    int batch_size = requestJson["batch_size"].int_value();

    batch_size = image_paths.size();
    string param_path = requestJson["param_path"].string_value();
    string bin_path = requestJson["bin_path"].string_value();

    if (image_paths.empty() || batch_size == 0 || param_path.empty() || bin_path.empty()) {
        res.set_content("Missing or invalid parameters", "text/plain");
        res.status = 400; 
        return;
    }

    Yolo::processImage(image_paths, param_path, bin_path, batch_size);

    // 创建响应的Json对象
    json11::Json responseJson = json11::Json::object {
        {"message", "process is ok"}
    };

    std::string response_string = responseJson.dump();
    res.set_content(response_string, "application/json");
    std::cout << log(req,res) << std::endl;
}
