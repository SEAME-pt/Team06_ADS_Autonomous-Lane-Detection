#include "lane.hpp"
#include "mask.hpp"

TensorRTInference::TensorRTInference(const std::string& engine_path) {
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) throw std::runtime_error("Erro ao abrir engine");

    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    context = engine->createExecutionContext();

    allocateBuffers();
}

/**************************************************************************************/
TensorRTInference::~TensorRTInference() {
    for (auto& mem : inputBuffers) cudaFree(mem.device);
    for (auto& mem : outputBuffers) cudaFree(mem.device);
}

/**************************************************************************************/
void TensorRTInference::allocateBuffers() {
    int nbBindings = engine->getNbBindings();
    inputBuffers.resize(1);
    outputBuffers.resize(nbBindings - 1);
    bindings.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) vol *= dims.d[j];

        size_t typeSize = sizeof(float);

        void* deviceMem;
        cudaMalloc(&deviceMem, vol * typeSize);
        float* hostMem = new float[vol];

        bindings[i] = deviceMem;
        if (engine->bindingIsInput(i)) {
            inputBuffers[0] = {deviceMem, hostMem, vol * typeSize};
        } else {
            outputBuffers[i - 1] = {deviceMem, hostMem, vol * typeSize};
        }
    }
}

/**************************************************************************************/
std::vector<std::vector<float>> TensorRTInference::infer(const std::vector<float>& inputData) {
    cudaMemcpy(inputBuffers[0].device, inputData.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);
    context->executeV2(bindings.data());
    std::vector<std::vector<float>> outputs;
    for (auto& out : outputBuffers) {
        cudaMemcpy(out.host, out.device, out.size, cudaMemcpyDeviceToHost);
        outputs.emplace_back(out.host, out.host + out.size / sizeof(float));
    }
    return outputs;
}

/**************************************************************************************/
CSICamera::CSICamera(int width, int height, int fps) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-mode=4 ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height
             << ", format=NV12, framerate=" << fps << "/1 ! "
             << "nvvidconv flip-method=0 ! video/x-raw, width=" << width
             << ", height=" << height << ", format=BGRx ! "
             << "videoconvert ! video/x-raw, format=BGR ! appsink";
    cap.open(pipeline.str(), cv::CAP_GSTREAMER);
}

/**************************************************************************************/
void CSICamera::start() {
    running = true;
    thread = std::thread(&CSICamera::update, this);
}

/**************************************************************************************/
void CSICamera::stop() {
    running = false;
    if (thread.joinable()) thread.join();
    cap.release();
}

/**************************************************************************************/
cv::Mat CSICamera::read() const {
    return frame.clone();
}

/**************************************************************************************/
void CSICamera::update() {
    while (running) {
        cv::Mat f;
        cap.read(f);
        if (!f.empty()) frame = f;
    }
}

/**************************************************************************************/
std::vector<float> preprocess_frame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(448, 448));
    std::vector<float> inputData(3 * 448 * 448);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 448; ++i) {
            for (int j = 0; j < 448; ++j) {
                inputData[idx++] = resized.at<cv::Vec3b>(i, j)[2 - c] / 255.0f;
            }
        }
    }
    return inputData;
}

/**************************************************************************************/
#include "mask.hpp"

cv::Mat postprocess(float* da_output, float* ll_output, cv::Mat& original_frame, 
                    std::vector<cv::Point>& medianPoints, LaneData& laneData, LineIntersect& intersect) {
    const int height = original_frame.rows;
    const int width = original_frame.cols;
    int roi_start_y = static_cast<int>(0.50 * height); // 224
    int roi_end_y = static_cast<int>(0.95 * height);   // 425.6
    int roi_height = roi_end_y - roi_start_y;
    LineCoef left_coeffs, right_coeffs;

    cv::Rect roi(0, roi_start_y, width, roi_height);
    cv::Mat da_logits(2, height * width, CV_32FC1, da_output);
    cv::Mat ll_logits(2, height * width, CV_32FC1, ll_output);
    cv::Mat da_mask(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat ll_mask(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < height * width; ++i) {
        float da0 = da_logits.at<float>(0, i);
        float da1 = da_logits.at<float>(1, i);
        float ll0 = ll_logits.at<float>(0, i);
        float ll1 = ll_logits.at<float>(1, i);

        da_mask.at<uchar>(i / width, i % width) = (da1 > da0) ? 255 : 0;
        ll_mask.at<uchar>(i / width, i % width) = (ll1 > ll0) ? 255 : 0;
    }

    da_mask(cv::Rect(0, 0, width, roi_start_y)) = 0;
    da_mask(cv::Rect(0, roi_end_y, width, height - roi_end_y)) = 0;
    ll_mask(cv::Rect(0, 0, width, roi_start_y)) = 0;
    ll_mask(cv::Rect(0, roi_end_y, width, height - roi_end_y)) = 0;

    cv::Mat da_resized, ll_resized;
    cv::resize(da_mask, da_resized, original_frame.size());
    cv::resize(ll_mask, ll_resized, original_frame.size());

    MaskProcessor processor;
    cv::Mat mask_output;
    //processor.processMask(da_resized, ll_resized, mask_output, medianPoints, left_coeffs, right_coeffs);
    processor.processMask(da_resized, mask_output, medianPoints);

    cv::Mat result_frame = mask_output.clone();
    
    //intersect = findIntersect(left_coeffs, right_coeffs, height, width);
    laneData.valid = !medianPoints.empty();
    laneData.num_points = 0;

    /*
        d_mtr = s(y) * x_img_frame
        s(y) = Asy * y_img_frame + Bsy
        d_mtr = Asy * y_img_frame + Bsy
    */

    float P1_x_img_frame = (Asy * roi_end_y + Bsy) * (medianPoints.back().x - 244);
    float P2_x_img_frame = (Asy * roi_start_y + Bsy) * (medianPoints.front().x - 244); // 

    float deltaX_car_frame = P2_x_car_frame - P1_x_car_frame; // Calibration data
    float deltaY_car_frame = P2_x_img_frame - P1_x_img_frame; // Instante measured data

    float slope_car_frame = deltaY_car_frame / deltaX_car_frame;

    float B = P2_x_img_frame - slope_car_frame * P2_x_car_frame; // b = y - m * x

    intersect.offset_cm = B;

    intersect.psi = atan(slope_car_frame);

/*     std::cout << "[" <<  __func__ <<"]" << std::endl
                << "median points back: " << medianPoints.back().x << std::endl
                << "median points front: " << medianPoints.front().x << std::endl; */

    if (laneData.valid) {
        int step = medianPoints.size() > 10 ? medianPoints.size() / 10 : 1;
        for (size_t i = 0; i < medianPoints.size() && laneData.num_points < 10; i += step) {
            if (medianPoints[i].y >= roi_start_y && medianPoints[i].y <= roi_end_y) {
                laneData.points[laneData.num_points].x = (medianPoints[i].x);
                laneData.points[laneData.num_points].y = (medianPoints[i].y);
                laneData.num_points++;
            }
        }
    }

    return result_frame;
}

/**************************************************************************************/
LineIntersect  findIntersect(const LineCoef& left_coeffs, const LineCoef& right_coeffs, int height, int width) {
    LineIntersect intersect;
    intersect.valid = false;

    int roi_start_y = static_cast<int>(0.50 * height); // y = 224
    int roi_end_y = static_cast<int>(0.95 * height);   // y = 426

    if (left_coeffs.valid && right_coeffs.valid) {
        intersect.xl_t = { static_cast<float>(left_coeffs.m * roi_start_y + left_coeffs.b), static_cast<float>(roi_start_y) };
        intersect.xl_b = { static_cast<float>(left_coeffs.m * roi_end_y + left_coeffs.b), static_cast<float>(roi_end_y) };
        intersect.xr_t = { static_cast<float>(right_coeffs.m * roi_start_y + right_coeffs.b), static_cast<float>(roi_start_y) };
        intersect.xr_b = { static_cast<float>(right_coeffs.m * roi_end_y + right_coeffs.b), static_cast<float>(roi_end_y) };
        
        intersect.ratio_top = (intersect.xr_t.x - width / 2.0f) / (intersect.xr_t.x - intersect.xl_t.x);
        intersect.xs_b = intersect.xr_b.x - intersect.ratio_top * (intersect.xr_b.x - intersect.xl_b.x);
        intersect.slope = (intersect.xs_b - width / 2.0f + 22.5) / (roi_end_y - roi_start_y);
        intersect.psi = std::atan(intersect.slope);
        intersect.valid = true;

        // Pixels on top and bottom image
        intersect.x_px_t = intersect.xr_t.x - intersect.xl_t.x;
        intersect.x_px_b = intersect.xr_b.x - intersect.xl_b.x;

        // Scale Factor | d = s(y).x_px + b | with b = 0
        intersect.scaleFactor_t = intersect.w_real / intersect.x_px_t;
        intersect.scaleFactor_b = intersect.w_real / intersect.x_px_b;

        // var a | s(y) = a * y + b (=) a = delta(s(y)) / delta(y) |,  with b = 0 
        intersect.var_a = (intersect.scaleFactor_b - intersect.scaleFactor_t) / (intersect.xl_b.y - intersect.xl_t.y);

        // var b para o top que será igual ao bottom
        intersect.var_b = intersect.scaleFactor_t - (intersect.var_a * intersect.xl_t.y);
    }

    return intersect;
}
/**************************************************************************************/

void drawLanes(LineCoef left_coeffs, LineCoef right_coeffs, cv::Mat& result_frame, std::vector<cv::Point> medianPoints, int roi_start_y,  int roi_end_y){
    if (left_coeffs.valid && right_coeffs.valid) {
            std::vector<cv::Point> left_line_points, right_line_points;
            for (int y = roi_start_y; y < roi_end_y; y++) {
                int left_x = static_cast<int>(left_coeffs.m * y + left_coeffs.b);
                int right_x = static_cast<int>(right_coeffs.m * y + right_coeffs.b);
                if (left_x >= 0 && left_x < result_frame.cols)
                left_line_points.push_back(cv::Point(left_x, y));
                if (right_x >= 0 && right_x < result_frame.cols)
                right_line_points.push_back(cv::Point(right_x, y));
            }
            
            if (!left_line_points.empty() && !right_line_points.empty()) {
                cv::line(result_frame, left_line_points.front(), left_line_points.back(), cv::Scalar(0, 0, 255), 2);
                cv::line(result_frame, right_line_points.front(), right_line_points.back(), cv::Scalar(255, 0, 0), 2);
            }
            
            std::vector<cv::Point> roi_median_points;
            for (const auto& p : medianPoints) {
                if (p.y >= roi_start_y && p.y < roi_end_y) {
                    roi_median_points.push_back(p);
                }
            }
            if (roi_median_points.size() >= 2) {
                cv::line(result_frame, roi_median_points.front(), roi_median_points.back(), cv::Scalar(0, 255, 0), 2);
            }
        }
}

/**************************************************************************************/

