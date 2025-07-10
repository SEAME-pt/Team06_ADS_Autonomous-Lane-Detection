#include "lane.hpp"
#include "mask.hpp"

TensorRTInference::TensorRTInference(const std::string& engine_path) {
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) throw std::runtime_error("Error engine");

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
    outputBuffers.resize(1);
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
            outputBuffers[0] = {deviceMem, hostMem, vol * typeSize};
        }
    }
}

/**************************************************************************************/
std::vector<float> TensorRTInference::infer(const std::vector<float>& inputData) {
      cudaMemcpy(inputBuffers[0].device, inputData.data(), inputBuffers[0].size, cudaMemcpyHostToDevice);
        context->executeV2(bindings.data());
        cudaMemcpy(outputBuffers[0].host, outputBuffers[0].device, outputBuffers[0].size, cudaMemcpyDeviceToHost);
        return std::vector<float>(outputBuffers[0].host, outputBuffers[0].host + outputBuffers[0].size / sizeof(float));
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
    cv::resize(frame, resized, cv::Size(224, 224));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f; // R
    channels[1] = (channels[1] - 0.456f) / 0.224f; // G
    channels[2] = (channels[2] - 0.406f) / 0.225f; // B
    cv::merge(channels,  resized);

    std::vector<float> inputData;
    for (int i = 0; i < 3; ++i) {
        inputData.insert(inputData.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
    }
    return inputData;
}

cv::Mat postprocess(float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints,
                    LaneData& laneData, LineIntersect& intersect) {

    const int height = original_frame.rows;
    const int width = original_frame.cols;
    int roi_start_y = static_cast<int>(0.50 * height); // 224 / 2
    int roi_end_y = static_cast<int>(0.95 * height);   // 224 * 0.95
    int roi_height = roi_end_y - roi_start_y;

    cv::Rect roi(0, roi_start_y, width, roi_height);
    cv::Mat ll_logits(2, height * width, CV_32FC1, ll_output);
    cv::Mat ll_mask(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < height * width; ++i) {
        float ll0 = ll_logits.at<float>(0, i);
        float ll1 = ll_logits.at<float>(1, i);

        ll_mask.at<uchar>(i / width, i % width) = (ll1 > ll0) ? 255 : 0;
    }

    ll_mask(cv::Rect(0, 0, width, roi_start_y)) = 0;
    ll_mask(cv::Rect(0, roi_end_y, width, height - roi_end_y)) = 0;

    cv::Mat ll_resized;
    cv::resize(ll_mask, ll_resized, original_frame.size());

    MaskProcessor processor;
    cv::Mat mask_output;
    processor.processMask(ll_resized, mask_output, medianPoints);

    cv::Mat result_frame = mask_output.clone();
    
    laneData.valid = !medianPoints.empty();
    laneData.num_points = 0;

    /*
        d_mtr = s(y) * x_img_frame
        s(y) = Asy * y_img_frame + Bsy
        d_mtr = Asy * y_img_frame + Bsy
    */
    
    // Verificar se medianPoints tem pelo menos um elemento
    if (medianPoints.size() >= 5) {
        // Realizar cálculos apenas se medianPoints não estiver vazio
        float P1_x_img_frame = (Asy * roi_end_y + Bsy) * (medianPoints.back().x - 224);
        float P2_x_img_frame = (Asy * roi_start_y + Bsy) * (medianPoints.front().x - 224);
        float deltaX_car_frame = P2_x_car_frame - P1_x_car_frame; // Calibration data
        float deltaY_car_frame = P2_x_img_frame - P1_x_img_frame; // Instante measured data

        // Verificar se deltaX_car_frame não é zero para evitar divisão por zero
        if (std::abs(deltaX_car_frame) > 1e-8) { // Usar um pequeno epsilon para evitar divisão por zero
            float slope_car_frame = deltaY_car_frame / deltaX_car_frame;
            float B = P2_x_img_frame - slope_car_frame * P2_x_car_frame; // b = y - m * x

            intersect.offset_cm = B;
            intersect.psi = std::atan(slope_car_frame);
        } else {
            // Caso deltaX_car_frame seja zero, definir valores padrão para intersect
            intersect.offset_cm = 0.0f;
            intersect.psi = 0.0f;
            std::cout << "Aviso: deltaX_car_frame é zero, valores padrão definidos para intersect." << std::endl;
        }
    } else {
        // Caso medianPoints esteja vazio, definir valores padrão
        intersect.offset_cm = 0.0f;
        intersect.psi = 0.0f;
        std::cout << "Aviso: medianPoints está vazio, valores padrão definidos para intersect." << std::endl;
    }

/*     std::cout << "[" <<  __func__ <<"]" << std::endl
                << "median points back: " << medianPoints.back().x << " " << medianPoints.back().y << std::endl
                << "median points front: " << medianPoints.front().x << " " << medianPoints.back().y << std::endl; */

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


