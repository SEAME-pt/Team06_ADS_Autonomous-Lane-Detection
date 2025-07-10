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

    // Criar máscara a partir da saída do modelo (canal único)
    cv::Mat ll_mask(height, width, CV_32FC1, ll_output);
    cv::Mat ll_bin;
    
    // Aplicar threshold para binarizar
    cv::threshold(ll_mask, ll_bin, 0.1, 255, cv::THRESH_BINARY);
    ll_bin.convertTo(ll_bin, CV_8UC1);

    // Aplicar operações morfológicas para limpar ruído e engrossar linhas
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(ll_bin, ll_bin, cv::MORPH_CLOSE, kernel);
    cv::dilate(ll_bin, ll_bin, kernel);

    // Exibir máscara para depuração
    cv::imshow("ll_mask_raw", ll_bin);
    cv::waitKey(1);

    // Aplicar ROI (zerar regiões fora do interesse)
    ll_bin(cv::Rect(0, 0, width, roi_start_y)) = 0;
    ll_bin(cv::Rect(0, roi_end_y, width, height - roi_end_y)) = 0;

    // Redimensionar para o tamanho do frame original
    cv::Mat ll_resized;
    cv::resize(ll_bin, ll_resized, original_frame.size(), 0, 0, cv::INTER_NEAREST);

    // Processar máscara para extrair pontos e linhas
    MaskProcessor processor;
    cv::Mat mask_output;
    processor.processMask(ll_resized, mask_output, medianPoints);

    // Criar máscara colorida para overlay
    cv::Mat color_mask = cv::Mat::zeros(original_frame.size(), CV_8UC3);
    color_mask.setTo(cv::Scalar(255, 0, 255), ll_resized); // Magenta para linhas

    // Combinar a imagem original com a máscara colorida
    cv::Mat result_frame;
    cv::addWeighted(original_frame, 0.5, color_mask, 1.0, 0, result_frame);

    // Cálculos de laneData e intersect (mantidos inalterados)
    laneData.valid = !medianPoints.empty();
    laneData.num_points = 0;

    if (medianPoints.size() >= 5) {
        float P1_x_img_frame = (Asy * roi_end_y + Bsy) * (medianPoints.back().x - 224);
        float P2_x_img_frame = (Asy * roi_start_y + Bsy) * (medianPoints.front().x - 224);
        float deltaX_car_frame = P2_x_car_frame - P1_x_car_frame;
        float deltaY_car_frame = P2_x_img_frame - P1_x_img_frame;

        if (std::abs(deltaX_car_frame) > 1e-8) {
            float slope_car_frame = deltaY_car_frame / deltaX_car_frame;
            float B = P2_x_img_frame - slope_car_frame * P2_x_car_frame;
            intersect.offset_cm = B;
            intersect.psi = std::atan(slope_car_frame);
        } else {
            intersect.offset_cm = 0.0f;
            intersect.psi = 0.0f;
            std::cout << "Aviso: deltaX_car_frame é zero, valores padrão definidos para intersect." << std::endl;
        }
    } else {
        intersect.offset_cm = 0.0f;
        intersect.psi = 0.0f;
        std::cout << "Aviso: medianPoints está vazio, valores padrão definidos para intersect." << std::endl;
    }

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


