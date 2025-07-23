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
        if (!f.empty()) {
            cv::resize(f, f, cv::Size(640, 360));  // ou cv::Size(320, 180)
            frame = f;
        }
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
/**************************************************************************************/
cv::Mat postprocess(float* ll_output, cv::Mat& original_frame, std::vector<cv::Point>& medianPoints,
                    LaneData& laneData, LineIntersect& intersect) {
    const int height_mask = 224;
    const int width_mask = 224;
    const int width_win = 640;
    const int height_win = 360;

    // ✅ Usa apenas o canal que representa a linha desejada (ex: canal 1 = linha central)
    int selected_channel = 0;
    cv::Mat ll_mask(height_mask, width_mask, CV_32FC1, ll_output + selected_channel * height_mask * width_mask);

    // Binarização
    cv::Mat ll_bin;
    cv::threshold(ll_mask, ll_bin, 0.1, 255, cv::THRESH_BINARY);
    ll_bin.convertTo(ll_bin, CV_8UC1);

/*     // Exibir a máscara usada (opcional para debug)
    cv::imshow("ll_mask_raw", ll_bin);
    cv::waitKey(1); */

    // Morfologia para limpeza
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(ll_bin, ll_bin, cv::MORPH_CLOSE, kernel);
    cv::dilate(ll_bin, ll_bin, kernel);

    // Redimensionar para o tamanho do frame original
    cv::Mat ll_resized;
    cv::resize(ll_bin, ll_resized, original_frame.size(), 0, 0, cv::INTER_NEAREST);

    // Aplicar ROI
    int roi_start_y = static_cast<int>(0.50 * original_frame.rows);
    int roi_end_y = static_cast<int>(0.95 * original_frame.rows);
    ll_resized(cv::Rect(0, 0, original_frame.cols, roi_start_y)) = 0;
    ll_resized(cv::Rect(0, roi_end_y, original_frame.cols, original_frame.rows - roi_end_y)) = 0;

    // Processar a máscara
    MaskProcessor processor;
    LineCoef left_coeffs, right_coeffs;
    cv::Mat mask_output;
    processor.processMask(ll_resized, mask_output, medianPoints, left_coeffs, right_coeffs, intersect);

    // Criar uma cópia da imagem original para desenhar as linhas
    cv::Mat result_frame = original_frame.clone();

    // Desenhar linhas diretamente na imagem original
    if (!mask_output.empty()) {
        // As linhas já foram desenhadas dentro da mask_output, 
        // então redimensione e sobreponha APENAS elas
        cv::Mat resized_mask_output;
        cv::resize(mask_output, resized_mask_output, original_frame.size(), 0, 0, cv::INTER_NEAREST);

        // Usar canais BGR diretamente (sem máscara binária)
        for (int y = 0; y < resized_mask_output.rows; ++y) {
            for (int x = 0; x < resized_mask_output.cols; ++x) {
                cv::Vec3b pix = resized_mask_output.at<cv::Vec3b>(y, x);
                if (pix != cv::Vec3b(0, 0, 0)) {
                    result_frame.at<cv::Vec3b>(y, x) = pix;  // Desenha o pixel colorido sobre a original
                }
            }
        }
    }

    // === Resto do processamento de geometria (mantido igual) ===
    laneData.valid = !medianPoints.empty();
    laneData.num_points = 0;

    if (medianPoints.size() >= 5) {
        // desvio do centro da pista
        intersect.xlt = (left_coeffs.m * (height_win / 2) + left_coeffs.b);
        intersect.xlb = (left_coeffs.m * height_win + left_coeffs.b);
        intersect.xrt = (right_coeffs.m * (height_win / 2) + right_coeffs.b);
        intersect.xrb = (right_coeffs.m * height_win + right_coeffs.b);
/*         std::cout << "xleft top: " << xlt << std::endl;
        std::cout << "xleft bottom: " << xlb << std::endl;        
        std::cout << "xright top: " << xrt << std::endl;
        std::cout << "xright bottom: " << xrb << std::endl << std::endl; */

        float xmt = (intersect.xrt + intersect.xlt) / 2;
        float xmb = (intersect.xrb + intersect.xlb) / 2;
/*         std::cout << "xm top: " << xmt << std::endl;
        std::cout << "xm bottom: " << xmb << std::endl << std::endl; */

        cv::Point xmtop(xmt, height_win / 2);
        cv::Point xmbottom(xmb, height_win);
        cv::line(result_frame, xmtop, xmbottom, cv::Scalar(255, 255, 255), 2); 

        float P1_x_img_frame = (Asy * height_win + Bsy) * ( xmb -(width_win / 2) );
        float P2_x_img_frame = (Asy * height_win / 2 + Bsy) * (xmt - (width_win / 2) );
        float deltaX_car_frame = P2_x_car_frame - P1_x_car_frame;
        float deltaY_car_frame = P2_x_img_frame - P1_x_img_frame;

        /* std::cout << "P2: " << P2_x_img_frame << std::endl;
        std::cout << "P1: " << P1_x_img_frame << std::endl << std::endl; */

        if (std::abs(deltaX_car_frame) > 1e-8) {
            intersect.slope = deltaY_car_frame / deltaX_car_frame;
            intersect.offset = P2_x_img_frame - intersect.slope * P2_x_car_frame;

            //std::cout << "Slope: " << slope_car_frame << std::endl;
            intersect.psi = std::atan(intersect.slope);
        } else {
            intersect.offset = 0.0f;
            intersect.psi = 0.0f;
            std::cout << "Aviso: deltaX_car_frame é zero, valores padrão definidos para intersect." << std::endl;
        }
    } else {
        intersect.offset = 0.0f;
        intersect.psi = 0.0f;
        std::cout << "Aviso: medianPoints está vazio, valores padrão definidos para intersect." << std::endl;
    }

    if (laneData.valid) {
        int step = medianPoints.size() > 10 ? medianPoints.size() / 10 : 1;
        for (size_t i = 0; i < medianPoints.size() && laneData.num_points < 10; i += step) {
            if (medianPoints[i].y >= roi_start_y && medianPoints[i].y <= roi_end_y) {
                laneData.points[laneData.num_points].x =  0.000458 * (medianPoints[i].x - (width_win/2));
                laneData.points[laneData.num_points].y = 0.001623 * ((height_win*0.95) - medianPoints[i].y);
                laneData.num_points++;
            }
        }
    }
    return result_frame;
}


