#include "control_systems/utils_control.hpp"
#include "object_detection/fps.hpp"
#include "object_detection/frame.hpp"

// Headers específicos para object detection
#include "object_detection/inferObject.hpp" // TensorRTYOLO

// Variáveis atómicas globais
std::atomic<double> current_speed_ms{0.0};
std::atomic<bool> keep_running{true};

// Estrutura para sincronizar resultados
struct SyncResults {
    // Lane detection results
    struct {
        double offset = 0.0;
        double psi = 0.0;
        bool valid = false;
        cv::Mat processed_frame;
        std::mutex mutex;
    } lane_data;
    
    // Object detection results
    struct {
        std::vector<Detection> detections;
        bool valid = false;
        std::mutex mutex;
    } object_data;
};

// Handler de sinal para terminar o programa
void signalHandler(int signum) {
    std::cout << "\nSinal de interrupcao (" << signum << ") recebido. A terminar a aplicacao..." << std::endl;
    keep_running.store(false);
}

// Thread para lane detection (crítica - baixo frame skipping)
void laneDetectionThread(
    TensorRTInference& lane_detector,
    CSICamera& camera,
    NMPCController& mpc,
    FServo& servo,
    BackMotors& backMotors,
    PID& pid,
    SyncResults& sync_results,
    ZmqPublisher* zmq_publisher
) {
    FrameSkipper lane_skipper(FrameSkipper::FIXED, 2, 30.0); // Skip apenas 2 frames
    
    auto pid_last_time = std::chrono::steady_clock::now();
    double last_delta = 0.0;
    double setpoint_velocity = 1.0; // m/s desejados
    
    while (keep_running.load()) {
        cv::Mat frame = camera.read();
        if (frame.empty()) continue;
        
        // Processa lanes com skipping mínimo
        if (lane_skipper.shouldProcessFrame()) {
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            // Lane detection
            std::vector<float> input = preprocess_frame(frame);
            auto outputs = lane_detector.infer(input);
            
            std::vector<cv::Point> medianPoints;
            LaneData laneData;
            LineIntersect intersect;
            cv::Mat result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);
            
            auto inference_end = std::chrono::high_resolution_clock::now();
            double inference_time = std::chrono::duration<double>(inference_end - inference_start).count();
            lane_skipper.recordProcessingTime(inference_time);
            
            // Controle baseado em lanes
            double offset = intersect.offset;
            double psi = intersect.psi;
            double delta = last_delta;
            
            if (!std::isnan(offset) && !std::isnan(psi)) {
                delta = -mpc.computeControl(offset, psi, 0.7);
                
                // Controle de steering
                double target_steering_angle = delta * 180.0 / M_PI;
                int steering_angle = static_cast<int>(target_steering_angle);
                steering_angle = std::max(-40, std::min(40, steering_angle));
                servo.set_steering(steering_angle);
                
                last_delta = delta;
                
                // ZMQ publishing
                int lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
                if (zmq_publisher && zmq_publisher->isConnected()) {
                    std::stringstream ss;
                    ss << "lane:" << lane;
                    zmq_publisher->publishMessage(ss.str());
                }
            }
            
            // Motor control com PID
            auto pid_now = std::chrono::steady_clock::now();
            double pid_dt = std::chrono::duration<double>(pid_now - pid_last_time).count();
            
            if (pid_dt >= 0.02) { // 50 ms → 20 Hz
                double v_actual = current_speed_ms.load();
                double motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
                backMotors.setSpeed(static_cast<int>(motor_pwm));
                pid_last_time = pid_now;
            }
            
            // Atualizar dados sincronizados
            {
                std::lock_guard<std::mutex> lock(sync_results.lane_data.mutex);
                sync_results.lane_data.offset = offset;
                sync_results.lane_data.psi = psi;
                sync_results.lane_data.valid = true;
                sync_results.lane_data.processed_frame = result.clone();
            }
        }
    }
}

// Thread para object detection (adaptativo - maior frame skipping)
void objectDetectionThread(
    TensorRTYOLO& object_detector,
    CSICamera& camera,
    SyncResults& sync_results
) {
    FrameSkipper object_skipper(FrameSkipper::ADAPTIVE, 8, 15.0); // Skip agressivo
    
    while (keep_running.load()) {
        cv::Mat frame = camera.read();
        if (frame.empty()) continue;
        
        if (object_skipper.shouldProcessFrame()) {
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            // Object detection
            std::vector<Detection> detections = object_detector.infer(frame);
            
            auto inference_end = std::chrono::high_resolution_clock::now();
            double inference_time = std::chrono::duration<double>(inference_end - inference_start).count();
            object_skipper.recordProcessingTime(inference_time);
            
            // Atualizar dados sincronizados
            {
                std::lock_guard<std::mutex> lock(sync_results.object_data.mutex);
                sync_results.object_data.detections = detections;
                sync_results.object_data.valid = true;
            }
        }
    }
}

int main() {
    std::signal(SIGINT, signalHandler);
    
    try {
        // === INICIALIZAÇÕES ===
        
        // Lane detection setup (usa a configuração existente)
        auto laneControl = initLaneControl();
        auto& lane_detector = laneControl->trt;
        auto& camera = laneControl->cam;
        
        // Object detection setup
        std::string obj_engine_path = "../best.engine";
        TensorRTYOLO object_detector(obj_engine_path, 640);
        
        // Controle de motores e servos
        NMPCController mpc;
        BackMotors backMotors;
        if (!initMotors(backMotors)) return 1;
        
        FServo servo;
        if (!initServo(servo)) return -1;
        
        // Sistemas auxiliares
        std::shared_ptr<CANMessageProcessor> messageProcessor;
        auto canBusManager = initCanBus(messageProcessor);
        if (!canBusManager) return 1;
        
        zmq::context_t context(1);
        ZmqPublisher* zmq_publisher = initZmq(context);
        
        // PID Controller
        PID pid;
        
        // Estrutura para sincronização de resultados
        SyncResults sync_results;
        
        // === LANÇAR THREADS ===
        
        std::cout << "Iniciando threads de processamento..." << std::endl;
        
        // Thread 1: Lane detection (crítica)
        std::thread lane_thread(laneDetectionThread,
            std::ref(lane_detector),
            std::ref(camera),
            std::ref(mpc),
            std::ref(servo),
            std::ref(backMotors),
            std::ref(pid),
            std::ref(sync_results),
            zmq_publisher
        );
        
        // Thread 2: Object detection (não crítica)
        std::thread object_thread(objectDetectionThread,
            std::ref(object_detector),
            std::ref(camera),
            std::ref(sync_results)
        );
        
        // === MAIN LOOP - VISUALIZAÇÃO E MONITORING ===
        
        FPSCalculator fps_calculator(30);
        auto lastTime = std::chrono::steady_clock::now();
        double smoothedFPS = 0.0;
        const double alpha = 0.9;
        
        std::cout << "Sistema integrado iniciado com sucesso!" << std::endl;
        std::cout << "Lane Detection: Skip 2 frames (crítico)" << std::endl;
        std::cout << "Object Detection: Skip 8 frames (adaptativo)" << std::endl;
        std::cout << "Pressione 'q' para sair" << std::endl;
        
        while (keep_running.load()) {
            fps_calculator.update();
            
            // Calcular FPS
            auto currentTime = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
            lastTime = currentTime;
            double currentFPS = 1.0 / elapsed;
            smoothedFPS = smoothedFPS == 0.0 ? currentFPS : alpha * smoothedFPS + (1.0 - alpha) * currentFPS;
            
            // Obter dados mais recentes
            cv::Mat display_frame;
            double offset = 0.0, psi = 0.0;
            std::vector<Detection> current_detections;
            
            // Copiar dados de lanes (sempre prioritário)
            {
                std::lock_guard<std::mutex> lock(sync_results.lane_data.mutex);
                if (sync_results.lane_data.valid) {
                    display_frame = sync_results.lane_data.processed_frame.clone();
                    offset = sync_results.lane_data.offset;
                    psi = sync_results.lane_data.psi;
                }
            }
            
            // Copiar dados de objects (se disponíveis)
            {
                std::lock_guard<std::mutex> lock(sync_results.object_data.mutex);
                if (sync_results.object_data.valid) {
                    current_detections = sync_results.object_data.detections;
                }
            }
            
            if (!display_frame.empty()) {
                // Desenhar detecções de objetos sobre o frame de lanes
                for (const auto& det : current_detections) {
                    cv::Scalar color = object_detector.getColor(det.class_id);
                    cv::rectangle(display_frame, det.bbox, color, 2);
                    
                    std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 4);
                    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
                    
                    cv::rectangle(display_frame,
                        cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                        cv::Point(det.bbox.x + label_size.width, det.bbox.y),
                        color, -1);
                    
                    cv::putText(display_frame, label,
                        cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                }
                
                // HUD integrado
                double v_actual = current_speed_ms.load();
                
                // Informações básicas
                std::string fps_text = "FPS: " + std::to_string(smoothedFPS).substr(0, 4);
                cv::putText(display_frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                
                std::string speed_text = "Speed: " + std::to_string(v_actual).substr(0, 4) + " m/s";
                cv::putText(display_frame, speed_text, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
                
                std::string offset_text = "Offset: " + std::to_string(offset).substr(0, 5);
                cv::putText(display_frame, offset_text, cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2);
                
                std::string objects_text = "Objects: " + std::to_string(current_detections.size());
                cv::putText(display_frame, objects_text, cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 2);
                
                // Mostrar resultado integrado
                cv::imshow("Integrated Vision System", display_frame);
            }
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                keep_running.store(false);
                break;
            }
        }
        
        // === CLEANUP ===
        
        std::cout << "A terminar threads..." << std::endl;
        
        // Esperar que as threads terminem
        if (lane_thread.joinable()) lane_thread.join();
        if (object_thread.joinable()) object_thread.join();
        
        // Parar sistemas
        servo.set_steering(0);
        backMotors.setSpeed(0);
        camera.stop();
        cv::destroyAllWindows();
        canBusManager->stop();
        
        if (zmq_publisher) {
            delete zmq_publisher;
            std::cout << "ZMQ Publisher liberado." << std::endl;
        }
        
        std::cout << "Sistema terminado com sucesso." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}