#include "control_systems/utils_control.hpp"     // initLaneControl, drawHUD, initMotors, initServo, initCanBus, initZmq
#include "object_detection/fps.hpp"              // FPSCalculator
#include "object_detection/frame.hpp"            // FrameSkipper
#include "object_detection/inferObject.hpp"      // TensorRTYOLO (objetos)
#include <filesystem>

// Variáveis globais atómicas
std::atomic<bool> keep_running{true};
std::atomic<double> current_speed_ms{0.0};

// Estruturas para resultados
// Atualiza estruturas com timestamp
struct ObjectResults {
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
};

struct LaneResults {
    cv::Mat processed_frame;
    double offset = 0.0;
    double psi = 0.0;
    std::vector<cv::Point> medianPoints;
    LaneData laneData;
    LineIntersect intersect;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
};

// Queues e sincronização
std::queue<cv::Mat> frame_queue_objects;
std::queue<cv::Mat> frame_queue_lanes;
std::mutex mtx_objects, mtx_lanes, mtx_results;
std::condition_variable cv_objects, cv_lanes, cv_results;
ObjectResults latest_objects;
LaneResults latest_lanes;
bool results_ready = false;

// Signal handler
void signalHandler(int signum) {
    std::cout << "\nSinal de interrupção (" << signum << ") recebido." << std::endl;
    keep_running.store(false);
}

void objectInferenceThread(TensorRTYOLO& detector, FrameSkipper& frame_skipper, FPSCalculator& fps_calculator, ZmqPublisher* zmq_publisher) {
    std::vector<Detection> last_detections;
    double total_time = 0.0;
    int frame_counter = 0;

    while (keep_running.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_objects);
            cv_objects.wait(lock, [&] { return !frame_queue_objects.empty() || !keep_running.load(); });
            if (!keep_running.load()) break;
            frame = frame_queue_objects.front();
            frame_queue_objects.pop();
        }

        fps_calculator.update();

        std::vector<Detection> detections;
        if (frame_skipper.shouldProcessFrame()) {
            auto start = std::chrono::high_resolution_clock::now();
            detections = detector.infer(frame);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            frame_skipper.recordProcessingTime(time_ms / 1000.0);

            total_time += time_ms;
            frame_counter++;
            if (frame_counter >= 10) {
                double avg_time = total_time / frame_counter;
                //std::cout << "Tempo médio objetos (últimos 10 frames): " << avg_time << " ms" << std::endl;
                total_time = 0.0;
                frame_counter = 0;
            }

            // Publica apenas nomes se detetado
            if (!detections.empty() && zmq_publisher && zmq_publisher->isConnected()) {
                std::stringstream ss;
                for (size_t i = 0; i < detections.size(); ++i) {
                    ss << detections[i].class_name;
                    if (i < detections.size() - 1) ss << ", ";
                }
                zmq_publisher->publishMessage(ss.str());
            }
        }
        last_detections = detections;

        {
            std::unique_lock<std::mutex> lock(mtx_results);
            latest_objects.detections = detections;
            latest_objects.timestamp = std::chrono::steady_clock::now();
            results_ready = true;
            cv_results.notify_one();
        }
    }
}


void laneInferenceThread(TensorRTInference& trt, NMPCController& mpc, PID& pid, FServo& servo, BackMotors& backMotors,
                         SCurveProfile& steering_profile, MovingAverage& filter, double setpoint_velocity,
                         FrameSkipper& frame_skipper, ZmqPublisher* zmq_publisher) {
    auto lastTime = std::chrono::steady_clock::now();
    double last_delta = 0.0;
    auto pid_last_time = std::chrono::steady_clock::now();
    double total_time = 0.0;
    int frame_counter = 0;

    while (keep_running.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_lanes);
            cv_lanes.wait(lock, [&] { return !frame_queue_lanes.empty() || !keep_running.load(); });
            if (!keep_running.load()) break;
            frame = frame_queue_lanes.front();
            frame_queue_lanes.pop();
        }

        if (frame.empty()) {
            std::cout << "Frame vazio na thread de lanes – saltando." << std::endl;
            continue;
        }

        if (!frame_skipper.shouldProcessFrame()) {
            std::cout << "Skip frame em lanes." << std::endl;
            continue;
        }

        std::cout << "Processando novo frame em lanes (tamanho: " << frame.size() << ")." << std::endl;  // Novo: Log para depuração

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        LineIntersect intersect;
        cv::Mat result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        frame_skipper.recordProcessingTime(time_ms / 1000.0);

        total_time += time_ms;
        frame_counter++;
        if (frame_counter >= 10) {
            double avg_time = total_time / frame_counter;
            std::cout << "Tempo médio lanes (últimos 10 frames): " << avg_time << " ms" << std::endl;  // Descomentado
            total_time = 0.0;
            frame_counter = 0;
        }

        auto currentTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        double v_actual = current_speed_ms.load();
        auto pid_now = std::chrono::steady_clock::now();
        double pid_dt = std::chrono::duration<double>(pid_now - pid_last_time).count();
        if (pid_dt >= 0.02) {
            double motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
            backMotors.setSpeed(static_cast<int>(motor_pwm));  // Descomentado (se queres ativar motores)
            pid_last_time = pid_now;
        }

        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta;
        if (!std::isnan(offset) && !std::isnan(psi)) {
            delta = -mpc.computeControl(offset, psi, 0.7);
        }

        // Cálculo de lane e publicação
        int lane;
        lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
        if (zmq_publisher && zmq_publisher->isConnected()) {
            std::stringstream ss;
            ss << "lane:" << lane;
            zmq_publisher->publishMessage(ss.str());
        }

        double target_steering_angle = delta * 180.0 / M_PI;
        double smoothed_steering_angle = steering_profile.computeNext(target_steering_angle, elapsed);
        int steering_angle = static_cast<int>(smoothed_steering_angle);
        steering_angle = std::max(-40, std::min(40, steering_angle));
        servo.set_steering(steering_angle);
        last_delta = delta;

        {
            std::unique_lock<std::mutex> lock(mtx_results);
            latest_lanes.processed_frame = result;
            latest_lanes.offset = offset;
            latest_lanes.psi = psi;
            latest_lanes.medianPoints = medianPoints;
            latest_lanes.laneData = laneData;
            latest_lanes.intersect = intersect;
            latest_lanes.timestamp = std::chrono::steady_clock::now();
            results_ready = true;
            cv_results.notify_one();
            std::cout << "Lanes atualizado com sucesso." << std::endl;
        }
    }
}

cv::Mat combineAndDraw(const cv::Mat& original_frame, const ObjectResults& obj_res, const LaneResults& lane_res,
                       double smooth_fps, int processed_frames, int skipped_frames, bool use_objects, bool use_lanes) {
    cv::Mat combined;

    // Usa apenas o processed_frame de lanes se válido (sem blending)
    if (!lane_res.processed_frame.empty() && use_lanes) {
        combined = lane_res.processed_frame.clone();  // Apenas lanes, sem mesclar com original
        if (combined.size() != original_frame.size()) cv::resize(combined, combined, original_frame.size());
        std::cout << "Usando apenas lanes processado (sem blending)." << std::endl;  // Log para depuração
    } else {
        combined = original_frame.clone();
        std::cout << "Usando frame original (lanes inválido)." << std::endl;
    }

    if (combined.empty()) {
        std::cerr << "Combined vazio em combineAndDraw." << std::endl;
        return cv::Mat();
    }

    // Desenhar HUD e info diretamente no combined (agora só lanes)
    drawHUD(combined, smooth_fps, 0.0, current_speed_ms.load(), 0.0, lane_res.offset, lane_res.psi, 0, 0.0);

    std::string info_text = "FPS: " + std::to_string(smooth_fps).substr(0, 4) +
                            " | Proc: " + std::to_string(processed_frames) +
                            " | Skip: " + std::to_string(skipped_frames);
    cv::putText(combined, info_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    return combined;
}



// Cleanup
void cleanup(CSICamera& camera, FServo& servo, BackMotors& backMotors, std::unique_ptr<CanBusManager>& canBusManager, 
                std::thread& obj_thread, std::thread& lane_thread, ZmqPublisher* objects_publisher, ZmqPublisher* lanes_publisher) {
    std::cout << "Iniciando cleanup..." << std::endl;
    keep_running.store(false);

    cv_objects.notify_all();
    cv_lanes.notify_all();
    cv_results.notify_all();

    if (obj_thread.joinable()) obj_thread.join();
    if (lane_thread.joinable()) lane_thread.join();

    camera.stop();
    servo.set_steering(0);
    backMotors.setSpeed(0);
    if (canBusManager) canBusManager->stop();
    cv::destroyAllWindows();

    if (lanes_publisher) delete lanes_publisher;
    if (objects_publisher) delete objects_publisher;

    std::cout << "Cleanup completo." << std::endl;
}

int main() {
    std::signal(SIGINT, signalHandler);

    CSICamera camera(640, 480, 30);
    std::unique_ptr<TensorRTYOLO> obj_detector;
    std::unique_ptr<TensorRTInference> lane_trt;
    FPSCalculator fps_calculator(30);
    FrameSkipper obj_skipper(FrameSkipper::FIXED, 8, 15.0);
    FrameSkipper lane_skipper(FrameSkipper::FIXED, 0, 15.0);
    NMPCController mpc;
    PID pid;
    BackMotors backMotors;
    FServo servo;
    SCurveProfile steering_profile(100.0, 300.0, 600.0);
    MovingAverage filter(5);
    double setpoint_velocity = 1.0;
    std::shared_ptr<CANMessageProcessor> messageProcessor;
    std::unique_ptr<CanBusManager> canBusManager;
    std::thread obj_thread;
    std::thread lane_thread;

    // Cria contexto
    zmq::context_t zmq_context(1);

    // Inicializa com initZmq
    ZmqPublisher* lanes_publisher = initZmq(zmq_context, "127.0.0.1", 5558);  // Porta para lanes
    ZmqPublisher* objects_publisher = initZmq(zmq_context, "127.0.0.1", 5559);  // Porta para objects
    if (lanes_publisher == nullptr)std::cerr << "Falha ao inicializar ZMQ lanes - continuando sem." << std::endl;
    if (objects_publisher == nullptr)std::cerr << "Falha ao inicializar ZMQ object - continuando sem." << std::endl;

    try {
        std::cout << "Inicializando sistema integrado..." << std::endl;

        // Validação e loading de engines
        std::string obj_engine_path = "../engines/best.engine";
        std::ifstream check_obj(obj_engine_path);
        if (!check_obj.good()) {
            throw std::runtime_error("Engine de objetos não encontrado ou inacessível: " + obj_engine_path);
        }
        obj_detector = std::make_unique<TensorRTYOLO>(obj_engine_path, 640);
        std::cout << "Engine de objetos carregado: " << obj_engine_path << std::endl;

        std::string lane_engine_path = "../engines/model.engine";
        std::ifstream check_lane(lane_engine_path);
        if (!check_lane.good()) {
            throw std::runtime_error("Engine de lanes não encontrado ou inacessível: " + lane_engine_path);
        }
        lane_trt = std::make_unique<TensorRTInference>(lane_engine_path);
        std::cout << "Engine de lanes carregado: " << lane_engine_path << std::endl;

        if (!initMotors(backMotors)) throw std::runtime_error("Falha nos motores");
        if (!initServo(servo)) throw std::runtime_error("Falha no servo");

        // CAN Bus (com handler default das alterações acima)
        canBusManager = initCanBus(messageProcessor);
        if (!canBusManager) throw std::runtime_error("Falha no CAN Bus");

        camera.start();

        // Lançar threads
        obj_thread = std::thread(objectInferenceThread, std::ref(*obj_detector), std::ref(obj_skipper), std::ref(fps_calculator), objects_publisher);
        lane_thread = std::thread(laneInferenceThread, std::ref(*lane_trt), std::ref(mpc), std::ref(pid), std::ref(servo),
                          std::ref(backMotors), std::ref(steering_profile), std::ref(filter), setpoint_velocity,
                          std::ref(lane_skipper), lanes_publisher);


        std::cout << "Threads lançadas. Pressione 'q' para sair." << std::endl;

        // Novo: Delay para inicialização
        std::this_thread::sleep_for(std::chrono::seconds(1));

        int frame_count = 0;
        int processed_frames = 0;
        int skipped_frames = 0;
        while (keep_running.load()) {
            cv::Mat frame = camera.read();
            if (frame.empty()) {
                std::cout << "Frame da câmara vazio – continuando." << std::endl;
                continue;
            }
        
            frame_count++;
            // Envia para threads
            {
                std::unique_lock<std::mutex> lock_obj(mtx_objects);
                frame_queue_objects.push(frame.clone());
                cv_objects.notify_one();
            }
            {
                std::unique_lock<std::mutex> lock_lane(mtx_lanes);
                frame_queue_lanes.push(frame.clone());
                cv_lanes.notify_one();
                std::cout << "Frame enviado para lanes queue." << std::endl;  // Novo: Log para depuração
            }
            // Espera com timeout (200ms)
            {
                std::unique_lock<std::mutex> lock(mtx_results);
                if (cv_results.wait_for(lock, std::chrono::milliseconds(200), [&] { return results_ready; })) {
                    results_ready = false;
                } else {
                    std::cout << "Timeout esperando resultados – usando dados anteriores." << std::endl;
                }
            }
            // Verificação de timestamps (aumentado para 500ms)
            auto now = std::chrono::steady_clock::now();
            bool use_objects = std::chrono::duration<double, std::milli>(now - latest_objects.timestamp).count() < 500;
            bool use_lanes = std::chrono::duration<double, std::milli>(now - latest_lanes.timestamp).count() < 500;
        
            // Chamada corrigida com parâmetros extras
            double smooth_fps = fps_calculator.getSmoothFPS();
            cv::Mat displayed = combineAndDraw(frame, latest_objects, latest_lanes, smooth_fps, processed_frames, skipped_frames, use_objects, use_lanes);
        
            if (!displayed.empty()) {
                cv::imshow("Integrated Detection", displayed);
            } else {
                std::cout << "Displayed vazio – saltando imshow." << std::endl;
            }
        
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') keep_running.store(false);
        
            processed_frames++;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro de runtime (falha inicial): " << e.what() << std::endl;
        keep_running.store(false);
    } catch (const std::exception& e) {
        std::cerr << "Erro geral: " << e.what() << std::endl;
        keep_running.store(false);
    }

    // Cleanup
    cleanup(camera, servo, backMotors, canBusManager, obj_thread, lane_thread, objects_publisher, lanes_publisher);

    return 0;
}
