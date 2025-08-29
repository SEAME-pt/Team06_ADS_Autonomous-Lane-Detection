#include "control_systems/utils_control.hpp"     // initLaneControl, drawHUD, initMotors, initServo, initCanBus, initZmq
#include "object_detection/fps.hpp"              // FPSCalculator
#include "object_detection/frame.hpp"            // FrameSkipper
#include "object_detection/inferObject.hpp"      // TensorRTYOLO (objetos)
#include <filesystem>
#include <thread>
#include <chrono>


// Variáveis globais atómicas
std::atomic<bool> keep_running{true};
std::atomic<double> current_speed_ms{0.0};
std::atomic<double> current_speed{0.0};

// Estruturas para resultados
struct ObjectResults {
    std::vector<Detection> detections;
};

struct LaneResults {
    cv::Mat processed_frame;
    double offset = 0.0;
    double psi = 0.0;
    double delta_rad = 0.0; // delta em radianos (ou converte para graus)
    int steering_angle_deg = 0;
    std::vector<cv::Point> medianPoints;
    LaneData laneData;
    LineIntersect intersect;
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
    std::cout << "/nSinal de interrupção (" << signum << ") recebido." << std::endl;
    keep_running.store(false);
}

// Thread de objetos
void objectInferenceThread(TensorRTYOLO& detector, FrameSkipper& frame_skipper, FPSCalculator& fps_calculator, ZmqPublisher* lane_pub) {
    std::vector<Detection> last_detections;
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
            double time = std::chrono::duration<double>(end - start).count();
            frame_skipper.recordProcessingTime(time);
            last_detections = detections;
        } else {
            detections = last_detections;
        }
        if (!detections.empty()) {
/*             std::cout << "Objetos detetados (" << detections.size() << "): ";
            for (const auto& det : detections) {
                std::cout << det.class_name << " (" << det.confidence << ") ";
            }
            std::cout << std::endl; */

            // Publicar nomes dos objetos via ZMQ na porta 5558
            if (lane_pub && lane_pub->isConnected()) {
                for (const auto& det : detections) {
                    std::stringstream ss;
                    ss << det.class_name;
                    lane_pub->publishMessage(ss.str());
                }
            }
        }
        {
            std::unique_lock<std::mutex> lock(mtx_results);
            latest_objects.detections = detections;
            results_ready = true;
            cv_results.notify_one();
        }
    }
}

// Thread de lanes
void laneInferenceThread(TensorRTInference& trt, NMPCController& mpc, PID& pid, /* FServo& servo, BackMotors& backMotors, */
                         SCurveProfile& steering_profile, MovingAverage& filter, double setpoint_velocity,
                         FrameSkipper& frame_skipper, ZmqPublisher* lane_pub, ZmqPublisher* ctrl_pub) {
    auto lastTime = std::chrono::steady_clock::now();
    double last_delta = 0.0;
    auto pid_last_time = std::chrono::steady_clock::now();

    while (keep_running.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_lanes);
            cv_lanes.wait(lock, [&] { return !frame_queue_lanes.empty() || !keep_running.load(); });
            if (!keep_running.load()) break;
            frame = frame_queue_lanes.front();
            frame_queue_lanes.pop();
        }

        if (!frame_skipper.shouldProcessFrame()) continue;

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        LineIntersect intersect;
        cv::Mat result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        frame_skipper.recordProcessingTime(time);

        auto currentTime = std::chrono::steady_clock::now();
        //double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        double v_actual = current_speed_ms.load();
        std::cout << "Speed now: " << v_actual << " m/s" << std::endl;
        auto pid_now = std::chrono::steady_clock::now();
        double pid_dt = std::chrono::duration<double>(pid_now - pid_last_time).count();
        double motor_pwm = 0.0;
        if (pid_dt >= 0.02) { // 50 ms → 20 Hz
            motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
            //backMotors.setSpeed(static_cast<int>(motor_pwm));
            pid_last_time = pid_now;
            //std::cout << "Motor Signal: " << motor_pwm << " PWM" << std::endl;
        }

        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta;
        if (!std::isnan(offset) && !std::isnan(psi)) {
            delta = -mpc.computeControl(offset, psi, v_actual);
        }

        double target_steering_angle = delta * 180.0 / M_PI;
        double smoothed_steering_angle =target_steering_angle;
        //double smoothed_steering_angle = steering_profile.computeNext(target_steering_angle, elapsed);
        int steering_angle = static_cast<int>(smoothed_steering_angle);
        steering_angle = std::max(-40, std::min(40, steering_angle));
        //servo.set_steering(steering_angle);
        last_delta = delta;
        // Publicar via ZMQ
        int lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
        if (lane_pub && lane_pub->isConnected()) {
            std::stringstream ss;
            ss << "lane:" << lane;
            lane_pub->publishMessage(ss.str());
        }
        if (ctrl_pub && ctrl_pub->isConnected()) {
            std::stringstream ss2;
            ss2 << "throttle:" << motor_pwm << ";steering:" << steering_angle << ";";
            ctrl_pub->publishMessage(ss2.str());
        }
        {
            std::unique_lock<std::mutex> lock(mtx_results);
            latest_lanes.processed_frame = result;
            latest_lanes.offset = offset;
            latest_lanes.psi = psi;
            latest_lanes.medianPoints = medianPoints;
            latest_lanes.laneData = laneData;
            latest_lanes.intersect = intersect;
            latest_lanes.delta_rad = delta;
            latest_lanes.steering_angle_deg = steering_angle;
            results_ready = true;
            cv_results.notify_one();
        }
    }
}

// Função combineAndDraw (como antes)
cv::Mat combineAndDraw(const cv::Mat& original_frame, const ObjectResults& obj_res, const LaneResults& lane_res,
                       double smooth_fps, int processed_frames, int skipped_frames) {
    if (original_frame.empty()) {
        std::cerr << "Frame original vazio em combineAndDraw." << std::endl;
        return cv::Mat();  // Retorna empty explicitamente
    }

    cv::Mat combined = original_frame.clone();

    // Desenhar objetos
    for (const auto& det : obj_res.detections) {
        // Validate bounding box is within image bounds
        if (det.bbox.x >= 0 && det.bbox.y >= 0 &&
            det.bbox.x + det.bbox.width <= combined.cols &&
            det.bbox.y + det.bbox.height <= combined.rows) {
            cv::Scalar color = cv::Scalar(255, 0, 0);
            cv::rectangle(combined, det.bbox, color, 2);
            std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 4);
            // Ensure text position is within bounds
            int text_y = std::max(15, det.bbox.y - 5);
            cv::putText(combined, label, cv::Point(det.bbox.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }

    // Desenhar lanes (verifica se processed_frame não é empty)
    if (!lane_res.processed_frame.empty()) {
        combined = lane_res.processed_frame.clone();  // Novo: Check para evitar cópia de empty
    } else {
        std::cout << "Processed frame de lanes vazio – usando original." << std::endl;
    }

    drawHUD(combined, smooth_fps, lane_res.delta_rad, current_speed_ms.load(), 0.0, lane_res.offset, lane_res.psi, lane_res.steering_angle_deg);

    std::string info_text = "FPS: " + std::to_string(smooth_fps).substr(0, 4) +
                            " | Proc: " + std::to_string(processed_frames) +
                            " | Skip: " + std::to_string(skipped_frames);
    cv::putText(combined, info_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    return combined;
}

// Cleanup
void cleanup(CSICamera& camera,/* FServo& servo, BackMotors& backMotors, std::unique_ptr<CanBusManager>& canBusManager, */
             std::thread& obj_thread, std::thread& lane_thread, ZmqPublisher* lane_pub, ZmqPublisher* ctrl_pub,
             ZmqPublisher* obj_pub, ZmqSubscriber* speed_sub) {
    std::cout << "Iniciando cleanup..." << std::endl;

    // Signal all threads to stop
    keep_running.store(false);

    // Wake up all waiting threads
    cv_objects.notify_all();
    cv_lanes.notify_all();
    cv_results.notify_all();

    // Give threads time to see the stop signal
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Join threads with timeout protection
    if (obj_thread.joinable()) {
        obj_thread.join();
        std::cout << "Object thread joined" << std::endl;
    }
    if (lane_thread.joinable()) {
        lane_thread.join();
        std::cout << "Lane thread joined" << std::endl;
    }

    camera.stop();
    //servo.set_steering(0);
    //backMotors.setSpeed(0);
    //if (canBusManager) canBusManager->stop();
    cv::destroyAllWindows();
    if (lane_pub) delete lane_pub;
    if (ctrl_pub) delete ctrl_pub;
    if (obj_pub) delete obj_pub;
    if (speed_sub) {
        speed_sub->stop();
        delete speed_sub;
    }
    std::cout << "Cleanup completo." << std::endl;
}

int main() {
    std::signal(SIGINT, signalHandler);

    // Declarações fora do try
    CSICamera camera(640, 480, 15);
    std::unique_ptr<TensorRTYOLO> obj_detector;
    std::unique_ptr<TensorRTInference> lane_trt;
    FPSCalculator fps_calculator(30);
    FrameSkipper obj_skipper(FrameSkipper::FIXED, 20, 15.0);
    FrameSkipper lane_skipper(FrameSkipper::FIXED, 8, 15.0);
    NMPCController mpc;
    PID pid;
    BackMotors backMotors;
    //FServo servo;
    SCurveProfile steering_profile(100.0, 300.0, 600.0);
    MovingAverage filter(5);
    double setpoint_velocity = 0.4;
    //std::shared_ptr<CANMessageProcessor> messageProcessor;
    //std::unique_ptr<CanBusManager> canBusManager;
    std::thread obj_thread;
    std::thread lane_thread;
    zmq::context_t zmq_context(1);
    ZmqPublisher* lane_pub = nullptr; // Para porta 5558 (object)
    ZmqPublisher* ctrl_pub = nullptr; // Para porta 5560 (throttle e steering)
    ZmqPublisher* obj_pub = nullptr; // Para porta 5559 (object)
    ZmqSubscriber* speed_sub = nullptr; // Novo: Para porta 5555 (speed)

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

        //if (!initMotors(backMotors)) throw std::runtime_error("Falha nos motores");
        //if (!initServo(servo)) throw std::runtime_error("Falha no servo");

        //CAN Bus (com handler default das alterações acima)
        //canBusManager = initCanBus(messageProcessor);
        //if (!canBusManager) throw std::runtime_error("Falha no CAN Bus");

        // Inicializar ZMQ publishers
        lane_pub = new ZmqPublisher(zmq_context, "127.0.0.1", 5558, "tcp");
        if (!lane_pub->isConnected()) throw std::runtime_error("Falha ao inicializar ZMQ na porta 5558");

        ctrl_pub = new ZmqPublisher(zmq_context, "127.0.0.1", 5560, "tcp");
        if (!ctrl_pub->isConnected()) throw std::runtime_error("Falha ao inicializar ZMQ na porta 5560");

        obj_pub = new ZmqPublisher(zmq_context, "127.0.0.1", 5559, "tcp");
        if (!obj_pub->isConnected()) throw std::runtime_error("Falha ao inicializar ZMQ na porta 5559");

        // Novo: Inicializar ZMQ Subscriber para velocidade na porta 5555
        speed_sub = new ZmqSubscriber(zmq_context, "127.0.0.1", 5555, current_speed_ms);
        if (!speed_sub->isConnected()) {
            throw std::runtime_error("Falha ao inicializar ZMQ Subscriber na porta 5555");
        }
        speed_sub->start();  // Inicia o thread de receção

        camera.start();

        // Lançar threads (usa obj_detector.get() para referência)
        obj_thread = std::thread(objectInferenceThread, std::ref(*obj_detector), std::ref(obj_skipper), std::ref(fps_calculator), obj_pub);
        lane_thread = std::thread(laneInferenceThread, std::ref(*lane_trt), std::ref(mpc), std::ref(pid), /* std::ref(servo), */
                                 /* std::ref(backMotors),  */std::ref(steering_profile), std::ref(filter), setpoint_velocity,
                                 std::ref(lane_skipper), lane_pub, ctrl_pub);

        std::cout << "Threads lançadas. Pressione 'q' para sair." << std::endl;

        //Loop principal
        int frame_count = 0;
        int processed_frames = 0;
        int skipped_frames = 0;

        // Wait for camera to be ready
        std::cout << "Waiting for camera to initialize..." << std::endl;
        cv::Mat test_frame;
        while (keep_running.load() && test_frame.empty()) {
            test_frame = camera.read();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "Camera ready!" << std::endl;

        while (keep_running.load()) {
            cv::Mat frame = camera.read();
            if (frame.empty()) {
                std::cout << "Warning: Empty frame received" << std::endl;
                continue;
            }
            frame_count++;

            //Enviar para threads
            {
                std::unique_lock lock_obj(mtx_objects);
                frame_queue_objects.push(frame.clone());
                cv_objects.notify_one();
            }

            {
                std::unique_lock lock_lane(mtx_lanes);
                frame_queue_lanes.push(frame.clone());
                cv_lanes.notify_one();
            }

            //Esperar resultados
            {
                std::unique_lock lock(mtx_results);
                cv_results.wait(lock, [&] { return results_ready || !keep_running.load(); });
                if (!keep_running.load()) break;
                results_ready = false;
            }

            //Combinar e exibir
            double smooth_fps = fps_calculator.getSmoothFPS();
            cv::Mat displayed = combineAndDraw(frame, latest_objects, latest_lanes, smooth_fps, processed_frames, skipped_frames);
            if (!displayed.empty()) cv::imshow("Integrated Detection", displayed);
            cv::imshow("Integrated Detection", displayed);
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

    cleanup(camera,/* servo, backMotors,  canBusManager, */ obj_thread, lane_thread, lane_pub, ctrl_pub, obj_pub, speed_sub);
    return 0;
}
