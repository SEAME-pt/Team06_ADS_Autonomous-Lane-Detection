#include "control_systems/utils_control.hpp"     // initLaneControl, drawHUD, initMotors, initServo, initCanBus, initZmq
#include "object_detection/fps.hpp"              // FPSCalculator
#include "object_detection/frame.hpp"            // FrameSkipper
#include "object_detection/inferObject.hpp"      // TensorRTYOLO (objetos)

#include <atomic>
#include <mutex>
#include <thread>
#include <sstream>
#include <csignal>

// ===================== Estado global =====================
std::atomic<double> current_speed_ms{0.0};
std::atomic<bool>   keep_running{true};

// Sinalização de erro controlada (evita abort/core dump)
std::atomic<bool> fatal_error{false};
std::string error_msg;
std::mutex err_mtx;

static inline void report_fatal(const std::string& msg) {
    {
        std::lock_guard<std::mutex> lk(err_mtx);
        error_msg = msg;
    }
    fatal_error.store(true);
    keep_running.store(false);
}

void signalHandler(int signum) {
    std::cout << "\nSinal de interrupcao (" << signum << ") recebido. A terminar..." << std::endl;
    keep_running.store(false);
}

// ===================== Estruturas de sincronização =====================
struct SyncResults {
    struct {
        double offset{0.0};
        double psi{0.0};
        bool valid{false};
        cv::Mat processed_frame;
        std::mutex mtx;
    } lane;

    struct {
        std::vector<Detection> detections;
        bool valid{false};
        std::mutex mtx;
    } objects;
};

// ===================== Threads de processamento =====================
// Lanes (prioritária, skip=2)
static void laneDetectionThread(
    TensorRTInference& lane_trt,
    CSICamera& camera,
    NMPCController& mpc,
    FServo& servo,
    BackMotors& backMotors,
    PID& pid,
    SyncResults& sync,
    ZmqPublisher* zmq_pub
) {
    try {
        FrameSkipper skipper(FrameSkipper::FIXED, 2, 30.0);  // processa 1 a cada (2+1)=3 frames
        auto pid_last = std::chrono::steady_clock::now();
        double last_delta = 0.0;
        const double setpoint_velocity = 1.0; // m/s alvo

        while (keep_running.load()) {
            cv::Mat frame = camera.read();
            if (frame.empty()) continue;

            if (!skipper.shouldProcessFrame()) continue;

            // Inferência lanes
            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> input = preprocess_frame(frame);
            auto outputs = lane_trt.infer(input);

            std::vector<cv::Point> medianPoints;
            LaneData laneData;
            LineIntersect isect;
            cv::Mat result = postprocess(outputs.data(), frame, medianPoints, laneData, isect);
            const auto t1 = std::chrono::high_resolution_clock::now();
            skipper.recordProcessingTime(std::chrono::duration<double>(t1 - t0).count());

            // Controlo de direção (NMPC) e servo
            double offset = isect.offset;
            double psi = isect.psi;
            if (!std::isnan(offset) && !std::isnan(psi)) {
                double delta = -mpc.computeControl(offset, psi, 0.7);
                int steering_angle = static_cast<int>(delta * 180.0 / M_PI);
                steering_angle = std::max(-40, std::min(40, steering_angle));
                servo.set_steering(steering_angle);
                last_delta = delta;

                // Publicação ZMQ (opcional)
                if (zmq_pub && zmq_pub->isConnected()) {
                    int lane_side = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
                    std::stringstream ss; ss << "lane:" << lane_side;
                    zmq_pub->publishMessage(ss.str());
                }
            }

            // Controlo de velocidade (PID) a 20 Hz
            const auto pid_now = std::chrono::steady_clock::now();
            double pid_dt = std::chrono::duration<double>(pid_now - pid_last).count();
            if (pid_dt >= 0.02) {
                double v_actual = current_speed_ms.load();
                double motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
                //backMotors.setSpeed(static_cast<int>(motor_pwm));
                pid_last = pid_now;
            }

            // Guardar resultados para visualização
            {
                std::lock_guard<std::mutex> lk(sync.lane.mtx);
                sync.lane.offset = offset;
                sync.lane.psi = psi;
                sync.lane.valid = true;
                sync.lane.processed_frame = result;
            }
        }
    } catch (const std::exception& e) {
        report_fatal(std::string("Lane thread error: ") + e.what());
    } catch (...) {
        report_fatal("Lane thread unknown error");
    }
}

// Objetos (adaptativo, skip agressivo)
static void objectDetectionThread(TensorRTYOLO& yolo, CSICamera& camera, SyncResults& sync) {
    try {
        FrameSkipper skipper(FrameSkipper::ADAPTIVE, 8, 15.0);

        while (keep_running.load()) {
            cv::Mat frame = camera.read();
            if (frame.empty()) continue;
            if (!skipper.shouldProcessFrame()) continue;

            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<Detection> dets = yolo.infer(frame);
            const auto t1 = std::chrono::high_resolution_clock::now();
            skipper.recordProcessingTime(std::chrono::duration<double>(t1 - t0).count());

            {
                std::lock_guard<std::mutex> lk(sync.objects.mtx);
                sync.objects.detections = std::move(dets);
                sync.objects.valid = true;
            }
        }
    } catch (const std::exception& e) {
        report_fatal(std::string("Object thread error: ") + e.what());
    } catch (...) {
        report_fatal("Object thread unknown error");
    }
}

// ===================== main =====================
int main() {
    std::signal(SIGINT, signalHandler);

    // Objetos com duração até ao final para permitir cleanup
    std::unique_ptr<LaneControl> laneControl;
    NMPCController mpc;
    BackMotors backMotors;
    FServo servo;
    std::shared_ptr<CANMessageProcessor> canMsgProcessor;
    std::unique_ptr<CanBusManager> canBusManager;
    std::unique_ptr<ZmqPublisher> zmq_publisher_ptr; // se initZmq devolver raw*, vamos embrulhar
    ZmqPublisher* zmq_publisher = nullptr;

    try {
        // 1) Inicializações lane (câmara + engine lanes)
        laneControl = initLaneControl();         // internamente abre a engine e start() na camera
        auto& lane_trt = laneControl->trt;
        auto& camera  = laneControl->cam;
        backMotors.setSpeed(0);

        // 2) Inicialização objetos (engine YOLO)
        const std::string obj_engine = "best.engine";
        TensorRTYOLO yolo(obj_engine, 640);

        // 3) Motores e servo
        if (!initMotors(backMotors)) throw std::runtime_error("initMotors() failed");
        if (!initServo(servo))       throw std::runtime_error("initServo() failed");

        // 4) CAN Bus e ZMQ
        canBusManager = initCanBus(canMsgProcessor);
        if (!canBusManager) throw std::runtime_error("initCanBus() failed");
        {
            zmq::context_t ctx(1);
            ZmqPublisher* raw = initZmq(ctx);
            if (!raw) throw std::runtime_error("initZmq() failed");
            // Embrulhar para garantir delete em cleanup
            zmq_publisher_ptr.reset(raw);
            zmq_publisher = zmq_publisher_ptr.get();
        }

        // 5) PID
        PID pid;

        // 6) Sincronização e threads
        SyncResults sync;
        std::thread th_lane, th_obj;

        th_lane = std::thread(laneDetectionThread,
                              std::ref(lane_trt),
                              std::ref(camera),
                              std::ref(mpc),
                              std::ref(servo),
                              std::ref(backMotors),
                              std::ref(pid),
                              std::ref(sync),
                              zmq_publisher);

        th_obj  = std::thread(objectDetectionThread,
                              std::ref(yolo),
                              std::ref(camera),
                              std::ref(sync));

        // 7) Visualização
        FPSCalculator fps(30);
        auto t_last = std::chrono::steady_clock::now();
        double smoothFPS = 0.0;
        const double alpha = 0.9;

        std::cout << "Sistema integrado iniciado. [Lane skip=2] [Object ADAPTIVE skip~8]\nPressione 'q' para sair.\n";

        while (keep_running.load()) {
            if (fatal_error.load()) break;

            fps.update();

            auto t_now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_now - t_last).count();
            t_last = t_now;
            double curFPS = (dt > 1e-6) ? 1.0 / dt : 0.0;
            smoothFPS = (smoothFPS == 0.0) ? curFPS : alpha * smoothFPS + (1.0 - alpha) * curFPS;

            cv::Mat frame;
            double offset = 0.0, psi = 0.0;
            std::vector<Detection> dets;

            {
                std::lock_guard<std::mutex> lk(sync.lane.mtx);
                if (sync.lane.valid) {
                    frame = sync.lane.processed_frame.clone();
                    offset = sync.lane.offset;
                    psi = sync.lane.psi;
                }
            }
            {
                std::lock_guard<std::mutex> lk(sync.objects.mtx);
                if (sync.objects.valid) dets = sync.objects.detections;
            }

            if (!frame.empty()) {
                for (const auto& d : dets) {
                    cv::Scalar color = cv::Scalar(255, 255, 255); // fallback
                    // getColor pode lançar? Em princípio não.
                    // Se preferir, proteja com try/catch aqui.
                    // Mantemos simples:
                    // Se tiver getColor no YOLO acessível aqui, teria de passar referência.
                    // Para manter isolado, refazemos as cores via det.class_id (simples).
                    // (ou faça YOLO.getColor via referência, o que exigiria partilhar yolo aqui)
                    cv::rectangle(frame, d.bbox, color, 2);
                    std::string label = d.class_name + ": " + std::to_string(d.confidence).substr(0, 4);
                    cv::putText(frame, label, cv::Point(d.bbox.x, d.bbox.y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
                }

                std::string info = "FPS: " + std::to_string(smoothFPS).substr(0,4);
                cv::putText(frame, info, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);

                cv::imshow("Integrated Vision System", frame);
            }

            char k = cv::waitKey(1) & 0xFF;
            if (k == 'q') {
                keep_running.store(false);
                break;
            }
        }

        // 8) Encerramento limpo
        if (th_lane.joinable()) th_lane.join();
        if (th_obj.joinable())  th_obj.join();

        // Paragens seguras
        try { servo.set_steering(0); } catch (...) {}
        try { backMotors.setSpeed(0); } catch (...) {}
        try { laneControl->cam.stop(); } catch (...) {}
        try { cv::destroyAllWindows(); } catch (...) {}
        try { if (canBusManager) canBusManager->stop(); } catch (...) {}

        if (fatal_error.load()) {
            std::lock_guard<std::mutex> lk(err_mtx);
            std::cerr << "Fatal error: " << error_msg << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Erro fatal: " << e.what() << std::endl;
        keep_running.store(false);
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Erro fatal: excecao desconhecida" << std::endl;
        keep_running.store(false);
        return EXIT_FAILURE;
    }
}
