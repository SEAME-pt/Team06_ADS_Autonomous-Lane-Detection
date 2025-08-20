#include "utils_control.hpp"

// Variável atómica para armazenar a velocidade atual, acessível de forma segura entre threads
std::atomic<double> current_speed_ms{0.0};
// Flag atómica para controlar a execução do loop principal
std::atomic<bool> keep_running{true};

// Handler de sinal para terminar o programa de forma limpa com Ctrl+C
void signalHandler(int signum) {
    std::cout << "\nSinal de interrupcao (" << signum << ") recebido. A terminar a aplicacao..." << std::endl;
    keep_running.store(false);
}

int main() {
    std::signal(SIGINT, signalHandler);

// --- INICIALIZAÇÃO DA LÓGICA DE CONTROLO DE FAIXAS ---
    auto laneControl = initLaneControl();
    if (!laneControl) return 1;
    auto& trt = laneControl->trt;
    auto& cam = laneControl->cam;
    
    NMPCController mpc;

// --- INICIALIZAÇÃO DO CONTROLO DE VELOCIDADE E MOTORES ---
    BackMotors backMotors;
    if (!initMotors(backMotors)) return 1;

    FServo servo;
    if (!initServo(servo)) return -1;

//#### Inicialização do sistema CAN Bus--------------
    std::shared_ptr<CANMessageProcessor> messageProcessor;
    auto canBusManager = initCanBus(messageProcessor);
    if (!canBusManager) return 1;
//---------------------------------------------------

// --- INICIALIZAÇÃO DO ZMQ ---------------
    zmq::context_t context(1);
    ZmqPublisher* zmq_publisher = initZmq(context);
//---------------------------------------------

    std::cout << "Pressione Ctrl+C para sair" << std::endl;

    auto lastTime = std::chrono::steady_clock::now();
    double smoothedFPS = 0.0;
    const double alpha = 0.9;
    double last_smoothed_angle = 0.0; // Para filtro low-pass
    double last_delta = 0.0;
    int frameCount = 0;
    double setpoint_velocity = 1.0; // m/s desejados

/*PID INICIALIZATION*/
    PID pid;
    auto pid_last_time = std::chrono::steady_clock::now();
    double motor_pwm = 0.0;
/********************/
    
/*S-CURVE INICIALIZATION*/
    double max_steering_vel_deg_s = 100.0;  // Velocidade máx: 30°/s (lento)
    double max_steering_acc_deg_s2 = 300.0; // Aceleração máx: 100°/s²
    double max_steering_jerk_deg_s3 = 600.0; // Jerk máx: 400°/s³
    SCurveProfile steering_profile(max_steering_vel_deg_s, max_steering_acc_deg_s2, max_steering_jerk_deg_s3);
/***************************/

    MovingAverage filter(5);      // média móvel de 5 amostras
    //SpeedFilter filter(0.1);       // 20% valor novo, 80% suavização

    while (keep_running.load()) {
        cv::Mat frame = cam.read();
        if (frame.empty()) continue;

        auto currentTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        double currentFPS = 1.0 / elapsed;
        smoothedFPS = smoothedFPS == 0.0 ? currentFPS : alpha * smoothedFPS + (1.0 - alpha) * currentFPS;

        std::vector<float> input = preprocess_frame(frame);
        auto outputs = trt.infer(input);
        std::vector<cv::Point> medianPoints;
        LaneData laneData;
        LineIntersect intersect;
        auto result = postprocess(outputs.data(), frame, medianPoints, laneData, intersect);

        double v_actual = current_speed_ms.load();
        std::cout << "Speed now: " << v_actual << " m/s" << std::endl;
        
        auto pid_now = std::chrono::steady_clock::now();
        double pid_dt = std::chrono::duration<double>(pid_now - pid_last_time).count();
        if (pid_dt >= 0.02) { // 50 ms → 20 Hz
            motor_pwm = pid.compute(setpoint_velocity, v_actual, pid_dt);
            backMotors.setSpeed(static_cast<int>(motor_pwm));
            pid_last_time = pid_now;
            //std::cout << "Motor Signal: " << motor_pwm << " PWM" << std::endl;
        }
        
        //double v_filtered = filter.update(v_actual); // velocidade suavizada
        //std::cout << "Velocidade filtrada: " << v_filtered << std::endl;

        double offset = intersect.offset;
        double psi = intersect.psi;
        double delta = last_delta;
        if (!std::isnan(offset) && !std::isnan(psi)) {
            delta = -mpc.computeControl(offset, psi, 0.7);
        }

        // Converte delta para graus (ângulo desejado)
        double target_steering_angle = delta * 180.0 / M_PI;
        double smoothed_steering_angle =target_steering_angle;
        // Aplica perfil S-Curve
        //double smoothed_steering_angle = steering_profile.computeNext(target_steering_angle, elapsed);

/*         // Aplica filtro low-pass leve para suavizar entre ciclos (opcional)
        const double alpha_lowpass = 0.8; // 0.8 = equilíbrio entre suavidade e reatividade
        smoothed_steering_angle = alpha_lowpass * smoothed_steering_angle + (1.0 - alpha_lowpass) * last_smoothed_angle;
        last_smoothed_angle = smoothed_steering_angle; */

        // Limita o ângulo final
        int steering_angle = static_cast<int>(smoothed_steering_angle);
        steering_angle = std::max(-40, std::min(40, steering_angle));
        servo.set_steering(steering_angle);
        last_delta = delta; // Atualiza last_delta

        int lane;
        lane = (offset < -0.01) ? 2 : ((offset > 0.02) ? 1 : 0);
        if (zmq_publisher && zmq_publisher->isConnected()) {
            std::stringstream ss;
            ss << "lane:" << lane;
            zmq_publisher->publishMessage(ss.str());
        }

        drawHUD(result, smoothedFPS, delta, v_actual,motor_pwm, offset, 
            psi, steering_angle, steering_angle);

        frameCount++;
        cv::imshow("Lane Detection", result);

        if (cv::waitKey(1) == 'q') {
            keep_running.store(false);
        }
    }

    servo.set_steering(0);
    backMotors.setSpeed(0);
    cam.stop();
    cv::destroyAllWindows();
    canBusManager->stop();
    if (zmq_publisher) {
        delete zmq_publisher;
        std::cout << "ZMQ Publisher liberado." << std::endl;
    }
    return 0;
}