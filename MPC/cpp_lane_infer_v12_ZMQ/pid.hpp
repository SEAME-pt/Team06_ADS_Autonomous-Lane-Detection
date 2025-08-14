#ifndef PID_HPP
#define PID_HPP

class PID {
public:
    PID(double kp, double ki, double kd, double outMin, double outMax)
        : Kp(kp), Ki(ki), Kd(kd), outputMin(outMin), outputMax(outMax),
          integral(0.0), prevError(0.0) {}

    double compute(double setpoint, double measurement, double dt) {
        double error = setpoint - measurement;

        // Integral com anti-windup
        integral += error * dt;
        if (integral > outputMax) integral = outputMax;
        else if (integral < outputMin) integral = outputMin;

        // Derivada
        double derivative = (error - prevError) / dt;

        // Saída PID
        double output = Kp * error + Ki * integral + Kd * derivative;

        // Limitar a saída
        if (output > outputMax) output = outputMax;
        else if (output < outputMin) output = outputMin;

        prevError = error;
        return output;
    }

    void reset() {
        integral = 0.0;
        prevError = 0.0;
    }

private:
    double Kp, Ki, Kd;
    double outputMin, outputMax;
    double integral;
    double prevError;
};

#endif
