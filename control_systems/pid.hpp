#ifndef PID_HPP
#define PID_HPP

#include <algorithm>

class PID {
public:
    // === Parâmetros ===
    double Kp = 15.0; // 5.0
    double Ki = 1.5; // 0.5
    double Kd = 1.0; //0.1

    double outputMin = 0.0;     // PWM mínimo
    double outputMax = 30.0;   // PWM máximo
    double maxStepChange = 2.0; // passo máx do PWM por ciclo (%)

    // Anti-windup: limite absoluto da parcela integral (em "unidades de saída")
    // Ex.: 30 => a parte integral nunca contribui com mais que ±30% de PWM
    double Imax = 30.0;

    // Filtro da derivada (0<alpha<=1). 1.0 = sem filtro, 0.2 = filtragem forte
    double dAlpha = 0.3;

    PID();
    double compute(double setpoint, double measurement, double dt);
    void reset();

private:
    // Estado
    double iTerm;        // contribuição integral acumulada (em unidades de saída)
    double prevMeasFilt; // medição filtrada anterior (para derivada)
    double measFilt;     // medição filtrada corrente
    double lastOutput;   // saída limitada do passo anterior

    bool first = true;
};

#endif
