#include "pid.hpp"
#include <cmath>

PID::PID()
    : iTerm(0.0), prevMeasFilt(0.0), measFilt(0.0), lastOutput(0.0), first(true) {}

double PID::compute(double setpoint, double measurement, double dt) {
    if (dt <= 0.0) return lastOutput;

    // ---- Filtrar medição para derivada (1ª ordem) ----
    if (first) {
        measFilt = measurement;
        prevMeasFilt = measurement;
        first = false;
    } else {
        measFilt = measFilt + dAlpha * (measurement - measFilt);
    }

    // Erro (para P e I)
    double error = setpoint - measurement;

    // Derivada NA MEDIÇÃO (melhor anti-ruído e sem "derivative kick" no setpoint)
    double dMeas = (measFilt - prevMeasFilt) / dt;
    double dTerm = -Kd * dMeas; // sinal invertido porque é d(measurement)/dt
    prevMeasFilt = measFilt;

    // Proporcional
    double pTerm = Kp * error;

    // Integral "candidata": só aplicamos ao iTerm se as condições permitirem
    double iCandidate = iTerm + Ki * error * dt;

    // Saída "não saturada" usando a integral candidata
    double outUnsat = pTerm + iCandidate + dTerm;

    // ---- Anti-windup por integração condicional ----
    // Regras:
    // 1) Se outUnsat estiver dentro dos limites -> podemos aceitar iCandidate.
    // 2) Se estiver saturado, só integramos se isso ajudar a sair da saturação.
    bool within = (outUnsat >= outputMin && outUnsat <= outputMax);
    bool helpsUpper = (outUnsat > outputMax) && (error < 0.0); // erro negativo puxa para baixo
    bool helpsLower = (outUnsat < outputMin) && (error > 0.0); // erro positivo puxa para cima

    if (within || helpsUpper || helpsLower) {
        iTerm = iCandidate;
        // Limitar contribuição integral
        iTerm = std::clamp(iTerm, -Imax, Imax);
    }
    // Recalcular com iTerm final
    double rawOutput = pTerm + iTerm + dTerm;

    // Saturação da saída
    if (rawOutput > outputMax) rawOutput = outputMax;
    else if (rawOutput < outputMin) rawOutput = outputMin;

    // Limitar variação por ciclo (suavidade extra)
    double limitedOutput = std::clamp(
        rawOutput,
        lastOutput - maxStepChange,
        lastOutput + maxStepChange
    );

    lastOutput = limitedOutput;
    return limitedOutput;
}

void PID::reset() {
    iTerm = 0.0;
    lastOutput = 0.0;
    first = true;
}
