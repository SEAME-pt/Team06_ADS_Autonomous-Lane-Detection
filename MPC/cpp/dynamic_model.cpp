#include <cmath>
#include <vector>
#include <iostream>

class VehicleModel {
public:
    // Parâmetros do veículo
    const float v = 3.0f;      // Velocidade: 3 m/s
    const float L = 0.15f;     // Distância entre eixos: 0,15 m
    const float dt = 0.05f;    // Passo de tempo: 0,05 s (20 Hz)
    const float mtsPixel = 0.001f; // Conversão: 0,001 m/pixel (estimativa)

    // Estado: [y, ψ]
    float y;    // Posição lateral (metros)
    float psi;  // Orientação (radianos)

    VehicleModel() : y(0.0f), psi(0.0f) {}

    // Atualizar estado com base no ângulo de direção δ
    void update(float delta) {
        y += dt * v * std::sin(psi);           // y[k+1] = y[k] + Δt * v * sin(ψ[k])
        psi += dt * (v / L) * std::tan(delta); // ψ[k+1] = ψ[k] + Δt * (v / L) * tan(δ[k])
    }

    // Converter deslocamento de pixels para metros
    float convertShift(float shift_pixels) {
        return shift_pixels * mtsPixel;
    }
};

// Exemplo de uso
int main() {
    VehicleModel jetson;

    // Simular alguns passos
    float shift_pixels = -10.0f; // Exemplo: -10 pixels (do teste anterior)
    float y_ref = jetson.convertShift(shift_pixels); // -0,01 m

    std::cout << "Desired shift: " << y_ref << " mts" << std::endl;

    // Simular 10 passos com ângulo de direção fixo (ex.: 0 rad)
    float delta = 0.0f;
    for (int k = 0; k < 10; ++k) {
        jetson.update(delta);
        std::cout << "Step " << k << ": y = " << jetson.y << " m, psi = " << jetson.psi << " rad" << std::endl;
    }

    return 0;
}