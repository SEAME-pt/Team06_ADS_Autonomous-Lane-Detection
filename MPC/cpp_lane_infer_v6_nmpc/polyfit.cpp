#include "polyfit.hpp"


// Estrutura para coeficientes do polinômio
struct PolyCoefficients {
    double a0, a1, a2; // y = a0 + a1*x + a2*x^2
    bool valid;
};

// Função para ajustar um polinômio de 2º grau via regressão linear
PolyCoefficients fitPolynomial(const LaneData& laneData) {
    PolyCoefficients coeffs = {0.0, 0.0, 0.0, false};
    if (laneData.num_points < 3 || !laneData.valid) return coeffs;

    // Montar sistema linear: y = X * beta, onde beta = [a0, a1, a2]
    std::vector<double> x(laneData.num_points), y(laneData.num_points);
    for (int i = 0; i < laneData.num_points; ++i) {
        x[i] = laneData.points[i].x;
        y[i] = laneData.points[i].y;
    }

    // Matriz X (n x 3): [1, x, x^2]
    cv::Mat X(laneData.num_points, 3, CV_64F);
    for (int i = 0; i < laneData.num_points; ++i) {
        X.at<double>(i, 0) = 1.0;
        X.at<double>(i, 1) = x[i];
        X.at<double>(i, 2) = x[i] * x[i];
    }

    // Vetor y
    cv::Mat Y(laneData.num_points, 1, CV_64F, y.data());

    // Resolver: beta = (X^T * X)^(-1) * X^T * y
    cv::Mat beta;
    cv::solve(X.t() * X, X.t() * Y, beta, cv::DECOMP_SVD);

    coeffs.a0 = beta.at<double>(0);
    coeffs.a1 = beta.at<double>(1);
    coeffs.a2 = beta.at<double>(2);
    coeffs.valid = true;

    return coeffs;
}

// Função para gerar trajetória de referência
std::vector<std::vector<double>> generateReference(const LaneData& laneData, int frameCount) {
    const int Np = 10; // Horizonte de predição (de NMPCController)
    const double dt = 0.1; // Passo de tempo
    const double v_ref = 2.0; // Velocidade constante de referência
    std::vector<std::vector<double>> x_ref(Np, std::vector<double>(4, 0.0)); // [x, y, psi, v]

    // Ajustar polinômio à mediana da pista
    PolyCoefficients coeffs = fitPolynomial(laneData);

    if (!coeffs.valid) {
        // Caso inválido: assumir linha reta com psi = 0
        for (int k = 0; k < Np; ++k) {
            x_ref[k][0] = 0.0; // x
            x_ref[k][1] = v_ref * dt * k; // y
            x_ref[k][2] = 0.0; // psi
            x_ref[k][3] = v_ref; // v
        }
        return x_ref;
    }

    // Gerar pontos de referência ao longo do horizonte
    for (int k = 0; k < Np; ++k) {
        // Posição x ao longo do horizonte (baseado em v_ref e dt)
        double x_k = v_ref * dt * k;

        // Calcular y usando o polinômio
        double y_k = coeffs.a0 + coeffs.a1 * x_k + coeffs.a2 * x_k * x_k;

        // Calcular psi (ângulo da tangente)
        double dy_dx = coeffs.a1 + 2.0 * coeffs.a2 * x_k; // Derivada: dy/dx
        double psi_k = std::atan(dy_dx);

        // Normalizar psi para [-pi, pi]
        psi_k = std::atan2(std::sin(psi_k), std::cos(psi_k));

        // Preencher vetor de referência
        x_ref[k][0] = x_k; // x
        x_ref[k][1] = y_k; // y
        x_ref[k][2] = psi_k; // psi
        x_ref[k][3] = v_ref; // v
    }

    return x_ref;
}