#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int h = 480;
int x = 640;

float calLatShift(const cv::Mat& mask){
    // mask: imagem binária 256x128
    // line y=120 (para evitar bordas)
    cv::Mat line = mask.row(479);

    // find white pixels(lines)
    std::vector<int> whitePixels;
    for (int x = 0; x < line.cols; ++x){
        if (line.at<uchar>(0, x) == 255)
            whitePixels.push_back(x);
    }


    // check if find lines
    if (whitePixels.empty()){
        std::cout << "None line found in mask" << std::endl;
        return 0.0f;
    }

    // left line: minor x; right line: larger x
    int xLeft = *std::min_element(whitePixels.begin(), whitePixels.end());
    int xRight = *std::max_element(whitePixels.begin(), whitePixels.end());

    // center cal
    float xCenter = (xLeft + xRight) / 2.0f;

    // Center of img (aligned with the vehicle)
    float xCenterImg = x / 2.0f;

    // Shift lateral (pixels)
    float shift = xCenter - xCenterImg;

    // Imprimir informações para depuração
    std::cout << "Linha esquerda: x=" << xLeft << ", Linha direita: x=" << xRight << std::endl;
    std::cout << "Centro da trajetória: x=" << xCenter << std::endl;
    std::cout << "Deslocamento lateral: " << shift << " pixels" << std::endl;

    return shift;
}

int main () {
    // Carregar máscara (exemplo: substitua pelo caminho da sua máscara)
    cv::Mat mask = cv::imread("mask/mask_test01.png", cv::IMREAD_GRAYSCALE);
    
    if (mask.empty()) {
        std::cerr << "Erro ao carregar a máscara!" << std::endl;
        return -1;
    }

    // Verificar dimensões
    if (mask.cols != x || mask.rows != h) {
        std::cerr << "Máscara com dimensões incorretas! Esperado: 256x128" << std::endl;
        return -1;
    }

    // Calcular deslocamento
    float shift = calLatShift(mask);

    return 0;
}