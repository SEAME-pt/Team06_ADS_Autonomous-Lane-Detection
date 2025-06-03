#include <iostream>
#include "mask_processor.hpp"

int main() {
    cv::Mat mask = cv::imread("../mask/output_masks/da_mask_binary_08.png", cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        std::cout << "Erro ao carregar a imagem!" << std::endl;
        return -1;
    }

    MaskProcessor processor;
    cv::Mat output;
    std::vector<cv::Point> medianPoints;
    processor.processMask(mask, output, medianPoints);

    // Exibir os pontos da mediana (opcional)
    for (const auto& p : medianPoints) {
        std::cout << "Mediana: (" << p.x << ", " << p.y << ")" << std::endl;
    }

    return 0;
}