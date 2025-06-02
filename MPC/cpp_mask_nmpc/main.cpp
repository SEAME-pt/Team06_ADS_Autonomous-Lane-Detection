#include "nmpc_controller.hpp"
#include "lane_processor.hpp"
#include "vehicle_interface.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    double L = 0.15;
    double dt = 0.05;
    int Np = 10;
    int Nc = 5;
    double delta_max = 0.349;
    double a_max = 1.0;
    
    NMPCController nmpc(L, dt, Np, Nc, delta_max, a_max);
    LaneProcessor lane_proc(Np, dt);
    VehicleInterface vehicle;
    
    cv::Mat ll_mask = cv::imread("../../mask/output_masks/da_mask_binary_07.png", cv::IMREAD_GRAYSCALE);
    if (ll_mask.empty()) {
        std::cerr << "Erro ao carregar ll_mask.png!" << std::endl;
        return -1;
    }
    
    cv::Mat display;
    cv::cvtColor(ll_mask, display, cv::COLOR_GRAY2BGR);
    
    auto x_ref = lane_proc.process_lane(ll_mask);
    auto centerline = lane_proc.extract_centerline(ll_mask);
    
    for (size_t i = 0; i < centerline.size() - 1; ++i) {
        cv::Point p1(centerline[i][0], centerline[i][1]);
        cv::Point p2(centerline[i + 1][0], centerline[i + 1][1]);
        cv::line(display, p1, p2, cv::Scalar(0, 255, 0), 2); // Verde
    }
    
    double scale = 50.0; // Ajuste da escala
    std::vector<cv::Point> vehicle_path;
    for (int i = 0; i < 100; ++i) {
        auto x0 = vehicle.get_state();
        auto control = nmpc.compute_control(x0, x_ref);
        vehicle.apply_control(control[0], control[1]);
        
        int x_pixel = static_cast<int>(ll_mask.cols / 2.0 + x0[0] * scale);
        int y_pixel = static_cast<int>(ll_mask.rows - x0[1] * scale);
        vehicle_path.push_back(cv::Point(x_pixel, y_pixel));
        cv::circle(display, cv::Point(x_pixel, y_pixel), 3, cv::Scalar(0, 0, 255), -1);
        
        std::cout << "Estado: [" << x0[0] << ", " << x0[1] << ", "
                  << x0[2] << ", " << x0[3] << "] "
                  << "Controle: [" << control[0] << ", " << control[1] << "]" << std::endl;
        
        cv::imshow("NMPC Visualization", display);
        cv::waitKey(50);
    }
    
    for (size_t i = 0; i < vehicle_path.size() - 1; ++i) {
        cv::line(display, vehicle_path[i], vehicle_path[i + 1], cv::Scalar(0, 0, 255), 1);
    }
    
    cv::imshow("NMPC Visualization", display);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}