#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>

class VehicleModel {
public:
    const int width = 640;      // Mask width
    const int height = 480;     // Mask height
    const float v = 3.0f;       // Speed: 3 m/s
    const float L = 0.15f;      // Wheelbase: 0.15 m
    const float dt = 0.05f;     // Time step: 0.05 s (20 Hz)
    const float lines_dist = 0.25f; // Distance between lines: 0.25 m
    float meters_per_pixel;     // Conversion: lines_dist / pixel distance between lines

    float y;    // Lateral position (meters)
    float psi;  // Heading angle (radians)

    VehicleModel() : y(0.0f), psi(0.0f), meters_per_pixel(0.0f) {}

    // Update state based on steering angle
    void update(float delta) {
        y += dt * v * std::sin(psi);
        psi += dt * (v / L) * std::tan(delta);
    }

    // Convert pixel shift to meters
    float convertShift(float pixel_shift) {
        return pixel_shift * meters_per_pixel;
    }

    // Compute meters_per_pixel based on pixel distance between lines
    void computeMetersPerPixel(float pixel_lane_width) {
        if (pixel_lane_width > 0) {
            meters_per_pixel = lines_dist / pixel_lane_width;
        }
    }
};

class MPCController {
public:
    const int N = 10;          // Prediction horizon
    const float R = 0.05f;     // Control weight
    const float delta_max = M_PI / 6.0f; // Max steering angle: 30 degrees
    const float dt = 0.05f;    // Time step: 0.05 s

    VehicleModel& model;

    MPCController(VehicleModel& vehicle) : model(vehicle) {}

    float computeCost(const std::vector<float>& delta, float y_ref) {
        float cost = 0.0f;
        float y = model.y;
        float psi = model.psi;

        for (int k = 0; k < N; ++k) {
            cost += std::pow(y - y_ref, 2) + R * std::pow(delta[k], 2);
            y += dt * model.v * std::sin(psi);
            psi += dt * (model.v / model.L) * std::tan(delta[k]);
        }
        return cost;
    }

    float optimize(float y_ref) {
        float delta = 0.0f;
        float learning_rate = 0.02f;
        float cost, cost_plus, cost_minus;
        const int max_iterations = 50;

        for (int i = 0; i < max_iterations; ++i) {
            std::vector<float> delta_seq(N, delta);
            cost = computeCost(delta_seq, y_ref);

            std::vector<float> delta_plus = delta_seq;
            delta_plus[0] = std::min(std::max(delta + 0.01f, -delta_max), delta_max);
            cost_plus = computeCost(delta_plus, y_ref);

            std::vector<float> delta_minus = delta_seq;
            delta_minus[0] = std::min(std::max(delta - 0.01f, -delta_max), delta_max);
            cost_minus = computeCost(delta_minus, y_ref);

            float gradient = (cost_plus - cost_minus) / 0.02f;
            delta -= learning_rate * gradient;
            delta = std::min(std::max(delta, -delta_max), delta_max);
        }

        return delta;
    }
};

float computeLateralShift(const cv::Mat& mask, VehicleModel& model) {
    if (mask.cols != model.width || mask.rows != model.height) {
        std::cerr << "Invalid mask dimensions! Expected: " << model.width << "x" << model.height << std::endl;
        return 0.0f;
    }

    float pixel_lane_width;
    int y_line = model.height - 1; // Line at bottom
    cv::Mat line = mask.row(y_line);

    std::vector<int> white_pixels;
    for (int x = 0; x < line.cols; ++x) {
        if (line.at<uchar>(0, x) == 255) {
            white_pixels.push_back(x);
        }
    }

    if (white_pixels.empty()) {
        std::cout << "No lines detected in mask!" << std::endl;
        return 0.0f;
    }

    int x_left = *std::min_element(white_pixels.begin(), white_pixels.end());
    int x_right = *std::max_element(white_pixels.begin(), white_pixels.end());
    pixel_lane_width = x_right - x_left; // Distance between lines in pixels
    model.computeMetersPerPixel(pixel_lane_width); // Update meters_per_pixel

    float x_center = (x_left + x_right) / 2.0f;
    float x_center_image = model.width / 2.0f;
    float shift = x_center - x_center_image;

    //std::cout << "Left line: x=" << x_left << ", Right line: x=" << x_right << std::endl;
    //std::cout << "Trajectory center: x=" << x_center << std::endl;
    //std::cout << "Lateral shift: " << shift << " pixels" << std::endl;
    //std::cout << "Meters per pixel: " << model.meters_per_pixel << std::endl;

    return shift;
}

// Example usage
int main() {
    VehicleModel vehicle;
    MPCController mpc(vehicle);

    // Simulate 20 steps (1 second at 20 Hz)
    for (int k = 0; k < 20; ++k) {
        cv::Mat mask = cv::imread("../mask/mask_test01.png", cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            std::cerr << "Error loading mask!" << std::endl;
            return -1;
        }

        float pixel_shift = computeLateralShift(mask, vehicle);
        float y_ref = vehicle.convertShift(pixel_shift);

        float delta = mpc.optimize(y_ref);
        vehicle.update(delta);

        std::cout << "Step " << k << ": y = " << vehicle.y << " m, psi = " << vehicle.psi
                  << " rad, delta = " << delta * 180.0f / M_PI << " deg, y_ref = " << y_ref << " m" << std::endl;
    }

    return 0;
}