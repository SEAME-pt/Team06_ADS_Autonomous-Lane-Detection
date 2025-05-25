#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/Vehicle.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageConverter.h>
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

    VehicleModel() : y(0.0f), psi(0.0f), meters_per_pixel(0.000496f) {}

    void update(float delta) {
        y += dt * v * std::sin(psi);
        psi += dt * (v / L) * std::tan(delta);
    }

    float convertShift(float pixel_shift) {
        return pixel_shift * meters_per_pixel;
    }

    void computeMetersPerPixel(float pixel_lane_width) {
        if (pixel_lane_width > 0) {
            meters_per_pixel = lines_dist / pixel_lane_width;
        }
    }
};

class MPCController {
public:
    const int N = 10;
    const float R = 0.05f;
    const float delta_max = M_PI / 6.0f;
    const float dt = 0.05f;

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

    int y_line = model.height - 1;
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
    float pixel_lane_width = x_right - x_left;
    model.computeMetersPerPixel(pixel_lane_width);

    float x_center = (x_left + x_right) / 2.0f;
    float x_center_image = model.width / 2.0f;
    float shift = x_center - x_center_image;

    std::cout << "Left line: x=" << x_left << ", Right line: x=" << x_right << std::endl;
    std::cout << "Trajectory center: x=" << x_center << std::endl;
    std::cout << "Lateral shift: " << shift << " pixels" << std::endl;
    std::cout << "Meters per pixel: " << model.meters_per_pixel << std::endl;

    return shift;
}

int main() {
    // CARLA setup
    auto client = carla::client::Client("localhost", 2000);
    client.SetTimeout(std::chrono::seconds(10));
    auto world = client.GetWorld();
    auto blueprint_library = world.GetBlueprintLibrary();

    // Synchronous mode
    auto settings = world.GetSettings();
    settings.SetSynchronousMode(true);
    settings.SetFixedDeltaSeconds(0.05); // 20 Hz
    world.ApplySettings(settings);

    // Spawn vehicle
    auto vehicle_bp = blueprint_library->Find("vehicle.tesla.model3");
    auto spawn_points = world.GetMap()->GetSpawnPoints();
    auto spawn_point = spawn_points[0]; // Use first spawn point
    auto vehicle_actor = world.SpawnActor(*vehicle_bp, spawn_point);
    auto vehicle = boost::static_pointer_cast<carla::client::Vehicle>(vehicle_actor);

    // Spawn RGB camera
    auto camera_bp = blueprint_library->Find("sensor.camera.rgb");
    camera_bp->SetAttribute("image_size_x", "640");
    camera_bp->SetAttribute("image_size_y", "480");
    camera_bp->SetAttribute("fov", "90");
    auto camera_transform = carla::geom::Transform(carla::geom::Location(2.0, 0, 1.4), carla::geom::Rotation(-15, 0, 0));
    auto camera_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
    auto camera = boost::static_pointer_cast<carla::client::Sensor>(camera_actor);

    // Spawn semantic segmentation camera
    auto seg_bp = blueprint_library->Find("sensor.camera.semantic_segmentation");
    seg_bp->SetAttribute("image_size_x", "640");
    seg_bp->SetAttribute("image_size_y", "480");
    seg_bp->SetAttribute("fov", "90");
    auto seg_actor = world.SpawnActor(*seg_bp, camera_transform, vehicle.get());
    auto seg_camera = boost::static_pointer_cast<carla::client::Sensor>(seg_actor);

    VehicleModel vehicle_model;
    MPCController mpc(vehicle_model);

    // Process segmentation image to generate mask
    seg_camera->Listen([&](auto data) {
        auto image = boost::static_pointer_cast<carla::sensor::data::Image>(data);
        auto height = image->GetHeight();
        auto width = image->GetWidth();
        cv::Mat mask(height, width, CV_8UC1);

        // Convert to raw segmentation data
        auto raw_data = image->GetRawData();
        auto pixels = reinterpret_cast<const uint8_t*>(raw_data.data());
        for (size_t i = 0; i < height * width; ++i) {
            uint8_t class_id = pixels[i * 4 + 2]; // Blue channel contains class ID
            mask.at<uchar>(i / width, i % width) = (class_id == 24) ? 255 : 0; // Class 24 = road lines
        }

        // Process mask with MPC
        float pixel_shift = computeLateralShift(mask, vehicle_model);
        float y_ref = vehicle_model.convertShift(pixel_shift);
        float delta = mpc.optimize(y_ref);
        vehicle_model.update(delta);

        // Apply control to vehicle
        float steering = delta / (M_PI / 6.0f); // Normalize to [-1, 1]
        vehicle->ApplyControl(carla::client::VehicleControl{.throttle = 0.5f, .steer = steering});

        std::cout << "y = " << vehicle_model.y << " m, psi = " << vehicle_model.psi
                  << " rad, delta = " << delta * 180.0f / M_PI << " deg, y_ref = " << y_ref << " m" << std::endl;

        // Tick world
        world->Tick();
    });

    // Simulate for 20 seconds
    for (int i = 0; i < 400; ++i) { // 20 s * 20 Hz = 400 ticks
        world->Tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Cleanup
    seg_camera->Stop();
    camera->Stop();
    vehicle_actor->Destroy();
    camera_actor->Destroy();
    seg_actor->Destroy();
    world->ApplySettings(carla::WorldSettings(false, 0.0));

    return 0;
}