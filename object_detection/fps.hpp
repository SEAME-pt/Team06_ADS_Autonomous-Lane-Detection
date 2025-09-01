#ifndef FPS_HPP
#define FPS_HPP

class FPSCalculator {
private:
    std::deque<double> frame_times;
    std::chrono::high_resolution_clock::time_point last_time;
    size_t window_size;
    int frame_count;

public:
    FPSCalculator(size_t window = 30) : window_size(window), frame_count(0) {
        last_time = std::chrono::high_resolution_clock::now();
    }

    void update() {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<double>(current_time - last_time).count();
        
        if (delta_time > 0.001 && delta_time < 1.0) {
            frame_times.push_back(delta_time);
            if (frame_times.size() > window_size) {
                frame_times.pop_front();
            }
        }
        
        last_time = current_time;
        frame_count++;
    }

    double getFPS() {
        if (frame_times.size() < 2) return 0.0;
        
        double avg_frame_time = 0.0;
        for (double time : frame_times) {
            avg_frame_time += time;
        }
        avg_frame_time /= frame_times.size();
        
        return avg_frame_time > 0 ? 1.0 / avg_frame_time : 0.0;
    }

    double getSmoothFPS() {
        if (frame_times.size() < 5) return getFPS();
        
        std::vector<double> sorted_times(frame_times.begin(), frame_times.end());
        std::sort(sorted_times.begin(), sorted_times.end());
        
        size_t trim_count = std::max(1, static_cast<int>(sorted_times.size() / 5));
        std::vector<double> trimmed_times(
            sorted_times.begin() + trim_count,
            sorted_times.end() - trim_count
        );
        
        if (trimmed_times.empty()) return getFPS();
        
        double avg_frame_time = 0.0;
        for (double time : trimmed_times) {
            avg_frame_time += time;
        }
        avg_frame_time /= trimmed_times.size();
        
        return avg_frame_time > 0 ? 1.0 / avg_frame_time : 0.0;
    }
};

#endif