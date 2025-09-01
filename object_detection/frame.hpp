#ifndef FRAME_HPP
#define FRAME_HPP

// Classe para frame skipping
class FrameSkipper {
public:
    enum Strategy { FIXED, ADAPTIVE, TIME_BASED };

private:
    Strategy skip_strategy;
    int skip_frames;
    double target_fps;
    int frame_counter;
    std::chrono::high_resolution_clock::time_point last_process_time;
    std::deque<double> processing_times;

public:
    FrameSkipper(Strategy strategy = FIXED, int skip = 2, double target = 15.0)
        : skip_strategy(strategy), skip_frames(skip), target_fps(target), frame_counter(0) {
        last_process_time = std::chrono::high_resolution_clock::now();
    }

    bool shouldProcessFrame() {
        bool should_process = false;
        
        if (skip_strategy == FIXED) {
            should_process = (frame_counter % (skip_frames + 1)) == 0;
        }
        else if (skip_strategy == ADAPTIVE) {
            if (processing_times.size() > 5) {
                double avg_processing_time = 0.0;
                for (double time : processing_times) {
                    avg_processing_time += time;
                }
                avg_processing_time /= processing_times.size();
                
                double target_interval = 1.0 / target_fps;
                
                if (avg_processing_time > target_interval) {
                    int skip_ratio = std::max(1, static_cast<int>(avg_processing_time / target_interval));
                    should_process = (frame_counter % skip_ratio) == 0;
                } else {
                    should_process = true;
                }
            } else {
                should_process = true;
            }
        }
        else if (skip_strategy == TIME_BASED) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto time_since_last = std::chrono::duration<double>(current_time - last_process_time).count();
            double target_interval = 1.0 / target_fps;
            
            should_process = time_since_last >= target_interval;
            
            if (should_process) {
                last_process_time = current_time;
            }
        }
        
        frame_counter++;
        return should_process;
    }

    void recordProcessingTime(double processing_time) {
        if (processing_time > 0.001 && processing_time < 5.0) {
            processing_times.push_back(processing_time);
            if (processing_times.size() > 20) {
                processing_times.pop_front();
            }
        }
    }

    void setStrategy(Strategy strategy) { skip_strategy = strategy; }
    Strategy getStrategy() const { return skip_strategy; }
};

#endif