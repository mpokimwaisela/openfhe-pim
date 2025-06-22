// ─────────────────────────── profiler.hpp ───────────────────────────
#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <map>
#include <mutex>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

namespace pim {

class Profiler {
public:
    struct Timer {
        std::chrono::high_resolution_clock::time_point start;
        std::string name;
        
        Timer(const std::string& timer_name) : name(timer_name) {
            start = std::chrono::high_resolution_clock::now();
        }
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            Profiler::instance().add_time(name, duration.count());
        }
    };
    
    static Profiler& instance() {
        static Profiler prof;
        return prof;
    }
     void add_time(const std::string& name, long microseconds) {
        // std::lock_guard<std::mutex> lock(mutex_); // REMOVED - single threaded
        times_[name] += microseconds;
        counts_[name]++;
        samples_[name].push_back(microseconds);
    }

    void print_report() const {
        // std::lock_guard<std::mutex> lock(mutex_); // REMOVED - single threaded
        
        std::cout << "\n" << std::string(110, '=') << std::endl;
        std::cout << "                           PERFORMANCE PROFILING REPORT" << std::endl;
        std::cout << std::string(110, '=') << std::endl;
        
        // Calculate total time
        long total_time = 0;
        for (std::map<std::string, long>::const_iterator it = times_.begin(); it != times_.end(); ++it) {
            total_time += it->second;
        }
        
        // Print header with additional columns
        std::cout << std::left << std::setw(25) << "Operation" 
                  << std::right << std::setw(10) << "Total(μs)" 
                  << std::setw(10) << "Avg(μs)" 
                  << std::setw(10) << "Min(μs)"
                  << std::setw(10) << "Max(μs)"
                  << std::setw(12) << "StdDev(μs)"
                  << std::setw(8) << "Count" 
                  << std::setw(8) << "%" << std::endl;
        std::cout << std::string(110, '-') << std::endl;
        
        // Sort by total time (descending)
        std::vector<std::pair<std::string, long> > sorted_times(times_.begin(), times_.end());
        
        // Custom comparator for sorting by time (descending)
        struct TimeComparator {
            bool operator()(const std::pair<std::string, long>& a, const std::pair<std::string, long>& b) const {
                return a.second > b.second;
            }
        };
        std::sort(sorted_times.begin(), sorted_times.end(), TimeComparator());
        
        for (std::vector<std::pair<std::string, long> >::const_iterator it = sorted_times.begin(); 
             it != sorted_times.end(); ++it) {
            const std::string& name = it->first;
            long time = it->second;
            int count = counts_.find(name)->second;
            
            // Calculate statistics from samples
            const std::vector<long>& samples = samples_.find(name)->second;
            
            // Average
            double avg = static_cast<double>(time) / count;
            
            // Min and Max
            long min_time = *std::min_element(samples.begin(), samples.end());
            long max_time = *std::max_element(samples.begin(), samples.end());
            
            // Median
            std::vector<long> sorted_samples = samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            // double median;
            // size_t n = sorted_samples.size();
            // if (n % 2 == 0) {
            //     median = (sorted_samples[n/2 - 1] + sorted_samples[n/2]) / 2.0;
            // } else {
            //     median = sorted_samples[n/2];
            // }
            
            // Standard deviation
            double sum_sq_diff = 0.0;
            for (std::vector<long>::const_iterator sample_it = samples.begin(); 
                 sample_it != samples.end(); ++sample_it) {
                double diff = *sample_it - avg;
                sum_sq_diff += diff * diff;
            }
            double std_dev = std::sqrt(sum_sq_diff / count);
            
            // Percentage
            double percentage = total_time > 0 ? (static_cast<double>(time) / total_time) * 100.0 : 0.0;
            
            // Print row with all statistics
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(10) << time
                      << std::setw(10) << std::fixed << std::setprecision(1) << avg
                      << std::setw(10) << min_time
                      << std::setw(10) << max_time
                      << std::setw(12) << std::fixed << std::setprecision(1) << std_dev
                      << std::setw(8) << count
                      << std::setw(7) << std::fixed << std::setprecision(1) << percentage << "%"
                      << std::endl;
        }
        
        std::cout << std::string(110, '-') << std::endl;
        std::cout << std::left << std::setw(25) << "TOTAL"
                  << std::right << std::setw(10) << total_time << " μs ("
                  << std::fixed << std::setprecision(2) << (total_time / 1000.0) << " ms)" << std::endl;
        std::cout << std::string(110, '=') << std::endl;
    }
    
    /**
     * @brief Get specific statistics for programmatic use
     */
    struct Statistics {
        long total;
        double average;
        long minimum;
        long maximum;
        double median;
        double std_deviation;
        int count;
        
        Statistics() : total(0), average(0.0), minimum(0), maximum(0), 
                      median(0.0), std_deviation(0.0), count(0) {}
    };
    
    Statistics get_statistics(const std::string& operation_name) const {
        // std::lock_guard<std::mutex> lock(mutex_); // REMOVED - single threaded
        
        Statistics stats;
        
        std::map<std::string, std::vector<long> >::const_iterator it = samples_.find(operation_name);
        if (it == samples_.end()) {
            return stats; // Return default-constructed stats
        }
        
        const std::vector<long>& samples = it->second;
        stats.total = times_.find(operation_name)->second;
        stats.count = counts_.find(operation_name)->second;
        stats.average = static_cast<double>(stats.total) / stats.count;
        
        std::vector<long> sorted_samples = samples;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        
        stats.minimum = sorted_samples.front();
        stats.maximum = sorted_samples.back();
        
        // Median
        size_t n = sorted_samples.size();
        if (n % 2 == 0) {
            stats.median = (sorted_samples[n/2 - 1] + sorted_samples[n/2]) / 2.0;
        } else {
            stats.median = sorted_samples[n/2];
        }
        
        // Standard deviation
        double sum_sq_diff = 0.0;
        for (std::vector<long>::const_iterator sample_it = samples.begin(); 
             sample_it != samples.end(); ++sample_it) {
            double diff = *sample_it - stats.average;
            sum_sq_diff += diff * diff;
        }
        stats.std_deviation = std::sqrt(sum_sq_diff / stats.count);
        
        return stats;
    }
    
    void clear() {
        // std::lock_guard<std::mutex> lock(mutex_); // REMOVED - single threaded
        times_.clear();
        counts_.clear();
        samples_.clear();
    }
    
private:
    mutable std::mutex mutex_;
    std::map<std::string, long> times_;
    std::map<std::string, int> counts_;
    std::map<std::string, std::vector<long> > samples_;  // Store individual timing samples
};

// RAII timer for easy profiling
#define PROFILE_SCOPE(name) pim::Profiler::Timer _timer(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

} // namespace pim
