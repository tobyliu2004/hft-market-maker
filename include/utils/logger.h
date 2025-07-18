#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace hft {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
private:
    LogLevel min_level_;
    std::string log_file_;
    std::ofstream file_stream_;
    mutable std::mutex mutex_;
    
    // Async logging
    std::queue<std::string> log_queue_;
    std::thread worker_thread_;
    std::condition_variable cv_;
    std::atomic<bool> running_{true};
    
    static std::shared_ptr<Logger> instance_;
    
    Logger(LogLevel level, const std::string& file_path) 
        : min_level_(level), log_file_(file_path) {
        if (!file_path.empty()) {
            file_stream_.open(file_path, std::ios::app);
            if (!file_stream_.is_open()) {
                std::cerr << "Failed to open log file: " << file_path << std::endl;
            }
        }
        
        // Start async logging thread
        worker_thread_ = std::thread(&Logger::processLogs, this);
    }
    
    void processLogs() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !log_queue_.empty() || !running_; });
            
            while (!log_queue_.empty()) {
                std::string message = log_queue_.front();
                log_queue_.pop();
                
                // Write to file
                if (file_stream_.is_open()) {
                    file_stream_ << message << std::endl;
                    file_stream_.flush();
                }
                
                // Also write to console
                std::cout << message << std::endl;
            }
        }
    }
    
    std::string levelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
    
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        
        return ss.str();
    }
    
public:
    ~Logger() {
        running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
    }
    
    static void initialize(LogLevel level, const std::string& file_path) {
        if (!instance_) {
            instance_ = std::shared_ptr<Logger>(new Logger(level, file_path));
        }
    }
    
    static std::shared_ptr<Logger> getInstance() {
        if (!instance_) {
            // Default initialization
            initialize(LogLevel::INFO, "market_maker.log");
        }
        return instance_;
    }
    
    void log(LogLevel level, const std::string& message, 
             const std::string& file = "", int line = 0) {
        if (level < min_level_) return;
        
        std::stringstream ss;
        ss << "[" << getCurrentTimestamp() << "] ";
        ss << "[" << levelToString(level) << "] ";
        
        if (!file.empty()) {
            // Extract filename from path
            size_t pos = file.find_last_of("/\\");
            std::string filename = (pos != std::string::npos) ? 
                file.substr(pos + 1) : file;
            ss << "[" << filename << ":" << line << "] ";
        }
        
        ss << message;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            log_queue_.push(ss.str());
        }
        cv_.notify_one();
    }
    
    void setLevel(LogLevel level) {
        min_level_ = level;
    }
};

// Initialize static member
std::shared_ptr<Logger> Logger::instance_ = nullptr;

// Convenience macros
#define LOG_DEBUG(msg) Logger::getInstance()->log(LogLevel::DEBUG, msg, __FILE__, __LINE__)
#define LOG_INFO(msg) Logger::getInstance()->log(LogLevel::INFO, msg, __FILE__, __LINE__)
#define LOG_WARNING(msg) Logger::getInstance()->log(LogLevel::WARNING, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) Logger::getInstance()->log(LogLevel::ERROR, msg, __FILE__, __LINE__)
#define LOG_CRITICAL(msg) Logger::getInstance()->log(LogLevel::CRITICAL, msg, __FILE__, __LINE__)

// Performance logging
class PerformanceLogger {
private:
    std::string operation_;
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    explicit PerformanceLogger(const std::string& operation) 
        : operation_(operation), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~PerformanceLogger() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start_).count();
        
        std::stringstream ss;
        ss << "Performance: " << operation_ << " took " << duration << " μs";
        LOG_DEBUG(ss.str());
    }
};

#define PERF_LOG(operation) PerformanceLogger _perf_logger(operation)

} // namespace hft