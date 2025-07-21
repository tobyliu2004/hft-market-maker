#include "utils/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>

namespace hft {

// Static member initialization
std::unique_ptr<Logger> Logger::instance_ = nullptr;
std::mutex Logger::instance_mutex_;

Logger::Logger(LogLevel level, const std::string& file_path)
    : log_level_(level), running_(true) {
    
    if (!file_path.empty()) {
        log_file_.open(file_path, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << file_path << std::endl;
        }
    }
    
    // Start the logging thread
    log_thread_ = std::thread(&Logger::processLogs, this);
}

Logger::~Logger() {
    running_ = false;
    
    // Wake up the logging thread
    log_cv_.notify_all();
    
    // Wait for the thread to finish
    if (log_thread_.joinable()) {
        log_thread_.join();
    }
    
    // Process any remaining logs
    while (!log_queue_.empty()) {
        auto& entry = log_queue_.front();
        writeLog(entry);
        log_queue_.pop();
    }
    
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::initialize(LogLevel level, const std::string& file_path) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_ = std::unique_ptr<Logger>(new Logger(level, file_path));
}

Logger& Logger::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::unique_ptr<Logger>(new Logger(LogLevel::INFO, ""));
    }
    return *instance_;
}

void Logger::log(LogLevel level, const std::string& message, 
                const std::string& file, int line) {
    if (level < log_level_) {
        return;
    }
    
    LogEntry entry;
    entry.timestamp = std::chrono::high_resolution_clock::now();
    entry.level = level;
    entry.message = message;
    entry.thread_id = std::this_thread::get_id();
    entry.file = file;
    entry.line = line;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        log_queue_.push(std::move(entry));
    }
    
    log_cv_.notify_one();
}

void Logger::processLogs() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for logs or timeout every 100ms to check running_ flag
        log_cv_.wait_for(lock, std::chrono::milliseconds(100),
                        [this] { return !log_queue_.empty() || !running_; });
        
        while (!log_queue_.empty()) {
            auto entry = std::move(log_queue_.front());
            log_queue_.pop();
            
            // Release lock while writing
            lock.unlock();
            writeLog(entry);
            lock.lock();
        }
    }
}

void Logger::writeLog(const LogEntry& entry) {
    std::stringstream ss;
    
    // Format timestamp
    auto time_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now()
    );
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()
    ) % 1000;
    
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    // Add log level
    ss << " [" << levelToString(entry.level) << "]";
    
    // Add thread ID
    ss << " [" << entry.thread_id << "]";
    
    // Add file and line if available
    if (!entry.file.empty()) {
        // Extract just the filename, not the full path
        size_t last_slash = entry.file.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos) 
            ? entry.file.substr(last_slash + 1) : entry.file;
        ss << " [" << filename << ":" << entry.line << "]";
    }
    
    // Add message
    ss << " " << entry.message;
    
    std::string log_line = ss.str();
    
    // Write to console with color coding
    {
        std::lock_guard<std::mutex> lock(console_mutex_);
        switch (entry.level) {
            case LogLevel::DEBUG:
                std::cout << "\033[36m" << log_line << "\033[0m" << std::endl;
                break;
            case LogLevel::INFO:
                std::cout << log_line << std::endl;
                break;
            case LogLevel::WARNING:
                std::cout << "\033[33m" << log_line << "\033[0m" << std::endl;
                break;
            case LogLevel::ERROR:
                std::cerr << "\033[31m" << log_line << "\033[0m" << std::endl;
                break;
            case LogLevel::CRITICAL:
                std::cerr << "\033[1;31m" << log_line << "\033[0m" << std::endl;
                break;
        }
    }
    
    // Write to file
    if (log_file_.is_open()) {
        std::lock_guard<std::mutex> lock(file_mutex_);
        log_file_ << log_line << std::endl;
        log_file_.flush();
    }
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRIT";
        default: return "UNKNOWN";
    }
}

void Logger::setLogLevel(LogLevel level) {
    log_level_ = level;
}

void Logger::flush() {
    if (log_file_.is_open()) {
        std::lock_guard<std::mutex> lock(file_mutex_);
        log_file_.flush();
    }
}

// Performance logger implementation
PerformanceLogger::PerformanceLogger(const std::string& operation)
    : operation_(operation),
      start_time_(std::chrono::high_resolution_clock::now()) {
    LOG_DEBUG("Performance timer started for: " + operation);
}

PerformanceLogger::~PerformanceLogger() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_
    );
    
    std::stringstream ss;
    ss << "Performance: " << operation_ << " took " 
       << duration.count() << " microseconds";
    LOG_DEBUG(ss.str());
}

} // namespace hft