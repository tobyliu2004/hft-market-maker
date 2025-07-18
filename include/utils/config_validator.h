#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <regex>

namespace hft {

class ConfigValidator {
public:
    struct ValidationRule {
        std::string field_path;
        std::function<bool(const nlohmann::json&)> validator;
        std::string error_message;
    };
    
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    static ValidationResult validateConfig(const nlohmann::json& config) {
        ValidationResult result{true, {}, {}};
        
        // FIX configuration validation
        validateFIXConfig(config, result);
        
        // Market data configuration validation
        validateMarketDataConfig(config, result);
        
        // Strategy configuration validation
        validateStrategyConfig(config, result);
        
        // Risk limits validation
        validateRiskLimits(config, result);
        
        // Execution configuration validation
        validateExecutionConfig(config, result);
        
        // Machine learning configuration validation
        validateMLConfig(config, result);
        
        return result;
    }
    
private:
    static void validateFIXConfig(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("fix")) {
            result.errors.push_back("Missing required 'fix' configuration section");
            result.is_valid = false;
            return;
        }
        
        auto& fix = config["fix"];
        
        // Required fields
        std::vector<std::string> required_fields = {
            "config_file", "sender_comp_id", "target_comp_id", "heartbeat_interval"
        };
        
        for (const auto& field : required_fields) {
            if (!fix.contains(field)) {
                result.errors.push_back("Missing required FIX field: " + field);
                result.is_valid = false;
            }
        }
        
        // Validate heartbeat interval
        if (fix.contains("heartbeat_interval")) {
            int heartbeat = fix["heartbeat_interval"];
            if (heartbeat < 1 || heartbeat > 300) {
                result.warnings.push_back("FIX heartbeat interval should be between 1 and 300 seconds");
            }
        }
    }
    
    static void validateMarketDataConfig(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("market_data")) {
            result.errors.push_back("Missing required 'market_data' configuration section");
            result.is_valid = false;
            return;
        }
        
        auto& md = config["market_data"];
        
        // Validate feed type
        if (md.contains("feed_type")) {
            std::string feed_type = md["feed_type"];
            if (feed_type != "polygon" && feed_type != "simulated" && feed_type != "nyse_taq") {
                result.errors.push_back("Invalid feed_type: " + feed_type);
                result.is_valid = false;
            }
        }
        
        // Validate API key for polygon
        if (md.contains("feed_type") && md["feed_type"] == "polygon") {
            if (!md.contains("polygon_api_key") || 
                md["polygon_api_key"] == "YOUR_POLYGON_API_KEY") {
                result.errors.push_back("Invalid or missing Polygon API key");
                result.is_valid = false;
            }
        }
        
        // Validate symbols
        if (md.contains("symbols")) {
            auto& symbols = md["symbols"];
            if (!symbols.is_array() || symbols.empty()) {
                result.errors.push_back("Symbols must be a non-empty array");
                result.is_valid = false;
            } else {
                // Validate symbol format
                std::regex symbol_regex("^[A-Z]{1,5}$");
                for (const auto& symbol : symbols) {
                    if (!std::regex_match(symbol.get<std::string>(), symbol_regex)) {
                        result.warnings.push_back("Symbol format warning: " + 
                            symbol.get<std::string>() + " may not be valid");
                    }
                }
            }
        }
        
        // Validate market depth
        if (md.contains("market_depth")) {
            int depth = md["market_depth"];
            if (depth < 1 || depth > 50) {
                result.warnings.push_back("Market depth should be between 1 and 50 levels");
            }
        }
    }
    
    static void validateStrategyConfig(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("strategy")) {
            result.errors.push_back("Missing required 'strategy' configuration section");
            result.is_valid = false;
            return;
        }
        
        auto& strategy = config["strategy"];
        
        // Validate strategy type
        if (strategy.contains("type")) {
            std::string type = strategy["type"];
            if (type != "avellaneda_stoikov" && type != "alpha" && type != "pairs") {
                result.errors.push_back("Unknown strategy type: " + type);
                result.is_valid = false;
            }
        }
        
        // Validate Avellaneda-Stoikov parameters
        if (strategy.contains("type") && strategy["type"] == "avellaneda_stoikov") {
            if (strategy.contains("risk_aversion")) {
                double risk_aversion = strategy["risk_aversion"];
                if (risk_aversion < 0 || risk_aversion > 10) {
                    result.warnings.push_back("Risk aversion should typically be between 0 and 10");
                }
            }
            
            if (strategy.contains("order_arrival_rate")) {
                double rate = strategy["order_arrival_rate"];
                if (rate <= 0) {
                    result.errors.push_back("Order arrival rate must be positive");
                    result.is_valid = false;
                }
            }
            
            if (strategy.contains("time_horizon")) {
                double horizon = strategy["time_horizon"];
                if (horizon <= 0 || horizon > 3600) {
                    result.warnings.push_back("Time horizon should be between 0 and 3600 seconds");
                }
            }
        }
        
        // Validate alpha weights
        if (strategy.contains("alpha_weights")) {
            auto& weights = strategy["alpha_weights"];
            double total = 0;
            for (auto& [key, value] : weights.items()) {
                double weight = value;
                if (weight < 0 || weight > 1) {
                    result.warnings.push_back("Alpha weight for " + key + 
                        " should be between 0 and 1");
                }
                total += weight;
            }
            
            if (std::abs(total - 1.0) > 0.01) {
                result.warnings.push_back("Alpha weights should sum to 1.0 (current: " + 
                    std::to_string(total) + ")");
            }
        }
    }
    
    static void validateRiskLimits(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("risk_limits")) {
            result.errors.push_back("Missing required 'risk_limits' configuration section");
            result.is_valid = false;
            return;
        }
        
        auto& risk = config["risk_limits"];
        
        // Validate position limits
        if (risk.contains("max_position_value") && risk.contains("max_total_exposure")) {
            double pos_value = risk["max_position_value"];
            double total_exp = risk["max_total_exposure"];
            
            if (pos_value > total_exp) {
                result.errors.push_back("max_position_value cannot exceed max_total_exposure");
                result.is_valid = false;
            }
        }
        
        // Validate percentage-based limits
        std::vector<std::pair<std::string, std::pair<double, double>>> pct_limits = {
            {"max_drawdown", {0, 1}},
            {"stop_loss_percent", {0, 1}},
            {"max_spread_percent", {0, 0.1}},
            {"min_liquidity_ratio", {0, 1}},
            {"max_market_impact", {0, 0.1}}
        };
        
        for (const auto& [field, range] : pct_limits) {
            if (risk.contains(field)) {
                double value = risk[field];
                if (value < range.first || value > range.second) {
                    result.errors.push_back(field + " must be between " + 
                        std::to_string(range.first) + " and " + 
                        std::to_string(range.second));
                    result.is_valid = false;
                }
            }
        }
        
        // Validate leverage and margin levels
        if (risk.contains("max_leverage")) {
            double leverage = risk["max_leverage"];
            if (leverage < 1 || leverage > 50) {
                result.warnings.push_back("max_leverage typically between 1 and 50");
            }
        }
        
        if (risk.contains("margin_call_level") && risk.contains("liquidation_level")) {
            double margin_call = risk["margin_call_level"];
            double liquidation = risk["liquidation_level"];
            
            if (liquidation >= margin_call) {
                result.errors.push_back("liquidation_level must be less than margin_call_level");
                result.is_valid = false;
            }
        }
        
        // Validate order limits
        if (risk.contains("max_orders_per_second")) {
            int rate = risk["max_orders_per_second"];
            if (rate < 1 || rate > 10000) {
                result.warnings.push_back("max_orders_per_second typically between 1 and 10000");
            }
        }
    }
    
    static void validateExecutionConfig(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("execution")) {
            result.warnings.push_back("Missing 'execution' configuration section");
            return;
        }
        
        auto& exec = config["execution"];
        
        // Validate order size consistency
        if (exec.contains("min_order_size") && exec.contains("max_order_size")) {
            double min_size = exec["min_order_size"];
            double max_size = exec["max_order_size"];
            
            if (min_size > max_size) {
                result.errors.push_back("min_order_size cannot exceed max_order_size");
                result.is_valid = false;
            }
            
            if (min_size <= 0) {
                result.errors.push_back("min_order_size must be positive");
                result.is_valid = false;
            }
        }
        
        // Validate order timeout
        if (exec.contains("order_timeout_ms")) {
            int timeout = exec["order_timeout_ms"];
            if (timeout < 100 || timeout > 60000) {
                result.warnings.push_back("order_timeout_ms typically between 100ms and 60s");
            }
        }
        
        // Validate iceberg orders
        if (exec.contains("enable_iceberg_orders") && exec["enable_iceberg_orders"]) {
            if (!exec.contains("iceberg_display_size")) {
                result.warnings.push_back("iceberg_display_size not specified for iceberg orders");
            }
        }
    }
    
    static void validateMLConfig(const nlohmann::json& config, ValidationResult& result) {
        if (!config.contains("machine_learning")) {
            return;  // ML config is optional
        }
        
        auto& ml = config["machine_learning"];
        
        if (ml.contains("enable_ml_predictions") && ml["enable_ml_predictions"]) {
            // Validate ML parameters
            if (ml.contains("confidence_threshold")) {
                double threshold = ml["confidence_threshold"];
                if (threshold < 0 || threshold > 1) {
                    result.errors.push_back("ML confidence_threshold must be between 0 and 1");
                    result.is_valid = false;
                }
            }
            
            if (ml.contains("prediction_horizon_ms")) {
                int horizon = ml["prediction_horizon_ms"];
                if (horizon < 10 || horizon > 60000) {
                    result.warnings.push_back("ML prediction_horizon_ms typically between 10ms and 60s");
                }
            }
            
            if (ml.contains("model_update_interval_minutes")) {
                int interval = ml["model_update_interval_minutes"];
                if (interval < 1 || interval > 1440) {
                    result.warnings.push_back("model_update_interval_minutes typically between 1 and 1440 (24h)");
                }
            }
        }
    }
};

} // namespace hft