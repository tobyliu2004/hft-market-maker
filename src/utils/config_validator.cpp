#include "utils/config_validator.h"
#include <regex>
#include <algorithm>
#include <numeric>
#include <set>

namespace hft {

ConfigValidator::ValidationResult ConfigValidator::validateConfig(const nlohmann::json& config) {
    ValidationResult result;
    result.is_valid = true;
    
    // Validate required sections
    validateRequiredSections(config, result);
    
    // Validate each section
    if (config.contains("fix")) {
        validateFIXConfig(config["fix"], result);
    }
    
    if (config.contains("market_data")) {
        validateMarketDataConfig(config["market_data"], result);
    }
    
    if (config.contains("strategy")) {
        validateStrategyConfig(config["strategy"], result);
    }
    
    if (config.contains("risk_limits")) {
        validateRiskLimitsConfig(config["risk_limits"], result);
    }
    
    if (config.contains("execution")) {
        validateExecutionConfig(config["execution"], result);
    }
    
    if (config.contains("machine_learning")) {
        validateMLConfig(config["machine_learning"], result);
    }
    
    // Cross-section validation
    validateCrossSectionConsistency(config, result);
    
    return result;
}

void ConfigValidator::validateRequiredSections(const nlohmann::json& config, ValidationResult& result) {
    const std::vector<std::string> required_sections = {
        "fix", "market_data", "strategy", "risk_limits", "execution"
    };
    
    for (const auto& section : required_sections) {
        if (!config.contains(section)) {
            result.errors.push_back("Missing required '" + section + "' configuration section");
            result.is_valid = false;
        }
    }
}

void ConfigValidator::validateFIXConfig(const nlohmann::json& fix_config, ValidationResult& result) {
    // Required fields
    const std::vector<std::string> required_fields = {
        "config_file", "sender_comp_id", "target_comp_id"
    };
    
    for (const auto& field : required_fields) {
        if (!fix_config.contains(field) || fix_config[field].empty()) {
            result.errors.push_back("FIX configuration missing required field: " + field);
            result.is_valid = false;
        }
    }
    
    // Validate heartbeat interval
    if (fix_config.contains("heartbeat_interval")) {
        int heartbeat = fix_config["heartbeat_interval"];
        if (heartbeat < 1 || heartbeat > 300) {
            result.warnings.push_back(
                "FIX heartbeat_interval should be between 1 and 300 seconds (current: " + 
                std::to_string(heartbeat) + ")"
            );
        }
    }
}

void ConfigValidator::validateMarketDataConfig(const nlohmann::json& md_config, ValidationResult& result) {
    // Validate feed type
    if (!md_config.contains("feed_type")) {
        result.errors.push_back("Market data configuration missing 'feed_type'");
        result.is_valid = false;
    } else {
        std::string feed_type = md_config["feed_type"];
        const std::set<std::string> valid_feeds = {"polygon", "simulated", "nasdaq", "nyse"};
        
        if (valid_feeds.find(feed_type) == valid_feeds.end()) {
            result.errors.push_back("Invalid feed_type: " + feed_type + 
                                  ". Must be one of: polygon, simulated, nasdaq, nyse");
            result.is_valid = false;
        }
        
        // Validate API key for polygon
        if (feed_type == "polygon") {
            if (!md_config.contains("polygon_api_key") || 
                md_config["polygon_api_key"] == "YOUR_POLYGON_API_KEY" ||
                md_config["polygon_api_key"].get<std::string>().empty()) {
                result.errors.push_back("Invalid or missing Polygon API key");
                result.is_valid = false;
            }
        }
    }
    
    // Validate symbols
    if (!md_config.contains("symbols") || md_config["symbols"].empty()) {
        result.errors.push_back("Market data configuration must include at least one symbol");
        result.is_valid = false;
    } else {
        for (const auto& symbol : md_config["symbols"]) {
            std::string sym = symbol;
            // Basic symbol validation (1-5 uppercase letters)
            if (!std::regex_match(sym, std::regex("^[A-Z]{1,5}$"))) {
                result.warnings.push_back("Symbol format warning: " + sym + 
                                        " should be 1-5 uppercase letters");
            }
        }
    }
    
    // Validate market depth
    if (md_config.contains("market_depth")) {
        int depth = md_config["market_depth"];
        if (depth < 1 || depth > 50) {
            result.warnings.push_back(
                "Market depth should be between 1 and 50 levels (current: " + 
                std::to_string(depth) + ")"
            );
        }
    }
}

void ConfigValidator::validateStrategyConfig(const nlohmann::json& strat_config, ValidationResult& result) {
    // Validate strategy type
    if (!strat_config.contains("type")) {
        result.errors.push_back("Strategy configuration missing 'type'");
        result.is_valid = false;
    } else {
        std::string strat_type = strat_config["type"];
        const std::set<std::string> valid_strategies = {
            "avellaneda_stoikov", "alpha", "statistical_arbitrage", "momentum"
        };
        
        if (valid_strategies.find(strat_type) == valid_strategies.end()) {
            result.errors.push_back("Unknown strategy type: " + strat_type);
            result.is_valid = false;
        }
        
        // Validate strategy-specific parameters
        if (strat_type == "avellaneda_stoikov") {
            // Risk aversion
            if (strat_config.contains("risk_aversion")) {
                double risk_aversion = strat_config["risk_aversion"];
                if (risk_aversion < 0) {
                    result.warnings.push_back(
                        "Risk aversion should be positive (current: " + 
                        std::to_string(risk_aversion) + ")"
                    );
                }
            }
            
            // Order arrival rate
            if (!strat_config.contains("order_arrival_rate")) {
                result.errors.push_back("Avellaneda-Stoikov strategy requires 'order_arrival_rate'");
                result.is_valid = false;
            } else {
                double rate = strat_config["order_arrival_rate"];
                if (rate <= 0) {
                    result.errors.push_back("Order arrival rate must be positive");
                    result.is_valid = false;
                }
            }
            
            // Time horizon
            if (!strat_config.contains("time_horizon")) {
                result.errors.push_back("Avellaneda-Stoikov strategy requires 'time_horizon'");
                result.is_valid = false;
            } else {
                double horizon = strat_config["time_horizon"];
                if (horizon <= 0) {
                    result.errors.push_back("Time horizon must be positive");
                    result.is_valid = false;
                }
            }
        }
    }
    
    // Validate alpha weights if present
    if (strat_config.contains("enable_alpha_signals") && strat_config["enable_alpha_signals"]) {
        if (strat_config.contains("alpha_weights")) {
            const auto& weights = strat_config["alpha_weights"];
            double total_weight = 0.0;
            
            for (const auto& [key, value] : weights.items()) {
                total_weight += value.get<double>();
            }
            
            if (std::abs(total_weight - 1.0) > 0.01) {
                result.warnings.push_back(
                    "Alpha weights should sum to 1.0 (current sum: " + 
                    std::to_string(total_weight) + ")"
                );
            }
        }
    }
}

void ConfigValidator::validateRiskLimitsConfig(const nlohmann::json& risk_config, ValidationResult& result) {
    // Validate position limits
    if (risk_config.contains("max_position_value") && risk_config.contains("max_total_exposure")) {
        double max_pos = risk_config["max_position_value"];
        double max_exp = risk_config["max_total_exposure"];
        
        if (max_pos > max_exp) {
            result.errors.push_back(
                "max_position_value cannot exceed max_total_exposure"
            );
            result.is_valid = false;
        }
    }
    
    // Validate percentage-based limits
    const std::vector<std::pair<std::string, double>> percentage_limits = {
        {"max_drawdown", 1.0},
        {"stop_loss_percent", 1.0},
        {"max_spread_percent", 0.1},
        {"min_liquidity_ratio", 1.0},
        {"max_market_impact", 0.1}
    };
    
    for (const auto& [field, max_val] : percentage_limits) {
        if (risk_config.contains(field)) {
            double value = risk_config[field];
            if (value < 0 || value > max_val) {
                result.errors.push_back(
                    field + " must be between 0 and " + std::to_string(max_val)
                );
                result.is_valid = false;
            }
        }
    }
    
    // Validate margin levels
    if (risk_config.contains("margin_call_level") && risk_config.contains("liquidation_level")) {
        double margin_call = risk_config["margin_call_level"];
        double liquidation = risk_config["liquidation_level"];
        
        if (liquidation >= margin_call) {
            result.errors.push_back(
                "liquidation_level must be less than margin_call_level"
            );
            result.is_valid = false;
        }
    }
    
    // Validate leverage
    if (risk_config.contains("max_leverage")) {
        double leverage = risk_config["max_leverage"];
        if (leverage < 1.0) {
            result.errors.push_back("max_leverage must be at least 1.0");
            result.is_valid = false;
        } else if (leverage > 50.0) {
            result.warnings.push_back(
                "Very high leverage detected: " + std::to_string(leverage) + 
                "x. Typical range is 1-50x"
            );
        }
    }
}

void ConfigValidator::validateExecutionConfig(const nlohmann::json& exec_config, ValidationResult& result) {
    // Validate order sizes
    if (exec_config.contains("min_order_size") && exec_config.contains("max_order_size")) {
        double min_size = exec_config["min_order_size"];
        double max_size = exec_config["max_order_size"];
        
        if (min_size <= 0) {
            result.errors.push_back("min_order_size must be positive");
            result.is_valid = false;
        }
        
        if (max_size <= 0) {
            result.errors.push_back("max_order_size must be positive");
            result.is_valid = false;
        }
        
        if (min_size > max_size) {
            result.errors.push_back("min_order_size cannot exceed max_order_size");
            result.is_valid = false;
        }
    }
    
    // Validate rate limits
    if (exec_config.contains("max_orders_per_second")) {
        int rate = exec_config["max_orders_per_second"];
        if (rate <= 0) {
            result.errors.push_back("max_orders_per_second must be positive");
            result.is_valid = false;
        } else if (rate > 10000) {
            result.warnings.push_back(
                "Very high order rate: " + std::to_string(rate) + 
                " orders/second. Ensure exchange can handle this rate"
            );
        }
    }
    
    // Validate timeout
    if (exec_config.contains("order_timeout_ms")) {
        int timeout = exec_config["order_timeout_ms"];
        if (timeout < 100) {
            result.warnings.push_back(
                "Order timeout very low: " + std::to_string(timeout) + 
                "ms. May cause premature cancellations"
            );
        }
    }
}

void ConfigValidator::validateMLConfig(const nlohmann::json& ml_config, ValidationResult& result) {
    // ML config is optional, only validate if present
    if (ml_config.contains("confidence_threshold")) {
        double threshold = ml_config["confidence_threshold"];
        if (threshold < 0 || threshold > 1) {
            result.errors.push_back("ML confidence_threshold must be between 0 and 1");
            result.is_valid = false;
        }
    }
    
    if (ml_config.contains("feature_window_size")) {
        int window = ml_config["feature_window_size"];
        if (window <= 0) {
            result.errors.push_back("ML feature_window_size must be positive");
            result.is_valid = false;
        }
    }
    
    if (ml_config.contains("model_checkpoint_dir")) {
        std::string dir = ml_config["model_checkpoint_dir"];
        if (dir.empty()) {
            result.warnings.push_back("ML model_checkpoint_dir is empty");
        }
    }
}

void ConfigValidator::validateCrossSectionConsistency(const nlohmann::json& config, 
                                                    ValidationResult& result) {
    // Check execution order size vs risk limits
    if (config.contains("execution") && config.contains("risk_limits")) {
        const auto& exec = config["execution"];
        const auto& risk = config["risk_limits"];
        
        if (exec.contains("max_order_size") && risk.contains("max_order_size")) {
            double exec_max = exec["max_order_size"];
            double risk_max = risk["max_order_size"];
            
            if (exec_max > risk_max) {
                result.errors.push_back(
                    "Execution max_order_size exceeds risk limit max_order_size"
                );
                result.is_valid = false;
            }
        }
    }
    
    // Check order rate consistency
    if (config.contains("execution") && config.contains("risk_limits")) {
        const auto& exec = config["execution"];
        const auto& risk = config["risk_limits"];
        
        if (exec.contains("max_orders_per_second") && risk.contains("max_orders_per_second")) {
            int exec_rate = exec["max_orders_per_second"];
            int risk_rate = risk["max_orders_per_second"];
            
            if (exec_rate > risk_rate) {
                result.warnings.push_back(
                    "Execution order rate exceeds risk limit order rate"
                );
            }
        }
    }
}

} // namespace hft