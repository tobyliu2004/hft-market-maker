#include <gtest/gtest.h>
#include "utils/config_validator.h"
#include <nlohmann/json.hpp>

using namespace hft;
using json = nlohmann::json;

class ConfigValidatorTest : public ::testing::Test {
protected:
    json createValidConfig() {
        return json::parse(R"({
            "fix": {
                "config_file": "config/fix_config.cfg",
                "sender_comp_id": "MARKET_MAKER",
                "target_comp_id": "EXCHANGE",
                "heartbeat_interval": 30
            },
            "market_data": {
                "feed_type": "polygon",
                "polygon_api_key": "valid_api_key_12345",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "market_depth": 10,
                "snapshot_interval_ms": 100
            },
            "strategy": {
                "type": "avellaneda_stoikov",
                "risk_aversion": 0.1,
                "order_arrival_rate": 10.0,
                "time_horizon": 300.0,
                "enable_alpha_signals": true,
                "alpha_weights": {
                    "microprice": 0.3,
                    "order_flow": 0.25,
                    "queue": 0.2,
                    "spread": 0.25
                }
            },
            "risk_limits": {
                "max_position_value": 100000.0,
                "max_total_exposure": 500000.0,
                "max_position_count": 20,
                "max_daily_loss": 10000.0,
                "max_drawdown": 0.05,
                "stop_loss_percent": 0.02,
                "max_order_size": 1000.0,
                "max_order_value": 50000.0,
                "max_orders_per_second": 100,
                "max_open_orders": 50,
                "max_spread_percent": 0.005,
                "min_liquidity_ratio": 0.1,
                "max_market_impact": 0.001,
                "max_delta": 10000.0,
                "max_gamma": 1000.0,
                "max_vega": 5000.0,
                "max_theta": 1000.0,
                "max_leverage": 3.0,
                "margin_call_level": 0.3,
                "liquidation_level": 0.25
            },
            "execution": {
                "min_order_size": 100.0,
                "max_order_size": 1000.0,
                "max_orders_per_second": 100,
                "order_timeout_ms": 5000,
                "enable_smart_routing": true,
                "enable_iceberg_orders": true,
                "iceberg_display_size": 100,
                "enable_post_only": true,
                "cancel_on_disconnect": true
            },
            "machine_learning": {
                "enable_ml_predictions": true,
                "model_update_interval_minutes": 60,
                "feature_window_size": 100,
                "prediction_horizon_ms": 1000,
                "confidence_threshold": 0.7,
                "enable_online_learning": true,
                "model_checkpoint_dir": "models/checkpoints"
            }
        })");
    }
};

TEST_F(ConfigValidatorTest, ValidConfigPasses) {
    auto config = createValidConfig();
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(ConfigValidatorTest, MissingRequiredSections) {
    auto config = createValidConfig();
    config.erase("fix");
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_FALSE(result.errors.empty());
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("Missing required 'fix'") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, InvalidFeedType) {
    auto config = createValidConfig();
    config["market_data"]["feed_type"] = "invalid_feed";
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("Invalid feed_type") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, MissingPolygonApiKey) {
    auto config = createValidConfig();
    config["market_data"]["polygon_api_key"] = "YOUR_POLYGON_API_KEY";
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("Invalid or missing Polygon API key") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, InvalidSymbolFormat) {
    auto config = createValidConfig();
    config["market_data"]["symbols"] = json::array({"AAPL", "INVALID123", "TOOLONGSYMBOL"});
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_TRUE(result.is_valid);  // Warnings don't make config invalid
    EXPECT_FALSE(result.warnings.empty());
    EXPECT_TRUE(std::any_of(result.warnings.begin(), result.warnings.end(),
        [](const std::string& warning) {
            return warning.find("Symbol format warning") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, InvalidStrategyType) {
    auto config = createValidConfig();
    config["strategy"]["type"] = "unknown_strategy";
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("Unknown strategy type") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, NegativeOrderArrivalRate) {
    auto config = createValidConfig();
    config["strategy"]["order_arrival_rate"] = -5.0;
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("Order arrival rate must be positive") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, AlphaWeightsValidation) {
    auto config = createValidConfig();
    config["strategy"]["alpha_weights"]["microprice"] = 0.5;
    config["strategy"]["alpha_weights"]["order_flow"] = 0.4;
    config["strategy"]["alpha_weights"]["queue"] = 0.3;
    config["strategy"]["alpha_weights"]["spread"] = 0.2;
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_FALSE(result.warnings.empty());
    EXPECT_TRUE(std::any_of(result.warnings.begin(), result.warnings.end(),
        [](const std::string& warning) {
            return warning.find("Alpha weights should sum to 1.0") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, PositionValueExceedsExposure) {
    auto config = createValidConfig();
    config["risk_limits"]["max_position_value"] = 600000.0;
    config["risk_limits"]["max_total_exposure"] = 500000.0;
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("max_position_value cannot exceed max_total_exposure") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, InvalidPercentageLimits) {
    auto config = createValidConfig();
    config["risk_limits"]["max_drawdown"] = 1.5;  // 150% - invalid
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("max_drawdown must be between") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, MarginLevelValidation) {
    auto config = createValidConfig();
    config["risk_limits"]["margin_call_level"] = 0.2;
    config["risk_limits"]["liquidation_level"] = 0.25;  // Higher than margin call
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("liquidation_level must be less than margin_call_level") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, OrderSizeConsistency) {
    auto config = createValidConfig();
    config["execution"]["min_order_size"] = 1000.0;
    config["execution"]["max_order_size"] = 500.0;  // Less than min
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("min_order_size cannot exceed max_order_size") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, NegativeOrderSize) {
    auto config = createValidConfig();
    config["execution"]["min_order_size"] = -100.0;
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("min_order_size must be positive") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, MLConfidenceThreshold) {
    auto config = createValidConfig();
    config["machine_learning"]["confidence_threshold"] = 1.5;  // > 1.0
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_TRUE(std::any_of(result.errors.begin(), result.errors.end(),
        [](const std::string& error) {
            return error.find("ML confidence_threshold must be between 0 and 1") != std::string::npos;
        }));
}

TEST_F(ConfigValidatorTest, WarningsDoNotInvalidate) {
    auto config = createValidConfig();
    
    // Add values that generate warnings but not errors
    config["fix"]["heartbeat_interval"] = 400;  // Warning: should be between 1-300
    config["market_data"]["market_depth"] = 100;  // Warning: should be between 1-50
    config["risk_limits"]["max_leverage"] = 100;  // Warning: typically between 1-50
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_FALSE(result.warnings.empty());
    EXPECT_GE(result.warnings.size(), 3);
}

TEST_F(ConfigValidatorTest, MissingOptionalSections) {
    auto config = createValidConfig();
    config.erase("machine_learning");  // ML config is optional
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(ConfigValidatorTest, ComplexValidationScenario) {
    auto config = createValidConfig();
    
    // Multiple errors and warnings
    config.erase("fix");  // Missing required section
    config["market_data"]["feed_type"] = "invalid";  // Invalid feed type
    config["strategy"]["risk_aversion"] = -0.5;  // Should be positive (warning)
    config["risk_limits"]["max_drawdown"] = 2.0;  // Invalid percentage
    config["execution"]["min_order_size"] = 2000.0;
    config["execution"]["max_order_size"] = 1000.0;  // Min > Max
    
    auto result = ConfigValidator::validateConfig(config);
    
    EXPECT_FALSE(result.is_valid);
    EXPECT_GE(result.errors.size(), 4);
    EXPECT_FALSE(result.warnings.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}