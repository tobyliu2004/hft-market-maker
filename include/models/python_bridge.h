#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace hft {

// Forward declarations
struct OrderBook;
struct MarketState;

// Python Global Interpreter Lock (GIL) management
class PythonGIL {
private:
    PyGILState_STATE state_;
public:
    PythonGIL() : state_(PyGILState_Ensure()) {}
    ~PythonGIL() { PyGILState_Release(state_); }
    
    // Delete copy/move constructors
    PythonGIL(const PythonGIL&) = delete;
    PythonGIL& operator=(const PythonGIL&) = delete;
};

// Python object wrapper for RAII
class PythonObject {
private:
    PyObject* obj_;
    
public:
    explicit PythonObject(PyObject* obj = nullptr) : obj_(obj) {}
    ~PythonObject() { Py_XDECREF(obj_); }
    
    // Move semantics
    PythonObject(PythonObject&& other) noexcept : obj_(other.obj_) {
        other.obj_ = nullptr;
    }
    
    PythonObject& operator=(PythonObject&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(obj_);
            obj_ = other.obj_;
            other.obj_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy constructor/assignment
    PythonObject(const PythonObject&) = delete;
    PythonObject& operator=(const PythonObject&) = delete;
    
    PyObject* get() const { return obj_; }
    PyObject* release() {
        PyObject* temp = obj_;
        obj_ = nullptr;
        return temp;
    }
    
    operator bool() const { return obj_ != nullptr; }
};

// Statistical models interface to Python
class StatisticalModels {
private:
    bool initialized_;
    PyObject* models_module_;
    PyObject* feature_extractor_;
    PyObject* adverse_detector_;
    PyObject* volatility_estimator_;
    PyObject* price_predictor_;
    PyObject* cointegration_analyzer_;
    
    std::mutex python_mutex_;  // Ensure thread safety
    
    // Helper methods
    PyObject* convertOrderBookToPython(const OrderBook& book);
    std::unordered_map<std::string, double> extractPythonDict(PyObject* dict);
    
public:
    StatisticalModels();
    ~StatisticalModels();
    
    // Initialize Python interpreter and load modules
    bool initialize();
    void shutdown();
    
    // Feature extraction
    struct MicrostructureFeatures {
        double spread;
        double relative_spread;
        double microprice;
        double microprice_deviation;
        double order_flow_imbalance;
        double book_pressure;
        double kyle_lambda;
        double volume_imbalance;
        std::unordered_map<std::string, double> additional_features;
    };
    
    MicrostructureFeatures extractFeatures(const OrderBook& book);
    
    // Adverse selection detection
    struct ToxicityResult {
        double score;          // 0-1 toxicity score
        std::vector<std::string> flags;  // Specific warnings
        bool is_toxic;        // Binary decision
    };
    
    ToxicityResult detectToxicFlow(const std::vector<OrderBook>& recent_books,
                                   const std::vector<std::pair<double, double>>& recent_trades);
    
    // Volatility estimation
    struct VolatilityEstimates {
        double realized_vol;
        double garch_vol;
        double parkinson_vol;
        double yang_zhang_vol;
        double forecast_1min;
        double forecast_5min;
    };
    
    VolatilityEstimates estimateVolatility(const std::vector<double>& prices,
                                          const std::vector<double>& high_prices,
                                          const std::vector<double>& low_prices);
    
    // Price prediction
    struct PricePrediction {
        double expected_return;
        double confidence;
        double prediction_horizon_ms;
        std::unordered_map<std::string, double> feature_contributions;
    };
    
    PricePrediction predictPrice(const OrderBook& current_book,
                                const std::vector<OrderBook>& historical_books);
    
    // Cointegration analysis for pairs
    struct PairAnalysis {
        std::string symbol1;
        std::string symbol2;
        double hedge_ratio;
        double spread_mean;
        double spread_std;
        double half_life;
        double z_score;
        std::string signal;  // "long_spread", "short_spread", "close", "hold"
    };
    
    std::vector<PairAnalysis> analyzePairs(
        const std::unordered_map<std::string, std::vector<double>>& price_series);
    
    // Model training and updating
    void trainPricePredictor(const std::vector<OrderBook>& historical_books);
    void updateVolatilityModel(const std::vector<double>& recent_returns);
    
    // Performance metrics
    struct ModelPerformance {
        double prediction_accuracy;
        double sharpe_ratio;
        double hit_rate;
        double average_edge;
        std::unordered_map<std::string, double> feature_importance;
    };
    
    ModelPerformance evaluatePerformance();
};

// High-level statistical arbitrage signals
class StatArbSignalGenerator {
private:
    StatisticalModels models_;
    
    struct SignalState {
        double position;
        double entry_price;
        double stop_loss;
        double take_profit;
        std::chrono::high_resolution_clock::time_point entry_time;
    };
    
    std::unordered_map<std::string, SignalState> active_signals_;
    
public:
    StatArbSignalGenerator();
    
    struct Signal {
        std::string symbol;
        std::string type;  // "mean_reversion", "momentum", "pairs", etc.
        double strength;   // Signal strength (0-1)
        double size_recommendation;
        double entry_price;
        double target_price;
        double stop_price;
        std::string rationale;
    };
    
    // Generate signals based on current market state
    std::vector<Signal> generateSignals(
        const std::unordered_map<std::string, OrderBook>& order_books,
        const MarketState& state);
    
    // Update signal states with executions
    void updateSignalState(const std::string& symbol, double executed_price, 
                          double executed_size);
    
    // Risk management
    bool shouldClosePosition(const std::string& symbol, double current_price);
    double calculatePositionSize(const Signal& signal, double account_equity);
};

// Integration with QuantLib for options and derivatives
class QuantLibIntegration {
private:
    // Cache for option pricing
    struct OptionCache {
        double spot;
        double strike;
        double volatility;
        double time_to_expiry;
        double risk_free_rate;
        double price;
        double delta;
        double gamma;
        double vega;
        double theta;
        double rho;
        std::chrono::high_resolution_clock::time_point last_update;
    };
    
    std::unordered_map<std::string, OptionCache> option_cache_;
    mutable std::mutex cache_mutex_;
    
public:
    QuantLibIntegration();
    
    struct OptionGreeks {
        double price;
        double delta;
        double gamma;
        double vega;
        double theta;
        double rho;
    };
    
    // Black-Scholes option pricing
    OptionGreeks calculateOptionGreeks(
        double spot,
        double strike,
        double volatility,
        double time_to_expiry,
        double risk_free_rate,
        bool is_call
    );
    
    // Implied volatility calculation
    double calculateImpliedVolatility(
        double option_price,
        double spot,
        double strike,
        double time_to_expiry,
        double risk_free_rate,
        bool is_call
    );
    
    // Volatility surface interpolation
    double interpolateVolatilitySurface(
        double strike,
        double time_to_expiry,
        const std::vector<std::pair<double, double>>& strike_vol_pairs
    );
    
    // Risk metrics for options portfolio
    struct PortfolioRisk {
        double total_delta;
        double total_gamma;
        double total_vega;
        double total_theta;
        double var_95;
        double expected_shortfall;
    };
    
    PortfolioRisk calculatePortfolioRisk(
        const std::unordered_map<std::string, double>& positions,
        const std::unordered_map<std::string, OptionGreeks>& greeks
    );
};

} // namespace hft