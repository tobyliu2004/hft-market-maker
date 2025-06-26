#include "models/python_bridge.h"
#include "core/order_book.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace hft {

StatisticalModels::StatisticalModels() 
    : initialized_(false), 
      models_module_(nullptr),
      feature_extractor_(nullptr),
      adverse_detector_(nullptr),
      volatility_estimator_(nullptr),
      price_predictor_(nullptr),
      cointegration_analyzer_(nullptr) {
}

StatisticalModels::~StatisticalModels() {
    shutdown();
}

bool StatisticalModels::initialize() {
    std::lock_guard<std::mutex> lock(python_mutex_);
    
    if (initialized_) {
        return true;
    }
    
    // Initialize Python interpreter
    Py_Initialize();
    
    // Add Python module path
    PyObject* sys_path = PySys_GetObject("path");
    PyObject* path = PyUnicode_FromString("./python/models");
    PyList_Append(sys_path, path);
    Py_DECREF(path);
    
    // Import statistical models module
    models_module_ = PyImport_ImportModule("statistical_models");
    if (!models_module_) {
        PyErr_Print();
        std::cerr << "Failed to import statistical_models module" << std::endl;
        return false;
    }
    
    // Create model instances
    feature_extractor_ = PyObject_CallMethod(models_module_, "create_feature_extractor", nullptr);
    if (!feature_extractor_) {
        PyErr_Print();
        std::cerr << "Failed to create feature extractor" << std::endl;
        return false;
    }
    
    adverse_detector_ = PyObject_CallMethod(models_module_, "create_adverse_selection_detector", "i", 1000);
    if (!adverse_detector_) {
        PyErr_Print();
        std::cerr << "Failed to create adverse selection detector" << std::endl;
        return false;
    }
    
    volatility_estimator_ = PyObject_CallMethod(models_module_, "create_volatility_estimator", nullptr);
    if (!volatility_estimator_) {
        PyErr_Print();
        std::cerr << "Failed to create volatility estimator" << std::endl;
        return false;
    }
    
    price_predictor_ = PyObject_CallMethod(models_module_, "create_price_predictor", nullptr);
    if (!price_predictor_) {
        PyErr_Print();
        std::cerr << "Failed to create price predictor" << std::endl;
        return false;
    }
    
    cointegration_analyzer_ = PyObject_CallMethod(models_module_, "create_cointegration_analyzer", nullptr);
    if (!cointegration_analyzer_) {
        PyErr_Print();
        std::cerr << "Failed to create cointegration analyzer" << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

void StatisticalModels::shutdown() {
    std::lock_guard<std::mutex> lock(python_mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Clean up Python objects
    Py_XDECREF(feature_extractor_);
    Py_XDECREF(adverse_detector_);
    Py_XDECREF(volatility_estimator_);
    Py_XDECREF(price_predictor_);
    Py_XDECREF(cointegration_analyzer_);
    Py_XDECREF(models_module_);
    
    // Finalize Python interpreter
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
    
    initialized_ = false;
}

PyObject* StatisticalModels::convertOrderBookToPython(const OrderBook& book) {
    // Create Python dictionary with order book data
    PyObject* dict = PyDict_New();
    
    // Add basic prices
    PyDict_SetItemString(dict, "best_bid", PyFloat_FromDouble(book.getBestBid()));
    PyDict_SetItemString(dict, "best_ask", PyFloat_FromDouble(book.getBestAsk()));
    
    // Get bid and ask levels
    auto bid_levels = book.getBidLevels(10);
    auto ask_levels = book.getAskLevels(10);
    
    // Convert bid prices and sizes to Python lists
    PyObject* bid_prices = PyList_New(bid_levels.size());
    PyObject* bid_sizes = PyList_New(bid_levels.size());
    
    for (size_t i = 0; i < bid_levels.size(); ++i) {
        PyList_SetItem(bid_prices, i, PyFloat_FromDouble(bid_levels[i].first));
        PyList_SetItem(bid_sizes, i, PyLong_FromLong(bid_levels[i].second));
    }
    
    // Convert ask prices and sizes to Python lists
    PyObject* ask_prices = PyList_New(ask_levels.size());
    PyObject* ask_sizes = PyList_New(ask_levels.size());
    
    for (size_t i = 0; i < ask_levels.size(); ++i) {
        PyList_SetItem(ask_prices, i, PyFloat_FromDouble(ask_levels[i].first));
        PyList_SetItem(ask_sizes, i, PyLong_FromLong(ask_levels[i].second));
    }
    
    // Convert to numpy arrays
    PyObject* numpy = PyImport_ImportModule("numpy");
    if (numpy) {
        PyObject* array_func = PyObject_GetAttrString(numpy, "array");
        
        PyObject* bid_prices_array = PyObject_CallFunctionObjArgs(array_func, bid_prices, nullptr);
        PyObject* bid_sizes_array = PyObject_CallFunctionObjArgs(array_func, bid_sizes, nullptr);
        PyObject* ask_prices_array = PyObject_CallFunctionObjArgs(array_func, ask_prices, nullptr);
        PyObject* ask_sizes_array = PyObject_CallFunctionObjArgs(array_func, ask_sizes, nullptr);
        
        PyDict_SetItemString(dict, "bid_prices", bid_prices_array);
        PyDict_SetItemString(dict, "bid_sizes", bid_sizes_array);
        PyDict_SetItemString(dict, "ask_prices", ask_prices_array);
        PyDict_SetItemString(dict, "ask_sizes", ask_sizes_array);
        
        Py_DECREF(bid_prices_array);
        Py_DECREF(bid_sizes_array);
        Py_DECREF(ask_prices_array);
        Py_DECREF(ask_sizes_array);
        Py_DECREF(array_func);
        Py_DECREF(numpy);
    } else {
        PyDict_SetItemString(dict, "bid_prices", bid_prices);
        PyDict_SetItemString(dict, "bid_sizes", bid_sizes);
        PyDict_SetItemString(dict, "ask_prices", ask_prices);
        PyDict_SetItemString(dict, "ask_sizes", ask_sizes);
    }
    
    Py_DECREF(bid_prices);
    Py_DECREF(bid_sizes);
    Py_DECREF(ask_prices);
    Py_DECREF(ask_sizes);
    
    return dict;
}

std::unordered_map<std::string, double> StatisticalModels::extractPythonDict(PyObject* dict) {
    std::unordered_map<std::string, double> result;
    
    if (!PyDict_Check(dict)) {
        return result;
    }
    
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    
    while (PyDict_Next(dict, &pos, &key, &value)) {
        // Get key as string
        const char* key_str = PyUnicode_AsUTF8(key);
        if (!key_str) continue;
        
        // Get value as double
        double val = PyFloat_AsDouble(value);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            continue;
        }
        
        result[key_str] = val;
    }
    
    return result;
}

StatisticalModels::MicrostructureFeatures StatisticalModels::extractFeatures(const OrderBook& book) {
    MicrostructureFeatures features;
    
    if (!initialized_) {
        return features;
    }
    
    PythonGIL gil;
    
    // Convert order book to Python format
    PythonObject book_dict(convertOrderBookToPython(book));
    if (!book_dict) {
        return features;
    }
    
    // Call Python feature extraction
    PythonObject result(PyObject_CallMethod(
        feature_extractor_, 
        "extract_features", 
        "O", 
        book_dict.get()
    ));
    
    if (!result || !PyDict_Check(result.get())) {
        PyErr_Print();
        return features;
    }
    
    // Extract features from result
    auto feature_map = extractPythonDict(result.get());
    
    // Map to struct fields
    features.spread = feature_map["spread"];
    features.relative_spread = feature_map["relative_spread"];
    features.microprice = feature_map["microprice"];
    features.microprice_deviation = feature_map["microprice_deviation"];
    features.order_flow_imbalance = feature_map["ofi_5"];
    features.book_pressure = feature_map["book_pressure"];
    features.kyle_lambda = feature_map.count("kyle_lambda") ? feature_map["kyle_lambda"] : 0.0;
    features.volume_imbalance = feature_map["volume_imbalance"];
    
    // Store additional features
    for (const auto& [key, value] : feature_map) {
        if (key != "spread" && key != "relative_spread" && key != "microprice" &&
            key != "microprice_deviation" && key != "order_flow_imbalance" &&
            key != "book_pressure" && key != "kyle_lambda" && key != "volume_imbalance") {
            features.additional_features[key] = value;
        }
    }
    
    return features;
}

StatisticalModels::ToxicityResult StatisticalModels::detectToxicFlow(
    const std::vector<OrderBook>& recent_books,
    const std::vector<std::pair<double, double>>& recent_trades) {
    
    ToxicityResult result;
    result.score = 0.0;
    result.is_toxic = false;
    
    if (!initialized_ || recent_trades.empty()) {
        return result;
    }
    
    PythonGIL gil;
    
    // Create trades DataFrame-like structure
    PyObject* trades_list = PyList_New(recent_trades.size());
    
    for (size_t i = 0; i < recent_trades.size(); ++i) {
        PyObject* trade_dict = PyDict_New();
        PyDict_SetItemString(trade_dict, "price", PyFloat_FromDouble(recent_trades[i].first));
        PyDict_SetItemString(trade_dict, "size", PyFloat_FromDouble(recent_trades[i].second));
        PyDict_SetItemString(trade_dict, "side", PyUnicode_FromString(recent_trades[i].second > 0 ? "buy" : "sell"));
        PyList_SetItem(trades_list, i, trade_dict);
    }
    
    // Convert order book snapshots
    PyObject* snapshots_list = PyList_New(recent_books.size());
    
    for (size_t i = 0; i < recent_books.size(); ++i) {
        PyList_SetItem(snapshots_list, i, convertOrderBookToPython(recent_books[i]));
    }
    
    // Call toxic flow detection
    PythonObject py_result(PyObject_CallMethod(
        adverse_detector_,
        "detect_toxic_flow",
        "OO",
        trades_list,
        snapshots_list
    ));
    
    Py_DECREF(trades_list);
    Py_DECREF(snapshots_list);
    
    if (!py_result || !PyTuple_Check(py_result.get()) || PyTuple_Size(py_result.get()) != 2) {
        PyErr_Print();
        return result;
    }
    
    // Extract toxicity score
    PyObject* score_obj = PyTuple_GetItem(py_result.get(), 0);
    result.score = PyFloat_AsDouble(score_obj);
    
    // Extract flags
    PyObject* flags_obj = PyTuple_GetItem(py_result.get(), 1);
    if (PyList_Check(flags_obj)) {
        Py_ssize_t num_flags = PyList_Size(flags_obj);
        for (Py_ssize_t i = 0; i < num_flags; ++i) {
            PyObject* flag = PyList_GetItem(flags_obj, i);
            const char* flag_str = PyUnicode_AsUTF8(flag);
            if (flag_str) {
                result.flags.push_back(flag_str);
            }
        }
    }
    
    // Set binary decision
    result.is_toxic = result.score > 0.5;
    
    return result;
}

StatisticalModels::VolatilityEstimates StatisticalModels::estimateVolatility(
    const std::vector<double>& prices,
    const std::vector<double>& high_prices,
    const std::vector<double>& low_prices) {
    
    VolatilityEstimates estimates;
    estimates.realized_vol = 0.0;
    estimates.garch_vol = 0.0;
    estimates.parkinson_vol = 0.0;
    estimates.yang_zhang_vol = 0.0;
    estimates.forecast_1min = 0.0;
    estimates.forecast_5min = 0.0;
    
    if (!initialized_ || prices.empty()) {
        return estimates;
    }
    
    PythonGIL gil;
    
    // Convert price vectors to numpy arrays
    PyObject* numpy = PyImport_ImportModule("numpy");
    if (!numpy) {
        return estimates;
    }
    
    PyObject* array_func = PyObject_GetAttrString(numpy, "array");
    
    // Create Python lists
    PyObject* py_prices = PyList_New(prices.size());
    for (size_t i = 0; i < prices.size(); ++i) {
        PyList_SetItem(py_prices, i, PyFloat_FromDouble(prices[i]));
    }
    
    // Convert to numpy array
    PythonObject prices_array(PyObject_CallFunctionObjArgs(array_func, py_prices, nullptr));
    Py_DECREF(py_prices);
    
    // Calculate realized volatility
    PythonObject realized_result(PyObject_CallMethod(
        volatility_estimator_,
        "calculate_realized_volatility",
        "O",
        prices_array.get()
    ));
    
    if (realized_result) {
        estimates.realized_vol = PyFloat_AsDouble(realized_result.get());
    }
    
    // Calculate returns for GARCH
    PyObject* diff_func = PyObject_GetAttrString(numpy, "diff");
    PyObject* log_func = PyObject_GetAttrString(numpy, "log");
    
    PythonObject log_prices(PyObject_CallFunctionObjArgs(log_func, prices_array.get(), nullptr));
    PythonObject returns(PyObject_CallFunctionObjArgs(diff_func, log_prices.get(), nullptr));
    
    // Calculate GARCH volatility
    PythonObject garch_result(PyObject_CallMethod(
        volatility_estimator_,
        "estimate_garch_volatility",
        "O",
        returns.get()
    ));
    
    if (garch_result) {
        estimates.garch_vol = PyFloat_AsDouble(garch_result.get());
    }
    
    // If high/low prices available, calculate Parkinson volatility
    if (!high_prices.empty() && high_prices.size() == prices.size()) {
        PyObject* py_high = PyList_New(high_prices.size());
        PyObject* py_low = PyList_New(low_prices.size());
        
        for (size_t i = 0; i < high_prices.size(); ++i) {
            PyList_SetItem(py_high, i, PyFloat_FromDouble(high_prices[i]));
            PyList_SetItem(py_low, i, PyFloat_FromDouble(low_prices[i]));
        }
        
        PythonObject high_array(PyObject_CallFunctionObjArgs(array_func, py_high, nullptr));
        PythonObject low_array(PyObject_CallFunctionObjArgs(array_func, py_low, nullptr));
        
        Py_DECREF(py_high);
        Py_DECREF(py_low);
        
        PythonObject parkinson_result(PyObject_CallMethod(
            volatility_estimator_,
            "calculate_parkinson_volatility",
            "OO",
            high_array.get(),
            low_array.get()
        ));
        
        if (parkinson_result) {
            estimates.parkinson_vol = PyFloat_AsDouble(parkinson_result.get());
        }
    }
    
    // Simple forecasts based on current estimates
    double avg_vol = (estimates.realized_vol + estimates.garch_vol) / 2.0;
    estimates.forecast_1min = avg_vol * std::sqrt(1.0 / 390.0);  // 1 min / 390 min trading day
    estimates.forecast_5min = avg_vol * std::sqrt(5.0 / 390.0);
    
    Py_DECREF(array_func);
    Py_DECREF(diff_func);
    Py_DECREF(log_func);
    Py_DECREF(numpy);
    
    return estimates;
}

StatisticalModels::PricePrediction StatisticalModels::predictPrice(
    const OrderBook& current_book,
    const std::vector<OrderBook>& historical_books) {
    
    PricePrediction prediction;
    prediction.expected_return = 0.0;
    prediction.confidence = 0.0;
    prediction.prediction_horizon_ms = 1000.0;  // 1 second default
    
    if (!initialized_) {
        return prediction;
    }
    
    PythonGIL gil;
    
    // Convert current order book
    PythonObject current_snapshot(convertOrderBookToPython(current_book));
    
    // For now, return a simple prediction based on order flow imbalance
    // In production, this would use the trained ML model
    auto features = extractFeatures(current_book);
    
    // Simple linear model for demonstration
    prediction.expected_return = 
        0.3 * features.order_flow_imbalance +
        0.2 * features.microprice_deviation +
        0.1 * features.book_pressure;
    
    // Scale to reasonable return range
    prediction.expected_return = std::tanh(prediction.expected_return * 10) * 0.0001;  // Max 1 bp
    
    // Confidence based on spread and volatility
    double spread_factor = std::exp(-features.relative_spread * 1000);
    prediction.confidence = std::min(0.9, spread_factor);
    
    // Feature contributions
    prediction.feature_contributions["order_flow_imbalance"] = 0.3 * features.order_flow_imbalance;
    prediction.feature_contributions["microprice_deviation"] = 0.2 * features.microprice_deviation;
    prediction.feature_contributions["book_pressure"] = 0.1 * features.book_pressure;
    
    return prediction;
}

void StatisticalModels::trainPricePredictor(const std::vector<OrderBook>& historical_books) {
    if (!initialized_ || historical_books.size() < 100) {
        return;
    }
    
    PythonGIL gil;
    
    // Convert historical books to Python format
    PyObject* snapshots_list = PyList_New(historical_books.size());
    
    for (size_t i = 0; i < historical_books.size(); ++i) {
        PyList_SetItem(snapshots_list, i, convertOrderBookToPython(historical_books[i]));
    }
    
    // Create empty trades DataFrame for now (would need actual trade data)
    PyObject* pandas = PyImport_ImportModule("pandas");
    if (!pandas) {
        Py_DECREF(snapshots_list);
        return;
    }
    
    PyObject* dataframe_class = PyObject_GetAttrString(pandas, "DataFrame");
    PythonObject empty_trades(PyObject_CallObject(dataframe_class, nullptr));
    
    // Call training method
    PyObject* result = PyObject_CallMethod(
        price_predictor_,
        "train",
        "OO",
        snapshots_list,
        empty_trades.get()
    );
    
    if (!result) {
        PyErr_Print();
    } else {
        Py_DECREF(result);
    }
    
    Py_DECREF(snapshots_list);
    Py_DECREF(dataframe_class);
    Py_DECREF(pandas);
}

// StatArbSignalGenerator Implementation
StatArbSignalGenerator::StatArbSignalGenerator() {
    models_.initialize();
}

std::vector<StatArbSignalGenerator::Signal> StatArbSignalGenerator::generateSignals(
    const std::unordered_map<std::string, OrderBook>& order_books,
    const MarketState& state) {
    
    std::vector<Signal> signals;
    
    for (const auto& [symbol, book] : order_books) {
        // Extract features
        auto features = models_.extractFeatures(book);
        
        // Mean reversion signal based on microprice deviation
        if (std::abs(features.microprice_deviation) > 0.0002) {  // 2 bps threshold
            Signal signal;
            signal.symbol = symbol;
            signal.type = "mean_reversion";
            signal.strength = std::min(1.0, std::abs(features.microprice_deviation) / 0.0005);
            
            if (features.microprice_deviation > 0) {
                // Microprice above mid - expect reversion down
                signal.size_recommendation = -1000;  // Sell
                signal.entry_price = book.getBestAsk();
                signal.target_price = book.getMidPrice();
                signal.stop_price = signal.entry_price * 1.002;  // 20 bps stop
            } else {
                // Microprice below mid - expect reversion up
                signal.size_recommendation = 1000;  // Buy
                signal.entry_price = book.getBestBid();
                signal.target_price = book.getMidPrice();
                signal.stop_price = signal.entry_price * 0.998;  // 20 bps stop
            }
            
            signal.rationale = "Microprice deviation suggests mean reversion opportunity";
            signals.push_back(signal);
        }
        
        // Order flow momentum signal
        if (std::abs(features.order_flow_imbalance) > 0.7) {
            Signal signal;
            signal.symbol = symbol;
            signal.type = "momentum";
            signal.strength = std::abs(features.order_flow_imbalance);
            
            if (features.order_flow_imbalance > 0) {
                // Strong buying pressure
                signal.size_recommendation = 1000;
                signal.entry_price = book.getBestAsk();
                signal.target_price = signal.entry_price * 1.001;  // 10 bps target
                signal.stop_price = signal.entry_price * 0.999;   // 10 bps stop
            } else {
                // Strong selling pressure
                signal.size_recommendation = -1000;
                signal.entry_price = book.getBestBid();
                signal.target_price = signal.entry_price * 0.999;  // 10 bps target
                signal.stop_price = signal.entry_price * 1.001;   // 10 bps stop
            }
            
            signal.rationale = "Strong order flow imbalance indicates directional pressure";
            signals.push_back(signal);
        }
    }
    
    return signals;
}

void StatArbSignalGenerator::updateSignalState(const std::string& symbol, 
                                               double executed_price, 
                                               double executed_size) {
    auto& state = active_signals_[symbol];
    state.position += executed_size;
    
    if (state.position == 0) {
        // Position closed
        active_signals_.erase(symbol);
    } else if (state.entry_price == 0) {
        // New position
        state.entry_price = executed_price;
        state.entry_time = std::chrono::high_resolution_clock::now();
    }
}

bool StatArbSignalGenerator::shouldClosePosition(const std::string& symbol, 
                                                 double current_price) {
    auto it = active_signals_.find(symbol);
    if (it == active_signals_.end()) {
        return false;
    }
    
    const auto& state = it->second;
    
    // Check stop loss
    if (state.position > 0 && current_price <= state.stop_loss) {
        return true;
    } else if (state.position < 0 && current_price >= state.stop_loss) {
        return true;
    }
    
    // Check take profit
    if (state.position > 0 && current_price >= state.take_profit) {
        return true;
    } else if (state.position < 0 && current_price <= state.take_profit) {
        return true;
    }
    
    // Time-based exit (hold for max 5 minutes)
    auto hold_time = std::chrono::high_resolution_clock::now() - state.entry_time;
    if (std::chrono::duration_cast<std::chrono::minutes>(hold_time).count() > 5) {
        return true;
    }
    
    return false;
}

double StatArbSignalGenerator::calculatePositionSize(const Signal& signal, 
                                                    double account_equity) {
    // Kelly criterion-based position sizing
    double kelly_fraction = 0.25;  // Conservative Kelly (1/4 Kelly)
    
    // Adjust for signal strength
    double adjusted_fraction = kelly_fraction * signal.strength;
    
    // Risk per trade (max 2% of equity)
    double max_risk = account_equity * 0.02;
    
    // Calculate position size based on stop distance
    double stop_distance = std::abs(signal.entry_price - signal.stop_price);
    double position_value = max_risk / stop_distance;
    
    // Apply Kelly adjustment
    position_value *= adjusted_fraction;
    
    // Convert to shares (assuming price is per share)
    double shares = position_value / signal.entry_price;
    
    // Round to lot size (100 shares)
    return std::floor(shares / 100) * 100;
}

} // namespace hft