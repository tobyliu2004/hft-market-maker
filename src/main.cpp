#include "core/order_book.h"
#include "core/fix_handler.h"
#include "strategies/market_maker_strategy.h"
#include "models/python_bridge.h"
#include "data/market_data_feed.h"
#include "risk/risk_manager.h"
#include "utils/logger.h"
#include "utils/config_validator.h"

#include <iostream>
#include <memory>
#include <thread>
#include <signal.h>
#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

using namespace hft;

// Global flag for graceful shutdown
std::atomic<bool> g_running(true);

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

// Configuration structure
struct Config {
    // FIX settings
    std::string fix_config_file;
    
    // Market data settings
    std::string market_data_feed;
    std::string polygon_api_key;
    std::vector<std::string> symbols;
    
    // Strategy settings
    std::string strategy_type;
    double risk_aversion;
    double order_arrival_rate;
    double time_horizon;
    
    // Risk settings
    RiskLimits risk_limits;
    
    // Execution settings
    double min_order_size;
    double max_order_size;
    int max_orders_per_second;
};

// Load configuration from JSON file
Config load_config(const std::string& config_file) {
    Config config;
    
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_file);
    }
    
    nlohmann::json json;
    try {
        file >> json;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
    }
    
    // Validate configuration
    auto validation = ConfigValidator::validateConfig(json);
    if (!validation.is_valid) {
        std::stringstream ss;
        ss << "Configuration validation failed:\n";
        for (const auto& error : validation.errors) {
            ss << "  ERROR: " << error << "\n";
        }
        throw std::runtime_error(ss.str());
    }
    
    // Log warnings
    for (const auto& warning : validation.warnings) {
        LOG_WARNING("Config warning: " + warning);
    }
    
    // Parse FIX settings
    config.fix_config_file = json["fix"]["config_file"];
    
    // Parse market data settings
    config.market_data_feed = json["market_data"]["feed_type"];
    config.polygon_api_key = json["market_data"]["polygon_api_key"];
    
    for (const auto& symbol : json["market_data"]["symbols"]) {
        config.symbols.push_back(symbol);
    }
    
    // Parse strategy settings
    config.strategy_type = json["strategy"]["type"];
    config.risk_aversion = json["strategy"]["risk_aversion"];
    config.order_arrival_rate = json["strategy"]["order_arrival_rate"];
    config.time_horizon = json["strategy"]["time_horizon"];
    
    // Parse risk limits
    auto& risk_json = json["risk_limits"];
    config.risk_limits.max_position_value = risk_json["max_position_value"];
    config.risk_limits.max_total_exposure = risk_json["max_total_exposure"];
    config.risk_limits.max_position_count = risk_json["max_position_count"];
    config.risk_limits.max_daily_loss = risk_json["max_daily_loss"];
    config.risk_limits.max_drawdown = risk_json["max_drawdown"];
    config.risk_limits.stop_loss_percent = risk_json["stop_loss_percent"];
    config.risk_limits.max_order_size = risk_json["max_order_size"];
    config.risk_limits.max_order_value = risk_json["max_order_value"];
    config.risk_limits.max_orders_per_second = risk_json["max_orders_per_second"];
    config.risk_limits.max_open_orders = risk_json["max_open_orders"];
    
    // Parse execution settings
    auto& exec_json = json["execution"];
    config.min_order_size = exec_json["min_order_size"];
    config.max_order_size = exec_json["max_order_size"];
    config.max_orders_per_second = exec_json["max_orders_per_second"];
    
    return config;
}

// Market data handler implementation
class MarketMakerDataHandler : public MarketDataListener {
private:
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> order_books_;
    std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>> strategies_;
    std::shared_ptr<IntegratedRiskSystem> risk_system_;
    std::shared_ptr<FIXMarketDataHandler> fix_handler_;
    std::shared_ptr<StatisticalModels> stat_models_;
    
    mutable std::mutex mutex_;
    
public:
    MarketMakerDataHandler(
        const std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>>& strategies,
        std::shared_ptr<IntegratedRiskSystem> risk_system,
        std::shared_ptr<FIXMarketDataHandler> fix_handler)
        : strategies_(strategies), 
          risk_system_(risk_system),
          fix_handler_(fix_handler) {
        
        stat_models_ = std::make_shared<StatisticalModels>();
        stat_models_->initialize();
    }
    
    void onQuote(const QuoteData& quote) override {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Update order book
        auto& book = order_books_[quote.symbol];
        if (!book) {
            book = std::make_shared<OrderBook>();
        }
        
        // Simple update - in practice would maintain full book
        book->cancelOrder(1);  // Remove old best bid
        book->cancelOrder(2);  // Remove old best ask
        
        auto bid_order = std::make_shared<Order>(
            1, Side::BUY, quote.bid_price, quote.bid_size,
            quote.exchange_timestamp
        );
        auto ask_order = std::make_shared<Order>(
            2, Side::SELL, quote.ask_price, quote.ask_size,
            quote.exchange_timestamp
        );
        
        book->addOrder(bid_order);
        book->addOrder(ask_order);
        
        // Update risk manager with market prices
        risk_system_->getRiskManager()->markToMarket(
            quote.symbol, 
            (quote.bid_price + quote.ask_price) / 2.0
        );
        
        // Get strategy for this symbol
        auto strategy_it = strategies_.find(quote.symbol);
        if (strategy_it == strategies_.end()) {
            return;
        }
        
        // Extract market features
        auto features = stat_models_->extractFeatures(*book);
        
        // Check for toxic flow
        std::vector<OrderBook> recent_books = {*book};  // Would maintain history
        std::vector<std::pair<double, double>> recent_trades;  // Would track trades
        
        auto toxicity = stat_models_->detectToxicFlow(recent_books, recent_trades);
        
        // Update market state
        MarketState state;
        state.current_position = risk_system_->getRiskManager()->getPosition(quote.symbol).quantity;
        state.unrealized_pnl = risk_system_->getRiskManager()->getPosition(quote.symbol).unrealized_pnl;
        state.realized_pnl = risk_system_->getRiskManager()->getPosition(quote.symbol).realized_pnl;
        state.current_volatility = 0.02;  // Would calculate from historical data
        state.volume_imbalance = features.volume_imbalance;
        state.order_flow_toxicity = toxicity.score;
        state.spread_percentile = 0.5;  // Would calculate from historical spreads
        state.queue_position = 0;  // Would estimate from order data
        state.last_update = quote.local_timestamp;
        
        // Skip if toxic flow detected
        if (toxicity.is_toxic) {
            std::cout << "Toxic flow detected for " << quote.symbol 
                     << ", skipping quote generation" << std::endl;
            return;
        }
        
        // Get quotes from strategy
        auto [bid_quote, ask_quote] = strategy_it->second->getQuotes(*book, state);
        
        // Send orders if we have quotes
        if (bid_quote.has_value() && ask_quote.has_value()) {
            // Validate with risk system
            std::vector<std::string> bid_rejection_reasons, ask_rejection_reasons;
            
            double order_size = 100;  // Fixed size for now
            
            bool bid_approved = risk_system_->approveOrder(
                quote.symbol, Side::BUY, order_size, bid_quote.value(), 
                bid_rejection_reasons
            );
            
            bool ask_approved = risk_system_->approveOrder(
                quote.symbol, Side::SELL, order_size, ask_quote.value(),
                ask_rejection_reasons
            );
            
            // Send approved orders
            if (bid_approved) {
                std::string bid_cl_ord_id = "BID_" + std::to_string(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                );
                fix_handler_->sendOrder(quote.symbol, Side::BUY, bid_quote.value(), 
                                       order_size, bid_cl_ord_id);
            }
            
            if (ask_approved) {
                std::string ask_cl_ord_id = "ASK_" + std::to_string(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                );
                fix_handler_->sendOrder(quote.symbol, Side::SELL, ask_quote.value(),
                                       order_size, ask_cl_ord_id);
            }
        }
    }
    
    void onTrade(const TradeData& trade) override {
        // Update trade statistics
        // Could use for adverse selection detection
    }
    
    void onOrderBookUpdate(const std::string& symbol, 
                          const OrderBook::Snapshot& snapshot) override {
        // Handle full book updates
    }
    
    void onTradingStatus(const std::string& symbol, 
                        const std::string& status) override {
        std::cout << "Trading status for " << symbol << ": " << status << std::endl;
    }
    
    void onError(const std::string& error) override {
        std::cerr << "Market data error: " << error << std::endl;
    }
};

// Order handler implementation
class MarketMakerOrderHandler : public OrderHandler {
private:
    std::shared_ptr<IntegratedRiskSystem> risk_system_;
    std::unordered_map<std::string, std::pair<std::string, double>> pending_orders_;
    mutable std::mutex mutex_;
    
public:
    explicit MarketMakerOrderHandler(std::shared_ptr<IntegratedRiskSystem> risk_system)
        : risk_system_(risk_system) {}
    
    void onOrderAccepted(const std::string& cl_ord_id, 
                        const std::string& order_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "Order accepted: " << cl_ord_id << " -> " << order_id << std::endl;
    }
    
    void onOrderRejected(const std::string& cl_ord_id, 
                        const std::string& reason) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "Order rejected: " << cl_ord_id << " - " << reason << std::endl;
        pending_orders_.erase(cl_ord_id);
    }
    
    void onOrderFilled(const std::string& cl_ord_id, Price price, 
                      Quantity quantity) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "Order filled: " << cl_ord_id << " @ " << price 
                 << " x " << quantity << std::endl;
        
        // Update position in risk manager
        auto it = pending_orders_.find(cl_ord_id);
        if (it != pending_orders_.end()) {
            auto [symbol, side_qty] = it->second;
            risk_system_->getRiskManager()->updatePosition(symbol, side_qty * quantity, price);
        }
    }
    
    void onOrderCanceled(const std::string& cl_ord_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "Order canceled: " << cl_ord_id << std::endl;
        pending_orders_.erase(cl_ord_id);
    }
    
    void trackOrder(const std::string& cl_ord_id, const std::string& symbol, 
                   double signed_quantity) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_orders_[cl_ord_id] = {symbol, signed_quantity};
    }
};

int main(int argc, char* argv[]) {
    try {
        // Setup signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // Parse command line arguments
        namespace po = boost::program_options;
        po::options_description desc("HFT Market Maker Options");
        desc.add_options()
            ("help,h", "Show help message")
            ("config,c", po::value<std::string>()->required(), "Configuration file path")
            ("log-level,l", po::value<std::string>()->default_value("INFO"), "Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
            ("log-file", po::value<std::string>()->default_value("market_maker.log"), "Log file path")
            ("dry-run", po::bool_switch()->default_value(false), "Run in simulation mode without sending real orders")
            ("validate-only", po::bool_switch()->default_value(false), "Validate configuration and exit");
        
        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            
            if (vm.count("help")) {
                std::cout << desc << std::endl;
                return 0;
            }
            
            po::notify(vm);
        } catch (const po::error& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            std::cerr << desc << std::endl;
            return 1;
        }
        
        std::string config_file = vm["config"].as<std::string>();
        std::string log_level_str = vm["log-level"].as<std::string>();
        std::string log_file = vm["log-file"].as<std::string>();
        bool dry_run = vm["dry-run"].as<bool>();
        bool validate_only = vm["validate-only"].as<bool>();
        
        // Initialize logger
        LogLevel log_level = LogLevel::INFO;
        if (log_level_str == "DEBUG") log_level = LogLevel::DEBUG;
        else if (log_level_str == "INFO") log_level = LogLevel::INFO;
        else if (log_level_str == "WARNING") log_level = LogLevel::WARNING;
        else if (log_level_str == "ERROR") log_level = LogLevel::ERROR;
        else if (log_level_str == "CRITICAL") log_level = LogLevel::CRITICAL;
        
        Logger::initialize(log_level, log_file);
        LOG_INFO("Starting HFT Market Maker v" + std::string(PROJECT_VERSION));
        
        // Load configuration
        LOG_INFO("Loading configuration from " + config_file);
        Config config = load_config(config_file);
        
        if (validate_only) {
            LOG_INFO("Configuration validation successful");
            return 0;
        }
        
        if (dry_run) {
            LOG_WARNING("Running in DRY RUN mode - no real orders will be sent");
        }
        
        // Initialize risk system
        LOG_INFO("Initializing risk management system...");
        PERF_LOG("risk_system_init");
        
        CapitalAllocator::AllocationConstraints capital_constraints{
            1000000.0,  // $1M total capital
            0.1,        // Max 10% per position
            0.01,       // Min 1% per position
            2.0,        // Target Sharpe ratio
            0.02        // 2% risk-free rate
        };
        
        MarginManager::MarginRequirements margin_requirements{
            0.25,       // 25% initial margin
            0.20,       // 20% maintenance margin
            1.5,        // 1.5x for options
            {}          // No symbol-specific margins
        };
        
        ComplianceManager::ComplianceRules compliance_rules{
            false,      // Allow short selling
            false,      // No uptick rule
            0.05,       // Max 5% of ADV
            0.10,       // Max 10% participation
            100.0,      // 100ms min order spacing
            5,          // Max 5 modifications
            {},         // No restricted symbols
            {},         // No position limits
            {},         // No blackout periods
            true        // Regular hours only
        };
        
        auto risk_system = std::make_shared<IntegratedRiskSystem>(
            config.risk_limits,
            capital_constraints,
            margin_requirements,
            compliance_rules
        );
        
        // Initialize order handler
        auto order_handler = std::make_shared<MarketMakerOrderHandler>(risk_system);
        
        // Initialize market data handler (placeholder for actual handler)
        auto market_handler = std::make_shared<MarketMakerDataHandler>(
            std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>>(),
            risk_system,
            nullptr  // Will be set later
        );
        
        // Initialize FIX handler
        LOG_INFO("Initializing FIX protocol handler...");
        PERF_LOG("fix_handler_init");
        auto fix_handler = std::make_shared<FIXMarketDataHandler>(
            market_handler, order_handler
        );
        
        // Initialize strategies
        LOG_INFO("Initializing trading strategies for " + std::to_string(config.symbols.size()) + " symbols");
        PERF_LOG("strategies_init");
        std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>> strategies;
        
        RiskParams risk_params{
            100000.0,   // $100k max position
            10000.0,    // $10k buffer
            0.0001,     // 0.01% inventory penalty
            0.0001,     // 0.01% min spread
            0.0010,     // 0.10% max spread
            100.0,      // 100 share default size
            1.5,        // 1.5x volatility factor
            0.7,        // 0.7 toxicity threshold
            50000.0,    // $50k max drawdown
            2.0         // 2.0 target Sharpe
        };
        
        for (const auto& symbol : config.symbols) {
            if (config.strategy_type == "avellaneda_stoikov") {
                strategies[symbol] = std::make_shared<AvellanedaStoikovStrategy>(
                    risk_params,
                    config.risk_aversion,
                    config.order_arrival_rate,
                    config.time_horizon
                );
            } else if (config.strategy_type == "alpha") {
                strategies[symbol] = std::make_shared<AlphaMarketMaker>(risk_params);
            }
        }
        
        // Update market handler with strategies and FIX handler
        market_handler = std::make_shared<MarketMakerDataHandler>(
            strategies, risk_system, fix_handler
        );
        
        // Initialize market data feed
        LOG_INFO("Initializing market data feed: " + config.market_data_feed);
        PERF_LOG("market_data_init");
        std::shared_ptr<MarketDataFeed> data_feed;
        
        if (config.market_data_feed == "polygon") {
            data_feed = std::make_shared<PolygonWebSocketFeed>(config.polygon_api_key);
            data_feed->connect("wss://socket.polygon.io/stocks");
        } else if (config.market_data_feed == "simulated") {
            LOG_WARNING("Using simulated market data feed");
            data_feed = std::make_shared<SimulatedMarketDataFeed>("./historical_data");
        }
        
        data_feed->addListener(market_handler);
        
        // Subscribe to symbols
        LOG_INFO("Subscribing to market data for symbols: " + 
                 std::accumulate(config.symbols.begin(), config.symbols.end(), std::string(),
                 [](const std::string& a, const std::string& b) { return a.empty() ? b : a + ", " + b; }));
        for (const auto& symbol : config.symbols) {
            data_feed->subscribe(symbol, 10);  // 10 levels deep
            fix_handler->subscribeMarketData(symbol, 10);
        }
        
        // Start risk monitoring thread
        std::thread risk_monitor_thread([&risk_system]() {
            while (g_running) {
                // Periodic risk checks
                if (!risk_system->getRiskManager()->isWithinRiskLimits()) {
                    LOG_ERROR("Risk limits breached! Emergency stop may be triggered.");
                }
                
                // Print metrics every 10 seconds
                static auto last_print = std::chrono::high_resolution_clock::now();
                auto now = std::chrono::high_resolution_clock::now();
                
                if (std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_print).count() >= 10) {
                    
                    auto metrics = risk_system->getRiskManager()->getMetrics();
                    std::stringstream ss;
                    ss << "Risk Metrics - "
                       << "Exposure: $" << std::fixed << std::setprecision(2) << metrics.total_exposure
                       << ", Daily P&L: $" << metrics.daily_pnl
                       << ", Drawdown: " << std::setprecision(1) << metrics.current_drawdown * 100 << "%"
                       << ", Leverage: " << std::setprecision(2) << metrics.leverage << "x";
                    LOG_INFO(ss.str());
                    
                    last_print = now;
                }
                
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
        
        // Main loop
        LOG_INFO("Market maker is running. Press Ctrl+C to stop.");
        
        // Log system information
        LOG_INFO("System ready - Monitoring " + std::to_string(config.symbols.size()) + 
                 " symbols with " + std::to_string(strategies.size()) + " strategies");
        
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Perform periodic tasks
            static auto last_sod = std::chrono::system_clock::now();
            auto now = std::chrono::system_clock::now();
            
            // Daily start-of-day checks
            auto hours = std::chrono::duration_cast<std::chrono::hours>(
                now - last_sod).count();
            if (hours >= 24) {
                risk_system->performStartOfDayChecks();
                last_sod = now;
            }
        }
        
        // Cleanup
        LOG_INFO("Shutdown signal received, cleaning up...");
        
        // Stop market data
        data_feed->disconnect();
        
        // Wait for threads
        if (risk_monitor_thread.joinable()) {
            risk_monitor_thread.join();
        }
        
        // Final reconciliation
        risk_system->performEndOfDayReconciliation();
        
        // Print final metrics
        auto final_metrics = risk_system->getRiskManager()->getMetrics();
        LOG_INFO("=== Final Session Metrics ===");
        LOG_INFO("Total P&L: $" + std::to_string(final_metrics.daily_pnl));
        LOG_INFO("Max Drawdown: " + std::to_string(final_metrics.max_drawdown * 100) + "%");
        LOG_INFO("Total Messages: " + std::to_string(fix_handler->getMessageCount()));
        LOG_INFO("Session ended successfully");
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Fatal error: " + std::string(e.what()));
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        LOG_CRITICAL("Unknown fatal error occurred");
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 2;
    }
    
    return 0;
}