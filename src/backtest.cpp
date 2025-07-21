#include "core/order_book.h"
#include "strategies/market_maker_strategy.h"
#include "models/python_bridge.h"
#include "risk/risk_manager.h"
#include "utils/logger.h"
#include "utils/config_validator.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <queue>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
using namespace hft;
using json = nlohmann::json;

struct BacktestConfig {
    std::string data_directory;
    std::string start_date;
    std::string end_date;
    std::vector<std::string> symbols;
    double initial_capital;
    double latency_ms;
    bool enable_slippage;
    double slippage_bps;
    std::string output_file;
};

struct BacktestEvent {
    std::chrono::nanoseconds timestamp;
    std::string symbol;
    enum Type { QUOTE, TRADE, ORDER_BOOK } type;
    json data;
    
    bool operator>(const BacktestEvent& other) const {
        return timestamp > other.timestamp;
    }
};

class BacktestEngine {
private:
    BacktestConfig config_;
    std::shared_ptr<IntegratedRiskSystem> risk_system_;
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> order_books_;
    std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>> strategies_;
    std::shared_ptr<StatisticalModels> stat_models_;
    
    std::priority_queue<BacktestEvent, std::vector<BacktestEvent>, std::greater<BacktestEvent>> event_queue_;
    
    struct BacktestMetrics {
        double total_pnl = 0.0;
        double max_drawdown = 0.0;
        double sharpe_ratio = 0.0;
        double win_rate = 0.0;
        int total_trades = 0;
        int winning_trades = 0;
        double avg_trade_size = 0.0;
        double total_volume = 0.0;
        double total_fees = 0.0;
        std::chrono::nanoseconds total_latency{0};
    } metrics_;
    
    std::vector<std::pair<std::chrono::nanoseconds, double>> equity_curve_;
    
public:
    BacktestEngine(const BacktestConfig& config,
                   std::shared_ptr<IntegratedRiskSystem> risk_system,
                   const std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>>& strategies)
        : config_(config), 
          risk_system_(risk_system),
          strategies_(strategies) {
        
        stat_models_ = std::make_shared<StatisticalModels>();
        stat_models_->initialize();
        
        for (const auto& symbol : config.symbols) {
            order_books_[symbol] = std::make_shared<OrderBook>();
        }
    }
    
    void loadHistoricalData() {
        LOG_INFO("Loading historical data from " + config_.data_directory);
        
        for (const auto& symbol : config_.symbols) {
            loadSymbolData(symbol);
        }
        
        LOG_INFO("Loaded " + std::to_string(event_queue_.size()) + " events");
    }
    
    void run() {
        LOG_INFO("Starting backtest from " + config_.start_date + " to " + config_.end_date);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (!event_queue_.empty()) {
            auto event = event_queue_.top();
            event_queue_.pop();
            
            processEvent(event);
            
            // Update equity curve periodically
            static auto last_equity_update = event.timestamp;
            if (event.timestamp - last_equity_update > std::chrono::minutes(1)) {
                updateEquityCurve(event.timestamp);
                last_equity_update = event.timestamp;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        LOG_INFO("Backtest completed in " + std::to_string(duration.count()) + " ms");
        
        calculateFinalMetrics();
        outputResults();
    }
    
private:
    void loadSymbolData(const std::string& symbol) {
        // Load quotes
        auto quotes_file = fs::path(config_.data_directory) / symbol / "quotes.csv";
        if (fs::exists(quotes_file)) {
            loadQuotesFile(symbol, quotes_file);
        }
        
        // Load trades
        auto trades_file = fs::path(config_.data_directory) / symbol / "trades.csv";
        if (fs::exists(trades_file)) {
            loadTradesFile(symbol, trades_file);
        }
        
        // Load order book snapshots
        auto book_file = fs::path(config_.data_directory) / symbol / "book_snapshots.csv";
        if (fs::exists(book_file)) {
            loadOrderBookFile(symbol, book_file);
        }
    }
    
    void loadQuotesFile(const std::string& symbol, const fs::path& file) {
        std::ifstream ifs(file);
        std::string line;
        
        // Skip header
        std::getline(ifs, line);
        
        while (std::getline(ifs, line)) {
            // Parse CSV: timestamp,bid_price,bid_size,ask_price,ask_size
            std::istringstream iss(line);
            std::string timestamp_str;
            double bid_price, bid_size, ask_price, ask_size;
            
            std::getline(iss, timestamp_str, ',');
            iss >> bid_price;
            iss.ignore();
            iss >> bid_size;
            iss.ignore();
            iss >> ask_price;
            iss.ignore();
            iss >> ask_size;
            
            BacktestEvent event;
            event.timestamp = parseTimestamp(timestamp_str);
            event.symbol = symbol;
            event.type = BacktestEvent::QUOTE;
            event.data = {
                {"bid_price", bid_price},
                {"bid_size", bid_size},
                {"ask_price", ask_price},
                {"ask_size", ask_size}
            };
            
            event_queue_.push(event);
        }
    }
    
    void loadTradesFile(const std::string& symbol, const fs::path& file) {
        // Similar implementation for trades
    }
    
    void loadOrderBookFile(const std::string& symbol, const fs::path& file) {
        // Similar implementation for order book snapshots
    }
    
    std::chrono::nanoseconds parseTimestamp(const std::string& timestamp_str) {
        // Parse ISO 8601 timestamp to nanoseconds since epoch
        std::tm tm = {};
        std::istringstream ss(timestamp_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        return tp.time_since_epoch();
    }
    
    void processEvent(const BacktestEvent& event) {
        switch (event.type) {
            case BacktestEvent::QUOTE:
                processQuote(event);
                break;
            case BacktestEvent::TRADE:
                processTrade(event);
                break;
            case BacktestEvent::ORDER_BOOK:
                processOrderBook(event);
                break;
        }
    }
    
    void processQuote(const BacktestEvent& event) {
        auto& book = order_books_[event.symbol];
        
        // Update order book with new quote
        book->cancelOrder(1);  // Remove old best bid
        book->cancelOrder(2);  // Remove old best ask
        
        auto bid_order = std::make_shared<Order>(
            1, Side::BUY, 
            event.data["bid_price"].get<double>(),
            event.data["bid_size"].get<double>(),
            event.timestamp
        );
        
        auto ask_order = std::make_shared<Order>(
            2, Side::SELL,
            event.data["ask_price"].get<double>(),
            event.data["ask_size"].get<double>(),
            event.timestamp
        );
        
        book->addOrder(bid_order);
        book->addOrder(ask_order);
        
        // Update mid price for risk calculations
        double mid_price = (event.data["bid_price"].get<double>() + 
                           event.data["ask_price"].get<double>()) / 2.0;
        risk_system_->getRiskManager()->markToMarket(event.symbol, mid_price);
        
        // Get strategy quotes
        auto strategy_it = strategies_.find(event.symbol);
        if (strategy_it != strategies_.end()) {
            evaluateStrategy(event.symbol, strategy_it->second, *book, event.timestamp);
        }
    }
    
    void processTrade(const BacktestEvent& event) {
        // Update trade statistics and check for fills
    }
    
    void processOrderBook(const BacktestEvent& event) {
        // Update full order book state
    }
    
    void evaluateStrategy(const std::string& symbol, 
                         std::shared_ptr<MarketMakingStrategy> strategy,
                         const OrderBook& book,
                         std::chrono::nanoseconds timestamp) {
        // Get market state
        MarketState state;
        state.current_position = risk_system_->getRiskManager()->getPosition(symbol).quantity;
        state.unrealized_pnl = risk_system_->getRiskManager()->getPosition(symbol).unrealized_pnl;
        state.realized_pnl = risk_system_->getRiskManager()->getPosition(symbol).realized_pnl;
        state.current_volatility = 0.02;  // Would calculate from historical data
        state.last_update = timestamp;
        
        // Get strategy quotes
        auto [bid_quote, ask_quote] = strategy->getQuotes(book, state);
        
        if (bid_quote.has_value() && ask_quote.has_value()) {
            // Simulate order execution with latency
            auto execution_time = timestamp + std::chrono::nanoseconds(
                static_cast<int64_t>(config_.latency_ms * 1e6)
            );
            
            simulateOrderExecution(symbol, Side::BUY, bid_quote.value(), 100, execution_time);
            simulateOrderExecution(symbol, Side::SELL, ask_quote.value(), 100, execution_time);
        }
    }
    
    void simulateOrderExecution(const std::string& symbol, Side side, 
                               double price, double quantity,
                               std::chrono::nanoseconds execution_time) {
        // Apply slippage if enabled
        if (config_.enable_slippage) {
            double slippage = price * config_.slippage_bps / 10000.0;
            price += (side == Side::BUY) ? slippage : -slippage;
        }
        
        // Check risk limits
        std::vector<std::string> rejection_reasons;
        if (!risk_system_->approveOrder(symbol, side, quantity, price, rejection_reasons)) {
            return;
        }
        
        // Simulate fill (simplified - would check against order book)
        double fill_price = price;
        double fill_quantity = quantity;
        
        // Update position
        double signed_quantity = (side == Side::BUY) ? quantity : -quantity;
        risk_system_->getRiskManager()->updatePosition(symbol, signed_quantity, fill_price);
        
        // Update metrics
        metrics_.total_trades++;
        metrics_.total_volume += fill_quantity * fill_price;
        metrics_.total_fees += fill_quantity * fill_price * 0.0001;  // 1 bps fee
        
        // Track latency
        metrics_.total_latency += std::chrono::nanoseconds(
            static_cast<int64_t>(config_.latency_ms * 1e6)
        );
    }
    
    void updateEquityCurve(std::chrono::nanoseconds timestamp) {
        auto metrics = risk_system_->getRiskManager()->getMetrics();
        double total_equity = config_.initial_capital + metrics.daily_pnl;
        equity_curve_.push_back({timestamp, total_equity});
        
        // Update max drawdown
        static double peak_equity = config_.initial_capital;
        peak_equity = std::max(peak_equity, total_equity);
        double drawdown = (peak_equity - total_equity) / peak_equity;
        metrics_.max_drawdown = std::max(metrics_.max_drawdown, drawdown);
    }
    
    void calculateFinalMetrics() {
        auto final_metrics = risk_system_->getRiskManager()->getMetrics();
        
        metrics_.total_pnl = final_metrics.daily_pnl;
        
        // Calculate Sharpe ratio
        if (!equity_curve_.empty()) {
            std::vector<double> returns;
            for (size_t i = 1; i < equity_curve_.size(); i++) {
                double prev_equity = equity_curve_[i-1].second;
                double curr_equity = equity_curve_[i].second;
                returns.push_back((curr_equity - prev_equity) / prev_equity);
            }
            
            double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
            double variance = 0.0;
            for (double r : returns) {
                variance += std::pow(r - mean_return, 2);
            }
            variance /= returns.size();
            
            double annualized_return = mean_return * 252;  // Assuming daily returns
            double annualized_vol = std::sqrt(variance * 252);
            
            metrics_.sharpe_ratio = annualized_return / annualized_vol;
        }
        
        // Calculate win rate
        if (metrics_.total_trades > 0) {
            // This would need trade-level P&L tracking
            metrics_.win_rate = static_cast<double>(metrics_.winning_trades) / metrics_.total_trades;
            metrics_.avg_trade_size = metrics_.total_volume / metrics_.total_trades;
        }
    }
    
    void outputResults() {
        json results = {
            {"summary", {
                {"total_pnl", metrics_.total_pnl},
                {"max_drawdown", metrics_.max_drawdown},
                {"sharpe_ratio", metrics_.sharpe_ratio},
                {"win_rate", metrics_.win_rate},
                {"total_trades", metrics_.total_trades},
                {"avg_trade_size", metrics_.avg_trade_size},
                {"total_volume", metrics_.total_volume},
                {"total_fees", metrics_.total_fees},
                {"avg_latency_ms", metrics_.total_trades > 0 ? 
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        metrics_.total_latency / metrics_.total_trades).count() / 1000.0 : 0.0}
            }},
            {"equity_curve", json::array()}
        };
        
        // Add equity curve data
        for (const auto& [timestamp, equity] : equity_curve_) {
            results["equity_curve"].push_back({
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    timestamp).count()},
                {"equity", equity}
            });
        }
        
        // Write results to file
        std::ofstream ofs(config_.output_file);
        ofs << results.dump(2) << std::endl;
        
        // Print summary
        std::cout << "\n=== Backtest Results ===" << std::endl;
        std::cout << "Total P&L: $" << std::fixed << std::setprecision(2) 
                  << metrics_.total_pnl << std::endl;
        std::cout << "Max Drawdown: " << std::fixed << std::setprecision(1) 
                  << metrics_.max_drawdown * 100 << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << std::fixed << std::setprecision(2) 
                  << metrics_.sharpe_ratio << std::endl;
        std::cout << "Total Trades: " << metrics_.total_trades << std::endl;
        std::cout << "Win Rate: " << std::fixed << std::setprecision(1) 
                  << metrics_.win_rate * 100 << "%" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        namespace po = boost::program_options;
        po::options_description desc("Backtest Options");
        desc.add_options()
            ("help,h", "Show help message")
            ("config,c", po::value<std::string>()->required(), "Configuration file path")
            ("date,d", po::value<std::string>(), "Specific date to backtest (YYYY-MM-DD)")
            ("start", po::value<std::string>(), "Start date (YYYY-MM-DD)")
            ("end", po::value<std::string>(), "End date (YYYY-MM-DD)")
            ("data-dir", po::value<std::string>()->default_value("./historical_data"), 
             "Historical data directory")
            ("output,o", po::value<std::string>()->default_value("backtest_results.json"), 
             "Output file for results")
            ("initial-capital", po::value<double>()->default_value(1000000.0), 
             "Initial capital")
            ("latency", po::value<double>()->default_value(0.1), 
             "Simulated latency in milliseconds")
            ("slippage", po::value<double>()->default_value(0.5), 
             "Slippage in basis points")
            ("no-slippage", po::bool_switch()->default_value(false), 
             "Disable slippage simulation");
        
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
        
        // Initialize logger
        Logger::initialize(LogLevel::INFO, "backtest.log");
        LOG_INFO("Starting backtest engine");
        
        // Load main configuration
        std::string config_file = vm["config"].as<std::string>();
        std::ifstream ifs(config_file);
        json config_json;
        ifs >> config_json;
        
        // Validate configuration
        auto validation = ConfigValidator::validateConfig(config_json);
        if (!validation.is_valid) {
            LOG_ERROR("Invalid configuration");
            for (const auto& error : validation.errors) {
                LOG_ERROR(error);
            }
            return 1;
        }
        
        // Build backtest config
        BacktestConfig backtest_config;
        backtest_config.data_directory = vm["data-dir"].as<std::string>();
        backtest_config.initial_capital = vm["initial-capital"].as<double>();
        backtest_config.latency_ms = vm["latency"].as<double>();
        backtest_config.enable_slippage = !vm["no-slippage"].as<bool>();
        backtest_config.slippage_bps = vm["slippage"].as<double>();
        backtest_config.output_file = vm["output"].as<std::string>();
        
        // Parse dates
        if (vm.count("date")) {
            backtest_config.start_date = vm["date"].as<std::string>();
            backtest_config.end_date = vm["date"].as<std::string>();
        } else {
            backtest_config.start_date = vm.count("start") ? 
                vm["start"].as<std::string>() : "2024-01-01";
            backtest_config.end_date = vm.count("end") ? 
                vm["end"].as<std::string>() : "2024-01-31";
        }
        
        // Get symbols from config
        for (const auto& symbol : config_json["market_data"]["symbols"]) {
            backtest_config.symbols.push_back(symbol);
        }
        
        // Initialize risk system
        RiskLimits risk_limits;
        auto& risk_json = config_json["risk_limits"];
        risk_limits.max_position_value = risk_json["max_position_value"];
        risk_limits.max_total_exposure = risk_json["max_total_exposure"];
        risk_limits.max_daily_loss = risk_json["max_daily_loss"];
        risk_limits.max_drawdown = risk_json["max_drawdown"];
        
        CapitalAllocator::AllocationConstraints capital_constraints{
            backtest_config.initial_capital,
            0.1, 0.01, 2.0, 0.02
        };
        
        MarginManager::MarginRequirements margin_requirements{
            0.25, 0.20, 1.5, {}
        };
        
        ComplianceManager::ComplianceRules compliance_rules{
            false, false, 0.05, 0.10, 100.0, 5, {}, {}, {}, true
        };
        
        auto risk_system = std::make_shared<IntegratedRiskSystem>(
            risk_limits, capital_constraints, margin_requirements, compliance_rules
        );
        
        // Initialize strategies
        std::unordered_map<std::string, std::shared_ptr<MarketMakingStrategy>> strategies;
        RiskParams risk_params{
            100000.0, 10000.0, 0.0001, 0.0001, 0.0010, 
            100.0, 1.5, 0.7, 50000.0, 2.0
        };
        
        for (const auto& symbol : backtest_config.symbols) {
            if (config_json["strategy"]["type"] == "avellaneda_stoikov") {
                strategies[symbol] = std::make_shared<AvellanedaStoikovStrategy>(
                    risk_params,
                    config_json["strategy"]["risk_aversion"],
                    config_json["strategy"]["order_arrival_rate"],
                    config_json["strategy"]["time_horizon"]
                );
            }
        }
        
        // Run backtest
        BacktestEngine engine(backtest_config, risk_system, strategies);
        engine.loadHistoricalData();
        engine.run();
        
        LOG_INFO("Backtest completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Backtest failed: " + std::string(e.what()));
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}