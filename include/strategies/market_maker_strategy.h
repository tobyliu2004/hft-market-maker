#pragma once

#include "core/order_book.h"
#include "core/fix_handler.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <optional>

namespace hft {

// Risk parameters for market making
struct RiskParams {
    double max_position;          // Maximum inventory allowed
    double position_limit_buffer; // Buffer before hitting max position
    double inventory_penalty;     // Penalty factor for inventory risk
    double min_spread;           // Minimum spread to quote
    double max_spread;           // Maximum spread to quote
    double order_size;           // Default order size
    double volatility_factor;    // Factor to adjust spread based on volatility
    double adverse_selection_threshold; // Threshold for toxic flow detection
    double max_drawdown;         // Maximum allowed drawdown
    double sharpe_target;        // Target Sharpe ratio
};

// Market state for strategy decisions
struct MarketState {
    double current_position;
    double unrealized_pnl;
    double realized_pnl;
    double current_volatility;
    double volume_imbalance;
    double order_flow_toxicity;
    double spread_percentile;    // Current spread as percentile of historical
    double queue_position;       // Estimated queue position
    Timestamp last_update;
};

// Strategy performance metrics
struct StrategyMetrics {
    double total_pnl;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
    uint64_t trades_count;
    double avg_spread_captured;
    double inventory_turnover;
    double adverse_selection_cost;
    std::vector<double> pnl_curve;
};

// Base class for market making strategies
class MarketMakingStrategy {
protected:
    RiskParams risk_params_;
    MarketState market_state_;
    StrategyMetrics metrics_;
    
public:
    MarketMakingStrategy(const RiskParams& params) : risk_params_(params) {
        market_state_ = {0, 0, 0, 0, 0, 0, 0, 0, Timestamp(0)};
        metrics_ = {0, 0, 0, 0, 0, 0, 0, 0, {}};
    }
    
    virtual ~MarketMakingStrategy() = default;
    
    // Core strategy method - returns bid/ask quotes
    virtual std::pair<std::optional<double>, std::optional<double>> getQuotes(
        const OrderBook& book,
        const MarketState& state
    ) = 0;
    
    // Update market state
    virtual void updateState(const MarketState& new_state) {
        market_state_ = new_state;
    }
    
    // Get current metrics
    const StrategyMetrics& getMetrics() const { return metrics_; }
    
    // Risk checks
    bool isWithinRiskLimits() const;
    double getInventoryPenalty() const;
};

// Avellaneda-Stoikov market making model
class AvellanedaStoikovStrategy : public MarketMakingStrategy {
private:
    // Model parameters
    double gamma_;           // Risk aversion parameter
    double k_;              // Order arrival rate
    double sigma_;          // Volatility estimate
    double T_;              // Time horizon
    double q_max_;          // Maximum inventory
    
    // Optimization parameters
    double reservation_price_;
    double optimal_spread_;
    
    // Helper methods
    double calculateReservationPrice(double mid_price, double inventory, double time_remaining);
    double calculateOptimalSpread(double volatility, double gamma, double time_remaining);
    double inventoryPenaltyFunction(double inventory);
    
public:
    AvellanedaStoikovStrategy(const RiskParams& params, double gamma, double k, double T);
    
    std::pair<std::optional<double>, std::optional<double>> getQuotes(
        const OrderBook& book,
        const MarketState& state
    ) override;
    
    // Update model parameters based on market conditions
    void updateVolatility(double new_sigma);
    void updateArrivalRate(double new_k);
    
    // Get internal state for monitoring
    double getReservationPrice() const { return reservation_price_; }
    double getOptimalSpread() const { return optimal_spread_; }
};

// Enhanced market making with multiple alpha signals
class AlphaMarketMaker : public MarketMakingStrategy {
private:
    struct AlphaSignal {
        std::string name;
        double weight;
        double value;
        double confidence;
    };
    
    std::vector<AlphaSignal> alpha_signals_;
    
    // Alpha generation methods
    double calculateMicropriceAlpha(const OrderBook& book);
    double calculateOrderFlowAlpha(const OrderBook& book);
    double calculateQueueAlpha(const OrderBook& book);
    double calculateSpreadAlpha(const OrderBook& book);
    
    // Signal combination
    double combineAlphas();
    
public:
    AlphaMarketMaker(const RiskParams& params);
    
    std::pair<std::optional<double>, std::optional<double>> getQuotes(
        const OrderBook& book,
        const MarketState& state
    ) override;
    
    void addAlphaSignal(const std::string& name, double weight);
    const std::vector<AlphaSignal>& getAlphaSignals() const { return alpha_signals_; }
};

// Statistical arbitrage market maker
class StatArbMarketMaker : public MarketMakingStrategy {
private:
    struct PairState {
        std::string symbol1;
        std::string symbol2;
        double hedge_ratio;
        double spread_mean;
        double spread_std;
        double current_spread;
        double z_score;
    };
    
    std::vector<PairState> pairs_;
    double entry_threshold_;
    double exit_threshold_;
    
    // Cointegration testing
    bool testCointegration(const std::vector<double>& prices1, 
                          const std::vector<double>& prices2);
    double calculateHedgeRatio(const std::vector<double>& prices1,
                              const std::vector<double>& prices2);
    
public:
    StatArbMarketMaker(const RiskParams& params, double entry_z, double exit_z);
    
    std::pair<std::optional<double>, std::optional<double>> getQuotes(
        const OrderBook& book,
        const MarketState& state
    ) override;
    
    void addPair(const std::string& symbol1, const std::string& symbol2);
    void updatePairStatistics();
};

// Intelligent order placement with queue modeling
class SmartOrderPlacer {
private:
    struct QueueModel {
        double arrival_rate;
        double cancellation_rate;
        double avg_order_size;
        std::vector<double> queue_lengths;
    };
    
    std::unordered_map<Price, QueueModel> queue_models_;
    
    // Queue position estimation
    double estimateQueuePosition(Price price, Quantity size, const OrderBook& book);
    double probabilityOfFill(double queue_position, double time_horizon);
    
public:
    SmartOrderPlacer();
    
    // Optimal order placement
    struct PlacementDecision {
        Price price;
        Quantity size;
        double expected_fill_time;
        double fill_probability;
    };
    
    PlacementDecision getOptimalPlacement(
        Side side,
        Quantity target_size,
        const OrderBook& book,
        double urgency
    );
    
    // Update queue models with execution data
    void updateQueueModel(Price price, double fill_time, Quantity filled_size);
};

// Market impact model for large orders
class MarketImpactModel {
private:
    // Almgren-Chriss model parameters
    double permanent_impact_coefficient_;
    double temporary_impact_coefficient_;
    double volatility_;
    
public:
    MarketImpactModel(double perm_impact, double temp_impact, double vol);
    
    // Calculate expected impact of an order
    double calculatePermanentImpact(Quantity size, double avg_daily_volume);
    double calculateTemporaryImpact(Quantity size, double participation_rate);
    
    // Optimal execution trajectory
    std::vector<Quantity> getOptimalTrajectory(
        Quantity total_size,
        int time_periods,
        double risk_aversion
    );
};

// Post-trade analysis
class PostTradeAnalyzer {
private:
    struct TradeRecord {
        Timestamp time;
        Side side;
        Price price;
        Quantity quantity;
        double market_price_before;
        double market_price_after;
        double realized_spread;
        double effective_spread;
    };
    
    std::vector<TradeRecord> trades_;
    
public:
    PostTradeAnalyzer();
    
    void recordTrade(const TradeRecord& trade);
    
    // Analysis methods
    double calculateRealizedSpread();
    double calculateEffectiveSpread();
    double calculatePriceImprovement();
    double calculateAdverseSelection(int time_window_ms);
    
    // Generate report
    struct AnalysisReport {
        double avg_realized_spread;
        double avg_effective_spread;
        double price_improvement_rate;
        double adverse_selection_cost;
        double information_ratio;
    };
    
    AnalysisReport generateReport();
};

} // namespace hft