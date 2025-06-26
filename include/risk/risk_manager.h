#pragma once

#include "core/order_book.h"
#include "strategies/market_maker_strategy.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <queue>
#include <optional>

namespace hft {

// Position tracking
struct Position {
    std::string symbol;
    double quantity;
    double average_price;
    double market_value;
    double unrealized_pnl;
    double realized_pnl;
    Timestamp last_update;
    
    double getTotalPnL() const { return unrealized_pnl + realized_pnl; }
};

// Risk limits configuration
struct RiskLimits {
    // Position limits
    double max_position_value;        // Maximum $ value per position
    double max_total_exposure;        // Maximum total $ exposure
    double max_position_count;        // Maximum number of positions
    
    // Loss limits
    double max_daily_loss;           // Maximum daily loss allowed
    double max_drawdown;             // Maximum drawdown from peak
    double stop_loss_percent;        // Stop loss percentage per position
    
    // Order limits
    double max_order_size;           // Maximum single order size
    double max_order_value;          // Maximum single order value
    int max_orders_per_second;       // Rate limit
    int max_open_orders;             // Maximum open orders
    
    // Market risk limits
    double max_spread_percent;       // Maximum spread to trade
    double min_liquidity_ratio;      // Minimum liquidity (our size / market size)
    double max_market_impact;        // Maximum acceptable market impact
    
    // Greeks limits (for options)
    double max_delta;
    double max_gamma;
    double max_vega;
    double max_theta;
    
    // Leverage limits
    double max_leverage;             // Maximum leverage allowed
    double margin_call_level;        // Margin call threshold
    double liquidation_level;        // Forced liquidation threshold
};

// Risk metrics
struct RiskMetrics {
    // Portfolio metrics
    double total_exposure;
    double net_exposure;
    double gross_exposure;
    double leverage;
    
    // P&L metrics
    double daily_pnl;
    double mtd_pnl;
    double ytd_pnl;
    double max_drawdown;
    double current_drawdown;
    
    // Risk measures
    double var_95;                   // Value at Risk (95% confidence)
    double var_99;                   // Value at Risk (99% confidence)
    double expected_shortfall;       // Conditional VaR
    double sharpe_ratio;
    double sortino_ratio;
    double information_ratio;
    
    // Greeks (aggregate)
    double portfolio_delta;
    double portfolio_gamma;
    double portfolio_vega;
    double portfolio_theta;
    
    // Liquidity metrics
    double liquidity_score;          // 0-1 score of portfolio liquidity
    double time_to_liquidate;        // Estimated time to close all positions
    
    Timestamp last_update;
};

// Risk events
enum class RiskEventType {
    LIMIT_BREACH,
    STOP_LOSS_TRIGGERED,
    MARGIN_CALL,
    CIRCUIT_BREAKER,
    ABNORMAL_MARKET,
    SYSTEM_ERROR
};

struct RiskEvent {
    RiskEventType type;
    std::string description;
    std::string symbol;
    double severity;  // 0-1 scale
    Timestamp timestamp;
    std::unordered_map<std::string, std::string> metadata;
};

// Risk manager interface
class RiskManager {
protected:
    RiskLimits limits_;
    RiskMetrics metrics_;
    std::unordered_map<std::string, Position> positions_;
    
    mutable std::mutex mutex_;
    std::atomic<bool> emergency_stop_{false};
    
    // Risk event handling
    std::vector<std::function<void(const RiskEvent&)>> risk_handlers_;
    std::queue<RiskEvent> risk_event_queue_;
    
    // Historical data for risk calculations
    std::deque<double> daily_returns_;
    std::deque<double> portfolio_values_;
    double peak_portfolio_value_;
    
    // Helper methods
    void updateMetrics();
    void checkLimits();
    void calculateVaR();
    void calculateGreeks();
    
public:
    explicit RiskManager(const RiskLimits& limits);
    virtual ~RiskManager() = default;
    
    // Position management
    virtual bool canTakePosition(const std::string& symbol, double quantity, double price);
    virtual void updatePosition(const std::string& symbol, double quantity, double price);
    virtual void markToMarket(const std::string& symbol, double market_price);
    
    // Order validation
    virtual bool validateOrder(const std::string& symbol, Side side, 
                             double quantity, double price);
    virtual bool checkOrderLimits(double order_value, int current_open_orders);
    
    // Risk checks
    virtual bool isWithinRiskLimits() const;
    virtual bool shouldStopTrading() const { return emergency_stop_.load(); }
    virtual void triggerEmergencyStop(const std::string& reason);
    
    // P&L management
    virtual void realizePnL(const std::string& symbol, double amount);
    virtual double getUnrealizedPnL() const;
    virtual double getRealizedPnL() const;
    virtual double getTotalPnL() const;
    
    // Risk metrics
    const RiskMetrics& getMetrics() const { return metrics_; }
    Position getPosition(const std::string& symbol) const;
    std::vector<Position> getAllPositions() const;
    
    // Risk event handling
    void addRiskHandler(std::function<void(const RiskEvent&)> handler);
    void publishRiskEvent(const RiskEvent& event);
    
    // Configuration
    void updateLimits(const RiskLimits& new_limits);
    const RiskLimits& getLimits() const { return limits_; }
};

// Advanced risk manager with dynamic hedging
class DynamicRiskManager : public RiskManager {
private:
    // Hedging parameters
    struct HedgeParams {
        bool auto_hedge_enabled;
        double delta_hedge_threshold;
        double gamma_hedge_threshold;
        double hedge_rebalance_interval_ms;
        std::vector<std::string> hedge_instruments;
    };
    
    HedgeParams hedge_params_;
    std::chrono::high_resolution_clock::time_point last_hedge_time_;
    
    // Correlation matrix for portfolio risk
    std::unordered_map<std::string, std::unordered_map<std::string, double>> correlations_;
    
    // Stress testing scenarios
    struct StressScenario {
        std::string name;
        std::unordered_map<std::string, double> price_shocks;
        double probability;
    };
    std::vector<StressScenario> stress_scenarios_;
    
    // Machine learning risk model
    std::shared_ptr<PricePredictionModel> ml_risk_model_;
    
    // Calculate portfolio risk considering correlations
    double calculatePortfolioRisk();
    
    // Generate hedge orders
    std::vector<std::pair<std::string, double>> calculateHedgeOrders();
    
public:
    DynamicRiskManager(const RiskLimits& limits, const HedgeParams& hedge_params);
    
    // Enhanced risk checks
    bool canTakePosition(const std::string& symbol, double quantity, double price) override;
    
    // Correlation management
    void updateCorrelation(const std::string& symbol1, const std::string& symbol2, 
                          double correlation);
    double getCorrelation(const std::string& symbol1, const std::string& symbol2) const;
    
    // Stress testing
    void addStressScenario(const StressScenario& scenario);
    double runStressTest(const std::string& scenario_name);
    std::unordered_map<std::string, double> runAllStressTests();
    
    // Dynamic hedging
    bool shouldRebalanceHedge() const;
    std::vector<std::pair<std::string, double>> getHedgeRecommendations();
    
    // ML-based risk prediction
    double predictRisk(const std::string& symbol, double time_horizon_minutes);
    double predictPortfolioRisk(double time_horizon_minutes);
};

// Capital allocation optimizer
class CapitalAllocator {
private:
    struct AllocationConstraints {
        double total_capital;
        double max_allocation_percent;
        double min_allocation_percent;
        double target_sharpe_ratio;
        double risk_free_rate;
    };
    
    AllocationConstraints constraints_;
    
    // Historical performance data
    std::unordered_map<std::string, std::vector<double>> strategy_returns_;
    std::unordered_map<std::string, double> strategy_sharpe_ratios_;
    std::unordered_map<std::string, double> strategy_max_drawdowns_;
    
    // Optimization methods
    std::vector<double> meanVarianceOptimization(
        const std::vector<std::vector<double>>& returns,
        double target_return
    );
    
    std::vector<double> kellyOptimization(
        const std::vector<double>& expected_returns,
        const std::vector<double>& variances
    );
    
public:
    explicit CapitalAllocator(const AllocationConstraints& constraints);
    
    // Update strategy performance
    void updateStrategyPerformance(const std::string& strategy_id, 
                                  const std::vector<double>& returns);
    
    // Calculate optimal allocations
    std::unordered_map<std::string, double> optimizeAllocation(
        const std::vector<std::string>& strategies,
        const std::string& method = "mean_variance"
    );
    
    // Get allocation recommendations
    double getRecommendedSize(const std::string& strategy_id, 
                             double signal_strength,
                             double current_exposure);
    
    // Risk parity allocation
    std::unordered_map<std::string, double> riskParityAllocation(
        const std::vector<std::string>& strategies
    );
};

// Margin and collateral manager
class MarginManager {
private:
    struct MarginRequirements {
        double initial_margin_percent;
        double maintenance_margin_percent;
        double option_margin_multiplier;
        std::unordered_map<std::string, double> symbol_specific_margins;
    };
    
    struct AccountBalance {
        double cash_balance;
        double securities_value;
        double margin_used;
        double available_margin;
        double excess_liquidity;
        Timestamp last_update;
    };
    
    MarginRequirements requirements_;
    AccountBalance balance_;
    
    // Collateral tracking
    std::unordered_map<std::string, double> collateral_positions_;
    std::unordered_map<std::string, double> haircuts_;  // Collateral haircuts
    
    mutable std::mutex mutex_;
    
public:
    explicit MarginManager(const MarginRequirements& requirements);
    
    // Margin calculations
    double calculateInitialMargin(const std::string& symbol, double quantity, double price);
    double calculateMaintenanceMargin(const std::string& symbol, double quantity, double price);
    
    // Margin checks
    bool hasAdequateMargin(double required_margin) const;
    double getExcessLiquidity() const;
    double getMarginUtilization() const;
    
    // Update account state
    void updateBalance(const AccountBalance& balance);
    void updateCollateral(const std::string& asset, double quantity, double value);
    
    // Margin call management
    bool isInMarginCall() const;
    std::vector<std::pair<std::string, double>> getMarginCallRequirements() const;
    
    // Get current state
    const AccountBalance& getBalance() const { return balance_; }
    double getTotalCollateralValue() const;
};

// Compliance and regulatory risk manager
class ComplianceManager {
private:
    struct ComplianceRules {
        // Trading rules
        bool no_short_selling;
        bool uptick_rule_enabled;
        double max_order_size_percent_adv;  // % of average daily volume
        
        // Market manipulation prevention
        double max_participation_rate;       // Max % of volume
        double min_order_spacing_ms;         // Prevent layering
        int max_order_modifications;         // Prevent spoofing
        
        // Regulatory limits
        std::unordered_map<std::string, bool> restricted_symbols;
        std::unordered_map<std::string, double> position_limits;
        
        // Time-based restrictions
        std::vector<std::pair<int, int>> blackout_periods;  // Hour ranges
        bool trade_only_regular_hours;
    };
    
    ComplianceRules rules_;
    
    // Tracking for compliance
    std::unordered_map<std::string, std::deque<Timestamp>> order_timestamps_;
    std::unordered_map<std::string, int> order_modifications_;
    std::unordered_map<std::string, double> participation_rates_;
    
    mutable std::mutex mutex_;
    
    bool isInBlackoutPeriod() const;
    bool checkParticipationRate(const std::string& symbol, double order_size);
    
public:
    explicit ComplianceManager(const ComplianceRules& rules);
    
    // Compliance checks
    bool isOrderCompliant(const std::string& symbol, Side side, 
                         double quantity, double price);
    bool canModifyOrder(const std::string& order_id);
    bool canTradeSymbol(const std::string& symbol) const;
    
    // Update tracking
    void recordOrder(const std::string& symbol, double quantity, Timestamp time);
    void recordModification(const std::string& order_id);
    void updateParticipationRate(const std::string& symbol, double rate);
    
    // Reporting
    struct ComplianceReport {
        std::unordered_map<std::string, int> violations_by_type;
        std::vector<std::string> flagged_orders;
        double total_participation_rate;
        int total_modifications;
    };
    
    ComplianceReport generateDailyReport() const;
    
    // Rule updates
    void updateRules(const ComplianceRules& new_rules);
    void addRestrictedSymbol(const std::string& symbol);
    void removeRestrictedSymbol(const std::string& symbol);
};

// Integrated risk management system
class IntegratedRiskSystem {
private:
    std::shared_ptr<DynamicRiskManager> risk_manager_;
    std::shared_ptr<CapitalAllocator> capital_allocator_;
    std::shared_ptr<MarginManager> margin_manager_;
    std::shared_ptr<ComplianceManager> compliance_manager_;
    
    // Risk committee (automated decision making)
    struct RiskDecision {
        bool approved;
        std::vector<std::string> reasons;
        double confidence;
        std::unordered_map<std::string, double> adjustments;
    };
    
    RiskDecision evaluateTradeProposal(const std::string& symbol, Side side,
                                      double quantity, double price);
    
public:
    IntegratedRiskSystem(const RiskLimits& risk_limits,
                        const CapitalAllocator::AllocationConstraints& capital_constraints,
                        const MarginManager::MarginRequirements& margin_requirements,
                        const ComplianceManager::ComplianceRules& compliance_rules);
    
    // Unified risk check
    bool approveOrder(const std::string& symbol, Side side, 
                     double quantity, double price,
                     std::vector<std::string>& rejection_reasons);
    
    // Position sizing with all constraints
    double calculateOptimalPositionSize(const std::string& symbol,
                                      double signal_strength,
                                      double target_risk);
    
    // Real-time risk monitoring
    void monitorRisk();
    void handleRiskEvent(const RiskEvent& event);
    
    // Risk-adjusted execution
    struct ExecutionPlan {
        std::vector<std::pair<double, double>> slices;  // (size, limit_price)
        double total_size;
        double average_price;
        double expected_impact;
        double risk_score;
    };
    
    ExecutionPlan planExecution(const std::string& symbol, Side side,
                               double target_quantity);
    
    // Daily operations
    void performStartOfDayChecks();
    void performEndOfDayReconciliation();
    
    // Access to subsystems
    std::shared_ptr<DynamicRiskManager> getRiskManager() { return risk_manager_; }
    std::shared_ptr<CapitalAllocator> getCapitalAllocator() { return capital_allocator_; }
    std::shared_ptr<MarginManager> getMarginManager() { return margin_manager_; }
    std::shared_ptr<ComplianceManager> getComplianceManager() { return compliance_manager_; }
};

} // namespace hft