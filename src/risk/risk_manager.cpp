#include "risk/risk_manager.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace hft {

// RiskManager implementation
RiskManager::RiskManager(const RiskLimits& limits) : limits_(limits) {
    metrics_ = {};
    peak_portfolio_value_ = 0.0;
}

bool RiskManager::canTakePosition(const std::string& symbol, double quantity, double price) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (emergency_stop_) {
        return false;
    }
    
    // Calculate new position value
    double position_value = std::abs(quantity * price);
    
    // Check position value limit
    if (position_value > limits_.max_position_value) {
        publishRiskEvent({
            RiskEventType::LIMIT_BREACH,
            "Position value exceeds limit",
            symbol,
            0.8,
            std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            {{"requested_value", std::to_string(position_value)},
             {"limit", std::to_string(limits_.max_position_value)}}
        });
        return false;
    }
    
    // Check total exposure
    double new_total_exposure = metrics_.total_exposure + position_value;
    if (new_total_exposure > limits_.max_total_exposure) {
        publishRiskEvent({
            RiskEventType::LIMIT_BREACH,
            "Total exposure exceeds limit",
            symbol,
            0.9,
            std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            {{"new_exposure", std::to_string(new_total_exposure)},
             {"limit", std::to_string(limits_.max_total_exposure)}}
        });
        return false;
    }
    
    // Check position count
    if (positions_.find(symbol) == positions_.end() && 
        positions_.size() >= limits_.max_position_count) {
        return false;
    }
    
    // Check leverage
    double new_leverage = new_total_exposure / peak_portfolio_value_;
    if (new_leverage > limits_.max_leverage) {
        return false;
    }
    
    return true;
}

void RiskManager::updatePosition(const std::string& symbol, double quantity, double price) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& position = positions_[symbol];
    
    if (position.quantity == 0) {
        // New position
        position.symbol = symbol;
        position.quantity = quantity;
        position.average_price = price;
        position.realized_pnl = 0;
    } else if ((position.quantity > 0 && quantity > 0) || 
               (position.quantity < 0 && quantity < 0)) {
        // Adding to position
        double total_cost = position.quantity * position.average_price + quantity * price;
        position.quantity += quantity;
        position.average_price = total_cost / position.quantity;
    } else {
        // Reducing or flipping position
        double closed_quantity = std::min(std::abs(position.quantity), std::abs(quantity));
        double pnl = closed_quantity * (price - position.average_price) * 
                    (position.quantity > 0 ? 1 : -1);
        
        position.realized_pnl += pnl;
        metrics_.daily_pnl += pnl;
        
        position.quantity += quantity;
        if (std::abs(position.quantity) < 1e-9) {
            // Position closed
            position.quantity = 0;
            position.average_price = 0;
        } else if ((position.quantity > 0) != ((position.quantity - quantity) > 0)) {
            // Position flipped
            position.average_price = price;
        }
    }
    
    position.last_update = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    
    if (position.quantity == 0 && position.realized_pnl == 0) {
        positions_.erase(symbol);
    }
    
    updateMetrics();
    checkLimits();
}

void RiskManager::markToMarket(const std::string& symbol, double market_price) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        return;
    }
    
    auto& position = it->second;
    position.market_value = position.quantity * market_price;
    position.unrealized_pnl = position.quantity * (market_price - position.average_price);
    position.last_update = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    
    // Check stop loss
    double loss_percent = -position.unrealized_pnl / std::abs(position.market_value);
    if (loss_percent > limits_.stop_loss_percent) {
        publishRiskEvent({
            RiskEventType::STOP_LOSS_TRIGGERED,
            "Stop loss triggered",
            symbol,
            1.0,
            position.last_update,
            {{"loss_percent", std::to_string(loss_percent * 100)},
             {"position_size", std::to_string(position.quantity)}}
        });
    }
    
    updateMetrics();
    checkLimits();
}

bool RiskManager::validateOrder(const std::string& symbol, Side side, 
                               double quantity, double price) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (emergency_stop_) {
        return false;
    }
    
    double order_value = quantity * price;
    
    // Check single order limits
    if (quantity > limits_.max_order_size) {
        return false;
    }
    
    if (order_value > limits_.max_order_value) {
        return false;
    }
    
    // Check if order would breach position limits
    auto it = positions_.find(symbol);
    double current_position = (it != positions_.end()) ? it->second.quantity : 0;
    double new_position = current_position + (side == Side::BUY ? quantity : -quantity);
    
    return canTakePosition(symbol, new_position - current_position, price);
}

bool RiskManager::checkOrderLimits(double order_value, int current_open_orders) {
    if (order_value > limits_.max_order_value) {
        return false;
    }
    
    if (current_open_orders >= limits_.max_open_orders) {
        return false;
    }
    
    return true;
}

bool RiskManager::isWithinRiskLimits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check daily loss
    if (metrics_.daily_pnl < -limits_.max_daily_loss) {
        return false;
    }
    
    // Check drawdown
    if (metrics_.current_drawdown > limits_.max_drawdown) {
        return false;
    }
    
    // Check leverage
    if (metrics_.leverage > limits_.max_leverage) {
        return false;
    }
    
    return true;
}

void RiskManager::triggerEmergencyStop(const std::string& reason) {
    emergency_stop_ = true;
    
    publishRiskEvent({
        RiskEventType::CIRCUIT_BREAKER,
        "Emergency stop triggered: " + reason,
        "",
        1.0,
        std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        {}
    });
}

void RiskManager::realizePnL(const std::string& symbol, double amount) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        it->second.realized_pnl += amount;
    }
    
    metrics_.daily_pnl += amount;
    updateMetrics();
}

double RiskManager::getUnrealizedPnL() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double total = 0.0;
    for (const auto& [symbol, position] : positions_) {
        total += position.unrealized_pnl;
    }
    return total;
}

double RiskManager::getRealizedPnL() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double total = 0.0;
    for (const auto& [symbol, position] : positions_) {
        total += position.realized_pnl;
    }
    return total;
}

double RiskManager::getTotalPnL() const {
    return getUnrealizedPnL() + getRealizedPnL();
}

Position RiskManager::getPosition(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        return it->second;
    }
    
    return Position{symbol, 0, 0, 0, 0, 0, Timestamp(0)};
}

std::vector<Position> RiskManager::getAllPositions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<Position> result;
    for (const auto& [symbol, position] : positions_) {
        result.push_back(position);
    }
    return result;
}

void RiskManager::addRiskHandler(std::function<void(const RiskEvent&)> handler) {
    risk_handlers_.push_back(handler);
}

void RiskManager::publishRiskEvent(const RiskEvent& event) {
    // Add to queue
    risk_event_queue_.push(event);
    
    // Notify handlers
    for (auto& handler : risk_handlers_) {
        handler(event);
    }
}

void RiskManager::updateLimits(const RiskLimits& new_limits) {
    std::lock_guard<std::mutex> lock(mutex_);
    limits_ = new_limits;
    checkLimits();
}

void RiskManager::updateMetrics() {
    // Calculate exposures
    metrics_.total_exposure = 0;
    metrics_.net_exposure = 0;
    metrics_.gross_exposure = 0;
    
    for (const auto& [symbol, position] : positions_) {
        double position_value = std::abs(position.market_value);
        metrics_.total_exposure += position_value;
        metrics_.net_exposure += position.market_value;
        metrics_.gross_exposure += position_value;
    }
    
    // Update portfolio value
    double portfolio_value = metrics_.net_exposure + getRealizedPnL();
    portfolio_values_.push_back(portfolio_value);
    
    if (portfolio_values_.size() > 1000) {
        portfolio_values_.pop_front();
    }
    
    // Update peak and drawdown
    if (portfolio_value > peak_portfolio_value_) {
        peak_portfolio_value_ = portfolio_value;
    }
    
    if (peak_portfolio_value_ > 0) {
        metrics_.current_drawdown = (peak_portfolio_value_ - portfolio_value) / 
                                   peak_portfolio_value_;
        metrics_.max_drawdown = std::max(metrics_.max_drawdown, metrics_.current_drawdown);
    }
    
    // Calculate leverage
    if (portfolio_value > 0) {
        metrics_.leverage = metrics_.gross_exposure / portfolio_value;
    }
    
    // Calculate VaR
    calculateVaR();
    
    // Update timestamp
    metrics_.last_update = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
}

void RiskManager::checkLimits() {
    // Check daily loss limit
    if (metrics_.daily_pnl < -limits_.max_daily_loss) {
        triggerEmergencyStop("Daily loss limit breached");
    }
    
    // Check drawdown limit
    if (metrics_.current_drawdown > limits_.max_drawdown) {
        triggerEmergencyStop("Maximum drawdown breached");
    }
    
    // Check leverage limit
    if (metrics_.leverage > limits_.max_leverage) {
        publishRiskEvent({
            RiskEventType::LIMIT_BREACH,
            "Leverage limit breached",
            "",
            0.9,
            metrics_.last_update,
            {{"current_leverage", std::to_string(metrics_.leverage)},
             {"limit", std::to_string(limits_.max_leverage)}}
        });
    }
}

void RiskManager::calculateVaR() {
    if (daily_returns_.size() < 20) {
        return;  // Not enough data
    }
    
    // Simple historical VaR
    std::vector<double> sorted_returns(daily_returns_.begin(), daily_returns_.end());
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t var95_index = static_cast<size_t>(sorted_returns.size() * 0.05);
    size_t var99_index = static_cast<size_t>(sorted_returns.size() * 0.01);
    
    metrics_.var_95 = -sorted_returns[var95_index] * metrics_.total_exposure;
    metrics_.var_99 = -sorted_returns[var99_index] * metrics_.total_exposure;
    
    // Expected shortfall (average of returns worse than VaR)
    double sum_tail = 0;
    for (size_t i = 0; i <= var95_index; ++i) {
        sum_tail += sorted_returns[i];
    }
    
    metrics_.expected_shortfall = -sum_tail / (var95_index + 1) * metrics_.total_exposure;
}

// DynamicRiskManager implementation
DynamicRiskManager::DynamicRiskManager(const RiskLimits& limits, const HedgeParams& hedge_params)
    : RiskManager(limits), hedge_params_(hedge_params) {
    
    last_hedge_time_ = std::chrono::high_resolution_clock::now();
    ml_risk_model_ = std::make_shared<PricePredictionModel>();
}

bool DynamicRiskManager::canTakePosition(const std::string& symbol, double quantity, double price) {
    // First check basic limits
    if (!RiskManager::canTakePosition(symbol, quantity, price)) {
        return false;
    }
    
    // Additional checks with correlation consideration
    double portfolio_risk = calculatePortfolioRisk();
    
    // Estimate marginal risk contribution
    double position_value = quantity * price;
    double marginal_risk = 0.0;
    
    for (const auto& [other_symbol, position] : positions_) {
        if (other_symbol != symbol) {
            double correlation = getCorrelation(symbol, other_symbol);
            marginal_risk += correlation * position.market_value * position_value / 
                           metrics_.total_exposure;
        }
    }
    
    // Check if new position increases risk too much
    double risk_increase = marginal_risk / portfolio_risk;
    if (risk_increase > 0.2) {  // More than 20% increase in risk
        publishRiskEvent({
            RiskEventType::LIMIT_BREACH,
            "Position would increase portfolio risk too much",
            symbol,
            0.7,
            std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            {{"risk_increase", std::to_string(risk_increase * 100) + "%"}}
        });
        return false;
    }
    
    return true;
}

void DynamicRiskManager::updateCorrelation(const std::string& symbol1, 
                                          const std::string& symbol2, 
                                          double correlation) {
    std::lock_guard<std::mutex> lock(mutex_);
    correlations_[symbol1][symbol2] = correlation;
    correlations_[symbol2][symbol1] = correlation;
}

double DynamicRiskManager::getCorrelation(const std::string& symbol1, 
                                         const std::string& symbol2) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (symbol1 == symbol2) {
        return 1.0;
    }
    
    auto it1 = correlations_.find(symbol1);
    if (it1 != correlations_.end()) {
        auto it2 = it1->second.find(symbol2);
        if (it2 != it1->second.end()) {
            return it2->second;
        }
    }
    
    return 0.0;  // Assume uncorrelated if not specified
}

void DynamicRiskManager::addStressScenario(const StressScenario& scenario) {
    stress_scenarios_.push_back(scenario);
}

double DynamicRiskManager::runStressTest(const std::string& scenario_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = std::find_if(stress_scenarios_.begin(), stress_scenarios_.end(),
                          [&scenario_name](const StressScenario& s) {
                              return s.name == scenario_name;
                          });
    
    if (it == stress_scenarios_.end()) {
        return 0.0;
    }
    
    double total_impact = 0.0;
    
    for (const auto& [symbol, position] : positions_) {
        auto shock_it = it->price_shocks.find(symbol);
        if (shock_it != it->price_shocks.end()) {
            double price_change = shock_it->second;
            double impact = position.quantity * position.average_price * price_change;
            total_impact += impact;
        }
    }
    
    return total_impact;
}

std::unordered_map<std::string, double> DynamicRiskManager::runAllStressTests() {
    std::unordered_map<std::string, double> results;
    
    for (const auto& scenario : stress_scenarios_) {
        results[scenario.name] = runStressTest(scenario.name);
    }
    
    return results;
}

bool DynamicRiskManager::shouldRebalanceHedge() const {
    if (!hedge_params_.auto_hedge_enabled) {
        return false;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_hedge_time_).count();
    
    return elapsed >= hedge_params_.hedge_rebalance_interval_ms;
}

std::vector<std::pair<std::string, double>> DynamicRiskManager::getHedgeRecommendations() {
    std::vector<std::pair<std::string, double>> hedges;
    
    // Calculate current portfolio Greeks
    calculateGreeks();
    
    // Check if delta hedging needed
    if (std::abs(metrics_.portfolio_delta) > hedge_params_.delta_hedge_threshold) {
        // Find best hedge instrument
        for (const auto& hedge_instrument : hedge_params_.hedge_instruments) {
            // Simple hedge: opposite position in correlated instrument
            double hedge_size = -metrics_.portfolio_delta;
            hedges.push_back({hedge_instrument, hedge_size});
            break;  // Use first available for now
        }
    }
    
    // Check if gamma hedging needed
    if (std::abs(metrics_.portfolio_gamma) > hedge_params_.gamma_hedge_threshold) {
        // Gamma hedging typically requires options
        // Simplified: suggest ATM options
        hedges.push_back({"SPY_OPTIONS", -metrics_.portfolio_gamma * 100});
    }
    
    return hedges;
}

double DynamicRiskManager::calculatePortfolioRisk() {
    if (positions_.empty()) {
        return 0.0;
    }
    
    // Build covariance matrix
    std::vector<std::string> symbols;
    std::vector<double> weights;
    
    for (const auto& [symbol, position] : positions_) {
        symbols.push_back(symbol);
        weights.push_back(position.market_value / metrics_.total_exposure);
    }
    
    // Calculate portfolio variance
    double portfolio_variance = 0.0;
    
    for (size_t i = 0; i < symbols.size(); ++i) {
        for (size_t j = 0; j < symbols.size(); ++j) {
            double correlation = getCorrelation(symbols[i], symbols[j]);
            double vol_i = 0.02;  // Placeholder - should use actual volatilities
            double vol_j = 0.02;
            
            portfolio_variance += weights[i] * weights[j] * correlation * vol_i * vol_j;
        }
    }
    
    return std::sqrt(portfolio_variance) * metrics_.total_exposure;
}

// CapitalAllocator implementation
CapitalAllocator::CapitalAllocator(const AllocationConstraints& constraints)
    : constraints_(constraints) {}

void CapitalAllocator::updateStrategyPerformance(const std::string& strategy_id,
                                                const std::vector<double>& returns) {
    strategy_returns_[strategy_id] = returns;
    
    // Calculate Sharpe ratio
    if (returns.size() >= 20) {
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / 
                           returns.size();
        
        double variance = 0.0;
        for (double r : returns) {
            variance += std::pow(r - mean_return, 2);
        }
        variance /= returns.size();
        
        double sharpe = (mean_return - constraints_.risk_free_rate) / std::sqrt(variance);
        strategy_sharpe_ratios_[strategy_id] = sharpe;
        
        // Calculate max drawdown
        double peak = 0.0;
        double max_dd = 0.0;
        double cumulative = 1.0;
        
        for (double r : returns) {
            cumulative *= (1 + r);
            if (cumulative > peak) {
                peak = cumulative;
            }
            double dd = (peak - cumulative) / peak;
            max_dd = std::max(max_dd, dd);
        }
        
        strategy_max_drawdowns_[strategy_id] = max_dd;
    }
}

std::unordered_map<std::string, double> CapitalAllocator::optimizeAllocation(
    const std::vector<std::string>& strategies,
    const std::string& method) {
    
    std::unordered_map<std::string, double> allocations;
    
    if (method == "equal_weight") {
        // Simple equal weighting
        double weight = 1.0 / strategies.size();
        for (const auto& strategy : strategies) {
            allocations[strategy] = weight * constraints_.total_capital;
        }
        
    } else if (method == "sharpe_weighted") {
        // Weight by Sharpe ratio
        double total_sharpe = 0.0;
        for (const auto& strategy : strategies) {
            auto it = strategy_sharpe_ratios_.find(strategy);
            if (it != strategy_sharpe_ratios_.end() && it->second > 0) {
                total_sharpe += it->second;
            }
        }
        
        if (total_sharpe > 0) {
            for (const auto& strategy : strategies) {
                auto it = strategy_sharpe_ratios_.find(strategy);
                if (it != strategy_sharpe_ratios_.end() && it->second > 0) {
                    double weight = it->second / total_sharpe;
                    allocations[strategy] = weight * constraints_.total_capital;
                }
            }
        }
        
    } else if (method == "mean_variance") {
        // Simplified mean-variance optimization
        // Would need full covariance matrix in practice
        
        // For now, use inverse volatility weighting
        double total_inv_vol = 0.0;
        std::unordered_map<std::string, double> volatilities;
        
        for (const auto& strategy : strategies) {
            auto it = strategy_returns_.find(strategy);
            if (it != strategy_returns_.end() && it->second.size() > 1) {
                double variance = 0.0;
                double mean = std::accumulate(it->second.begin(), it->second.end(), 0.0) / 
                             it->second.size();
                
                for (double r : it->second) {
                    variance += std::pow(r - mean, 2);
                }
                variance /= it->second.size();
                
                double vol = std::sqrt(variance);
                volatilities[strategy] = vol;
                total_inv_vol += 1.0 / vol;
            }
        }
        
        for (const auto& [strategy, vol] : volatilities) {
            double weight = (1.0 / vol) / total_inv_vol;
            weight = std::max(constraints_.min_allocation_percent, 
                            std::min(constraints_.max_allocation_percent, weight));
            allocations[strategy] = weight * constraints_.total_capital;
        }
    }
    
    return allocations;
}

double CapitalAllocator::getRecommendedSize(const std::string& strategy_id,
                                           double signal_strength,
                                           double current_exposure) {
    // Kelly criterion with safety factor
    auto it = strategy_sharpe_ratios_.find(strategy_id);
    if (it == strategy_sharpe_ratios_.end()) {
        return 0.0;
    }
    
    double sharpe = it->second;
    double kelly_fraction = sharpe / 2.0;  // Simplified Kelly for Gaussian returns
    
    // Apply safety factor (1/4 Kelly)
    kelly_fraction *= 0.25;
    
    // Adjust for signal strength
    kelly_fraction *= signal_strength;
    
    // Calculate position size
    double target_exposure = kelly_fraction * constraints_.total_capital;
    double position_change = target_exposure - current_exposure;
    
    // Apply constraints
    if (std::abs(target_exposure) > constraints_.max_allocation_percent * constraints_.total_capital) {
        target_exposure = constraints_.max_allocation_percent * constraints_.total_capital * 
                         (target_exposure > 0 ? 1 : -1);
        position_change = target_exposure - current_exposure;
    }
    
    return position_change;
}

std::unordered_map<std::string, double> CapitalAllocator::riskParityAllocation(
    const std::vector<std::string>& strategies) {
    
    // Risk parity: equal risk contribution from each strategy
    std::unordered_map<std::string, double> allocations;
    std::unordered_map<std::string, double> volatilities;
    
    // Calculate volatilities
    for (const auto& strategy : strategies) {
        auto it = strategy_returns_.find(strategy);
        if (it != strategy_returns_.end() && it->second.size() > 1) {
            double variance = 0.0;
            double mean = std::accumulate(it->second.begin(), it->second.end(), 0.0) / 
                         it->second.size();
            
            for (double r : it->second) {
                variance += std::pow(r - mean, 2);
            }
            variance /= it->second.size();
            
            volatilities[strategy] = std::sqrt(variance);
        }
    }
    
    // Allocate inversely proportional to volatility
    double sum_inv_vol = 0.0;
    for (const auto& [strategy, vol] : volatilities) {
        sum_inv_vol += 1.0 / vol;
    }
    
    for (const auto& [strategy, vol] : volatilities) {
        double weight = (1.0 / vol) / sum_inv_vol;
        allocations[strategy] = weight * constraints_.total_capital;
    }
    
    return allocations;
}

// IntegratedRiskSystem implementation
IntegratedRiskSystem::IntegratedRiskSystem(
    const RiskLimits& risk_limits,
    const CapitalAllocator::AllocationConstraints& capital_constraints,
    const MarginManager::MarginRequirements& margin_requirements,
    const ComplianceManager::ComplianceRules& compliance_rules) {
    
    risk_manager_ = std::make_shared<DynamicRiskManager>(
        risk_limits, 
        DynamicRiskManager::HedgeParams{true, 0.1, 0.05, 60000, {"SPY", "IWM", "QQQ"}}
    );
    
    capital_allocator_ = std::make_shared<CapitalAllocator>(capital_constraints);
    margin_manager_ = std::make_shared<MarginManager>(margin_requirements);
    compliance_manager_ = std::make_shared<ComplianceManager>(compliance_rules);
}

bool IntegratedRiskSystem::approveOrder(const std::string& symbol, Side side,
                                       double quantity, double price,
                                       std::vector<std::string>& rejection_reasons) {
    rejection_reasons.clear();
    
    // Risk checks
    if (!risk_manager_->validateOrder(symbol, side, quantity, price)) {
        rejection_reasons.push_back("Failed risk validation");
    }
    
    // Margin checks
    double required_margin = margin_manager_->calculateInitialMargin(symbol, quantity, price);
    if (!margin_manager_->hasAdequateMargin(required_margin)) {
        rejection_reasons.push_back("Insufficient margin");
    }
    
    // Compliance checks
    if (!compliance_manager_->isOrderCompliant(symbol, side, quantity, price)) {
        rejection_reasons.push_back("Compliance violation");
    }
    
    return rejection_reasons.empty();
}

double IntegratedRiskSystem::calculateOptimalPositionSize(const std::string& symbol,
                                                         double signal_strength,
                                                         double target_risk) {
    // Get current position
    auto position = risk_manager_->getPosition(symbol);
    double current_exposure = position.quantity * position.average_price;
    
    // Get capital allocation recommendation
    double recommended_size = capital_allocator_->getRecommendedSize(
        symbol, signal_strength, current_exposure
    );
    
    // Apply risk constraints
    if (!risk_manager_->canTakePosition(symbol, recommended_size, position.average_price)) {
        recommended_size = 0.0;
    }
    
    // Check margin requirements
    double required_margin = margin_manager_->calculateInitialMargin(
        symbol, std::abs(recommended_size), position.average_price
    );
    
    if (!margin_manager_->hasAdequateMargin(required_margin)) {
        // Scale down to available margin
        double available = margin_manager_->getExcessLiquidity();
        recommended_size *= available / required_margin;
    }
    
    return recommended_size;
}

void IntegratedRiskSystem::monitorRisk() {
    // Real-time monitoring loop
    while (true) {
        // Check risk metrics
        if (!risk_manager_->isWithinRiskLimits()) {
            handleRiskEvent({
                RiskEventType::LIMIT_BREACH,
                "Risk limits breached",
                "",
                1.0,
                std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
                {}
            });
        }
        
        // Check margin
        if (margin_manager_->isInMarginCall()) {
            handleRiskEvent({
                RiskEventType::MARGIN_CALL,
                "Margin call triggered",
                "",
                1.0,
                std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
                {}
            });
        }
        
        // Check for hedge rebalancing
        if (risk_manager_->shouldRebalanceHedge()) {
            auto hedges = risk_manager_->getHedgeRecommendations();
            // Process hedge recommendations
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void IntegratedRiskSystem::handleRiskEvent(const RiskEvent& event) {
    switch (event.type) {
        case RiskEventType::LIMIT_BREACH:
            // Reduce positions or stop trading
            if (event.severity > 0.8) {
                risk_manager_->triggerEmergencyStop("Severe limit breach");
            }
            break;
            
        case RiskEventType::MARGIN_CALL:
            // Liquidate positions to meet margin
            {
                auto requirements = margin_manager_->getMarginCallRequirements();
                // Process liquidation
            }
            break;
            
        case RiskEventType::STOP_LOSS_TRIGGERED:
            // Close position
            // Send market order to close
            break;
            
        default:
            break;
    }
}

} // namespace hft