#include "strategies/market_maker_strategy.h"
#include <cmath>
#include <algorithm>

namespace hft {

// Base MarketMakingStrategy implementations
bool MarketMakingStrategy::isWithinRiskLimits() const {
    // Check position limits
    if (std::abs(market_state_.current_position) >= risk_params_.max_position) {
        return false;
    }
    
    // Check drawdown limits
    if (metrics_.max_drawdown > risk_params_.max_drawdown) {
        return false;
    }
    
    // Check adverse selection threshold
    if (market_state_.order_flow_toxicity > risk_params_.adverse_selection_threshold) {
        return false;
    }
    
    return true;
}

double MarketMakingStrategy::getInventoryPenalty() const {
    double position_ratio = market_state_.current_position / risk_params_.max_position;
    return risk_params_.inventory_penalty * std::pow(position_ratio, 2);
}

// Avellaneda-Stoikov Strategy Implementation
AvellanedaStoikovStrategy::AvellanedaStoikovStrategy(
    const RiskParams& params, double gamma, double k, double T)
    : MarketMakingStrategy(params), gamma_(gamma), k_(k), T_(T) {
    
    q_max_ = params.max_position;
    sigma_ = 0.001;  // Initial volatility estimate (0.1%)
    reservation_price_ = 0.0;
    optimal_spread_ = params.min_spread;
}

std::pair<std::optional<double>, std::optional<double>> 
AvellanedaStoikovStrategy::getQuotes(const OrderBook& book, const MarketState& state) {
    
    // Update market state
    updateState(state);
    
    // Check risk limits
    if (!isWithinRiskLimits()) {
        return {std::nullopt, std::nullopt};
    }
    
    // Get mid price
    double mid_price = book.getMidPrice();
    if (mid_price == 0) {
        return {std::nullopt, std::nullopt};
    }
    
    // Calculate time remaining (as fraction of trading day)
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_open = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count() % 86400;  // Seconds since midnight
    double market_open_seconds = 9.5 * 3600;     // 9:30 AM
    double market_close_seconds = 16 * 3600;     // 4:00 PM
    double trading_seconds = market_close_seconds - market_open_seconds;
    double elapsed_seconds = std::max(0.0, time_since_open - market_open_seconds);
    double time_remaining = std::max(0.0, 1.0 - elapsed_seconds / trading_seconds);
    
    // Update volatility estimate
    sigma_ = state.current_volatility > 0 ? state.current_volatility : sigma_;
    
    // Calculate reservation price
    reservation_price_ = calculateReservationPrice(
        mid_price, 
        state.current_position, 
        time_remaining
    );
    
    // Calculate optimal spread
    optimal_spread_ = calculateOptimalSpread(sigma_, gamma_, time_remaining);
    
    // Apply inventory penalty
    double inventory_penalty = inventoryPenaltyFunction(state.current_position);
    
    // Calculate bid and ask prices
    double half_spread = optimal_spread_ / 2.0;
    double bid_adjustment = half_spread + inventory_penalty;
    double ask_adjustment = half_spread - inventory_penalty;
    
    // Ensure minimum spread
    bid_adjustment = std::max(bid_adjustment, risk_params_.min_spread / 2.0);
    ask_adjustment = std::max(ask_adjustment, risk_params_.min_spread / 2.0);
    
    // Calculate final quotes
    double bid_price = reservation_price_ - bid_adjustment;
    double ask_price = reservation_price_ + ask_adjustment;
    
    // Skew quotes based on inventory
    if (state.current_position > 0) {
        // Long inventory - be more aggressive on ask
        ask_price -= optimal_spread_ * 0.1 * (state.current_position / q_max_);
    } else if (state.current_position < 0) {
        // Short inventory - be more aggressive on bid
        bid_price += optimal_spread_ * 0.1 * (std::abs(state.current_position) / q_max_);
    }
    
    // Ensure quotes are better than current book
    double best_bid = book.getBestBid();
    double best_ask = book.getBestAsk();
    
    if (best_bid > 0 && bid_price >= best_bid) {
        bid_price = best_bid - 0.01;  // Penny improvement
    }
    if (best_ask > 0 && ask_price <= best_ask) {
        ask_price = best_ask + 0.01;  // Penny improvement
    }
    
    // Final sanity checks
    if (bid_price <= 0 || ask_price <= 0 || bid_price >= ask_price) {
        return {std::nullopt, std::nullopt};
    }
    
    // Return quotes only if we can place orders
    std::optional<double> bid_quote = std::nullopt;
    std::optional<double> ask_quote = std::nullopt;
    
    // Check position limits for each side
    if (state.current_position < risk_params_.max_position - risk_params_.position_limit_buffer) {
        bid_quote = bid_price;
    }
    if (state.current_position > -risk_params_.max_position + risk_params_.position_limit_buffer) {
        ask_quote = ask_price;
    }
    
    return {bid_quote, ask_quote};
}

double AvellanedaStoikovStrategy::calculateReservationPrice(
    double mid_price, double inventory, double time_remaining) {
    
    // Avellaneda-Stoikov reservation price formula
    // r = s - q * gamma * sigma^2 * (T - t)
    // where:
    // s = mid price
    // q = inventory
    // gamma = risk aversion
    // sigma = volatility
    // T - t = time remaining
    
    double inventory_adjustment = inventory * gamma_ * std::pow(sigma_, 2) * time_remaining;
    return mid_price - inventory_adjustment;
}

double AvellanedaStoikovStrategy::calculateOptimalSpread(
    double volatility, double gamma, double time_remaining) {
    
    // Avellaneda-Stoikov optimal spread formula
    // delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    // where:
    // gamma = risk aversion
    // sigma = volatility  
    // T - t = time remaining
    // k = order arrival rate
    
    double term1 = gamma * std::pow(volatility, 2) * time_remaining;
    double term2 = (2.0 / gamma) * std::log(1.0 + gamma / k_);
    
    double spread = term1 + term2;
    
    // Apply bounds
    spread = std::max(risk_params_.min_spread, spread);
    spread = std::min(risk_params_.max_spread, spread);
    
    // Adjust for market conditions
    spread *= (1.0 + risk_params_.volatility_factor * (volatility / 0.001 - 1.0));
    
    return spread;
}

double AvellanedaStoikovStrategy::inventoryPenaltyFunction(double inventory) {
    // Quadratic penalty function for inventory risk
    // Penalty increases as inventory approaches limits
    
    double inventory_ratio = inventory / q_max_;
    double base_penalty = std::pow(inventory_ratio, 2) * gamma_ * sigma_;
    
    // Exponential penalty near limits
    if (std::abs(inventory_ratio) > 0.8) {
        base_penalty *= std::exp(5.0 * (std::abs(inventory_ratio) - 0.8));
    }
    
    return base_penalty * risk_params_.inventory_penalty;
}

void AvellanedaStoikovStrategy::updateVolatility(double new_sigma) {
    // Exponential moving average update
    const double alpha = 0.1;  // Smoothing factor
    sigma_ = alpha * new_sigma + (1.0 - alpha) * sigma_;
}

void AvellanedaStoikovStrategy::updateArrivalRate(double new_k) {
    // Update order arrival rate estimate
    const double alpha = 0.05;  // Slower adaptation for arrival rate
    k_ = alpha * new_k + (1.0 - alpha) * k_;
}

// AlphaMarketMaker Implementation
AlphaMarketMaker::AlphaMarketMaker(const RiskParams& params) 
    : MarketMakingStrategy(params) {
    
    // Initialize default alpha signals
    addAlphaSignal("microprice", 0.3);
    addAlphaSignal("order_flow", 0.25);
    addAlphaSignal("queue", 0.2);
    addAlphaSignal("spread", 0.25);
}

std::pair<std::optional<double>, std::optional<double>>
AlphaMarketMaker::getQuotes(const OrderBook& book, const MarketState& state) {
    
    updateState(state);
    
    if (!isWithinRiskLimits()) {
        return {std::nullopt, std::nullopt};
    }
    
    // Calculate all alpha signals
    alpha_signals_[0].value = calculateMicropriceAlpha(book);
    alpha_signals_[1].value = calculateOrderFlowAlpha(book);
    alpha_signals_[2].value = calculateQueueAlpha(book);
    alpha_signals_[3].value = calculateSpreadAlpha(book);
    
    // Combine alphas into directional signal
    double combined_alpha = combineAlphas();
    
    // Get base prices
    double mid_price = book.getMidPrice();
    double best_bid = book.getBestBid();
    double best_ask = book.getBestAsk();
    
    if (mid_price == 0) {
        return {std::nullopt, std::nullopt};
    }
    
    // Calculate base spread based on volatility
    double base_spread = risk_params_.min_spread + 
                        risk_params_.volatility_factor * state.current_volatility;
    
    // Adjust spread based on alpha signal
    double alpha_adjustment = combined_alpha * base_spread * 0.5;
    
    // Calculate quotes with alpha adjustment
    double bid_price = mid_price - base_spread / 2.0 - alpha_adjustment;
    double ask_price = mid_price + base_spread / 2.0 - alpha_adjustment;
    
    // Apply inventory skew
    double inventory_skew = getInventoryPenalty() * base_spread;
    bid_price -= inventory_skew;
    ask_price += inventory_skew;
    
    // Ensure competitive quotes
    if (best_bid > 0 && bid_price >= best_bid) {
        bid_price = best_bid - 0.01;
    }
    if (best_ask > 0 && ask_price <= best_ask) {
        ask_price = best_ask + 0.01;
    }
    
    // Position limit checks
    std::optional<double> bid_quote = std::nullopt;
    std::optional<double> ask_quote = std::nullopt;
    
    if (state.current_position < risk_params_.max_position * 0.9) {
        bid_quote = bid_price;
    }
    if (state.current_position > -risk_params_.max_position * 0.9) {
        ask_quote = ask_price;
    }
    
    return {bid_quote, ask_quote};
}

double AlphaMarketMaker::calculateMicropriceAlpha(const OrderBook& book) {
    // Alpha based on microprice vs mid price divergence
    double mid_price = book.getMidPrice();
    double microprice = book.calculateMicroprice(5);
    
    if (mid_price == 0) return 0.0;
    
    double divergence = (microprice - mid_price) / mid_price;
    
    // Normalize to [-1, 1]
    return std::tanh(divergence * 1000);  // Scale factor for sensitivity
}

double AlphaMarketMaker::calculateOrderFlowAlpha(const OrderBook& book) {
    // Alpha based on order flow imbalance
    double imbalance = book.getOrderFlowImbalance(5000);  // 5 second window
    
    // Add momentum component
    double recent_imbalance = book.getOrderFlowImbalance(1000);  // 1 second
    double momentum = recent_imbalance - imbalance;
    
    return 0.7 * imbalance + 0.3 * momentum;
}

double AlphaMarketMaker::calculateQueueAlpha(const OrderBook& book) {
    // Alpha based on queue dynamics
    auto bid_levels = book.getBidLevels(3);
    auto ask_levels = book.getAskLevels(3);
    
    if (bid_levels.empty() || ask_levels.empty()) return 0.0;
    
    // Calculate relative queue sizes
    double total_bid_size = 0;
    double total_ask_size = 0;
    
    for (const auto& [price, size] : bid_levels) {
        total_bid_size += size;
    }
    for (const auto& [price, size] : ask_levels) {
        total_ask_size += size;
    }
    
    if (total_bid_size + total_ask_size == 0) return 0.0;
    
    // Queue imbalance
    double queue_imbalance = (total_bid_size - total_ask_size) / 
                            (total_bid_size + total_ask_size);
    
    // Adjust for spread width (tighter spreads = stronger signal)
    double spread = book.getSpread();
    double spread_factor = std::exp(-spread / risk_params_.min_spread);
    
    return queue_imbalance * spread_factor;
}

double AlphaMarketMaker::calculateSpreadAlpha(const OrderBook& book) {
    // Alpha based on spread dynamics
    double current_spread = book.getSpread();
    double mid_price = book.getMidPrice();
    
    if (mid_price == 0) return 0.0;
    
    // Relative spread
    double rel_spread = current_spread / mid_price;
    
    // Compare to expected spread based on volatility
    double expected_spread = risk_params_.min_spread + 
                           2 * market_state_.current_volatility * std::sqrt(1.0/252);
    
    // Signal: tight spreads suggest informed trading
    double spread_signal = (expected_spread - rel_spread) / expected_spread;
    
    return std::tanh(spread_signal * 5);  // Scale and bound
}

double AlphaMarketMaker::combineAlphas() {
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    
    for (const auto& signal : alpha_signals_) {
        weighted_sum += signal.value * signal.weight * signal.confidence;
        weight_sum += signal.weight * signal.confidence;
    }
    
    return (weight_sum > 0) ? weighted_sum / weight_sum : 0.0;
}

void AlphaMarketMaker::addAlphaSignal(const std::string& name, double weight) {
    alpha_signals_.push_back({name, weight, 0.0, 1.0});
}

// SmartOrderPlacer Implementation
SmartOrderPlacer::SmartOrderPlacer() {}

double SmartOrderPlacer::estimateQueuePosition(Price price, Quantity size, 
                                              const OrderBook& book) {
    // Estimate position in queue based on current book state
    auto levels = (price > book.getMidPrice()) ? 
                  book.getAskLevels(10) : book.getBidLevels(10);
    
    for (const auto& [level_price, level_size] : levels) {
        if (std::abs(level_price - price) < 0.0001) {  // Found our price level
            // Assume uniform distribution in queue
            return level_size / 2.0;
        }
    }
    
    // New price level
    return 0.0;
}

double SmartOrderPlacer::probabilityOfFill(double queue_position, double time_horizon) {
    // Simple exponential decay model for fill probability
    // P(fill) = 1 - exp(-lambda * t / queue_position)
    
    double lambda = 0.1;  // Arrival rate parameter
    
    if (queue_position <= 0) return 1.0;
    
    double fill_prob = 1.0 - std::exp(-lambda * time_horizon / queue_position);
    return std::min(1.0, std::max(0.0, fill_prob));
}

SmartOrderPlacer::PlacementDecision SmartOrderPlacer::getOptimalPlacement(
    Side side, Quantity target_size, const OrderBook& book, double urgency) {
    
    PlacementDecision decision;
    decision.size = target_size;
    
    // Get relevant book levels
    auto levels = (side == Side::BUY) ? book.getBidLevels(10) : book.getAskLevels(10);
    
    if (levels.empty()) {
        // No levels available
        decision.price = (side == Side::BUY) ? 
                        book.getMidPrice() - 0.01 : 
                        book.getMidPrice() + 0.01;
        decision.expected_fill_time = 1000.0;  // High uncertainty
        decision.fill_probability = 0.1;
        return decision;
    }
    
    // Evaluate different price points
    double best_score = -1e9;
    Price best_price = 0;
    double best_fill_time = 0;
    double best_fill_prob = 0;
    
    // Check joining best level
    {
        Price join_price = levels[0].first;
        double queue_pos = estimateQueuePosition(join_price, target_size, book);
        double fill_prob = probabilityOfFill(queue_pos, 60.0);  // 60 second horizon
        double fill_time = queue_pos / 100.0;  // Rough estimate
        
        // Score based on urgency
        double score = urgency * fill_prob - (1 - urgency) * fill_time;
        
        if (score > best_score) {
            best_score = score;
            best_price = join_price;
            best_fill_time = fill_time;
            best_fill_prob = fill_prob;
        }
    }
    
    // Check improving best level
    {
        Price improve_price = (side == Side::BUY) ? 
                             levels[0].first + 0.01 : 
                             levels[0].first - 0.01;
        
        double queue_pos = 0;  // Front of new queue
        double fill_prob = probabilityOfFill(queue_pos, 30.0);  // 30 second horizon
        double fill_time = 5.0;  // Quick fill expected
        
        double score = urgency * fill_prob - (1 - urgency) * fill_time;
        
        if (score > best_score) {
            best_score = score;
            best_price = improve_price;
            best_fill_time = fill_time;
            best_fill_prob = fill_prob;
        }
    }
    
    decision.price = best_price;
    decision.expected_fill_time = best_fill_time;
    decision.fill_probability = best_fill_prob;
    
    return decision;
}

void SmartOrderPlacer::updateQueueModel(Price price, double fill_time, 
                                       Quantity filled_size) {
    // Update queue model with execution data
    auto& model = queue_models_[price];
    
    // Exponential moving average updates
    const double alpha = 0.1;
    
    if (model.queue_lengths.empty()) {
        model.arrival_rate = 1.0 / fill_time;
        model.avg_order_size = filled_size;
    } else {
        model.arrival_rate = alpha * (1.0 / fill_time) + 
                           (1 - alpha) * model.arrival_rate;
        model.avg_order_size = alpha * filled_size + 
                             (1 - alpha) * model.avg_order_size;
    }
    
    model.queue_lengths.push_back(filled_size / fill_time);
    
    // Keep last 1000 observations
    if (model.queue_lengths.size() > 1000) {
        model.queue_lengths.erase(model.queue_lengths.begin());
    }
}

// Market Impact Model Implementation
MarketImpactModel::MarketImpactModel(double perm_impact, double temp_impact, double vol)
    : permanent_impact_coefficient_(perm_impact),
      temporary_impact_coefficient_(temp_impact),
      volatility_(vol) {}

double MarketImpactModel::calculatePermanentImpact(Quantity size, double avg_daily_volume) {
    // Permanent impact: g(v) = gamma * v^alpha
    // where v = size / ADV (participation rate)
    
    if (avg_daily_volume <= 0) return 0.0;
    
    double participation = static_cast<double>(size) / avg_daily_volume;
    double alpha = 0.5;  // Square root impact
    
    return permanent_impact_coefficient_ * std::pow(participation, alpha);
}

double MarketImpactModel::calculateTemporaryImpact(Quantity size, double participation_rate) {
    // Temporary impact: h(v) = eta * v^beta
    
    double beta = 0.5;  // Square root impact
    return temporary_impact_coefficient_ * std::pow(participation_rate, beta);
}

std::vector<Quantity> MarketImpactModel::getOptimalTrajectory(
    Quantity total_size, int time_periods, double risk_aversion) {
    
    // Almgren-Chriss optimal execution trajectory
    // For now, implement simple TWAP with front-loading based on risk aversion
    
    std::vector<Quantity> trajectory(time_periods);
    
    if (risk_aversion < 0.1) {
        // Low risk aversion - spread evenly (TWAP)
        Quantity per_period = total_size / time_periods;
        std::fill(trajectory.begin(), trajectory.end(), per_period);
        
    } else {
        // Higher risk aversion - front load execution
        double decay_rate = 1.0 + risk_aversion;
        double sum = 0.0;
        
        std::vector<double> weights(time_periods);
        for (int i = 0; i < time_periods; ++i) {
            weights[i] = std::exp(-decay_rate * i / time_periods);
            sum += weights[i];
        }
        
        // Normalize and allocate
        Quantity allocated = 0;
        for (int i = 0; i < time_periods - 1; ++i) {
            trajectory[i] = static_cast<Quantity>(total_size * weights[i] / sum);
            allocated += trajectory[i];
        }
        trajectory[time_periods - 1] = total_size - allocated;  // Remainder
    }
    
    return trajectory;
}

// Post Trade Analyzer Implementation
PostTradeAnalyzer::PostTradeAnalyzer() {}

void PostTradeAnalyzer::recordTrade(const TradeRecord& trade) {
    trades_.push_back(trade);
    
    // Keep only last 10000 trades
    if (trades_.size() > 10000) {
        trades_.erase(trades_.begin(), trades_.begin() + 5000);
    }
}

double PostTradeAnalyzer::calculateRealizedSpread() {
    if (trades_.empty()) return 0.0;
    
    double total_spread = 0.0;
    int count = 0;
    
    for (const auto& trade : trades_) {
        if (trade.realized_spread > 0) {
            total_spread += trade.realized_spread;
            count++;
        }
    }
    
    return (count > 0) ? total_spread / count : 0.0;
}

double PostTradeAnalyzer::calculateEffectiveSpread() {
    if (trades_.empty()) return 0.0;
    
    double total_spread = 0.0;
    int count = 0;
    
    for (const auto& trade : trades_) {
        double mid_price = (trade.market_price_before + trade.market_price_after) / 2.0;
        double effective = 2.0 * std::abs(trade.price - mid_price);
        total_spread += effective;
        count++;
    }
    
    return (count > 0) ? total_spread / count : 0.0;
}

double PostTradeAnalyzer::calculatePriceImprovement() {
    if (trades_.empty()) return 0.0;
    
    int improved_count = 0;
    
    for (const auto& trade : trades_) {
        bool improved = false;
        
        if (trade.side == Side::BUY) {
            improved = trade.price < trade.market_price_before;
        } else {
            improved = trade.price > trade.market_price_before;
        }
        
        if (improved) improved_count++;
    }
    
    return static_cast<double>(improved_count) / trades_.size();
}

double PostTradeAnalyzer::calculateAdverseSelection(int time_window_ms) {
    if (trades_.empty()) return 0.0;
    
    double total_adverse = 0.0;
    int count = 0;
    
    for (const auto& trade : trades_) {
        double price_move = trade.market_price_after - trade.price;
        
        if (trade.side == Side::BUY) {
            // For buys, negative move is adverse
            if (price_move < 0) {
                total_adverse += std::abs(price_move);
                count++;
            }
        } else {
            // For sells, positive move is adverse
            if (price_move > 0) {
                total_adverse += price_move;
                count++;
            }
        }
    }
    
    return (count > 0) ? total_adverse / count : 0.0;
}

PostTradeAnalyzer::AnalysisReport PostTradeAnalyzer::generateReport() {
    AnalysisReport report;
    
    report.avg_realized_spread = calculateRealizedSpread();
    report.avg_effective_spread = calculateEffectiveSpread();
    report.price_improvement_rate = calculatePriceImprovement();
    report.adverse_selection_cost = calculateAdverseSelection(5000);  // 5 second window
    
    // Calculate information ratio (realized spread / effective spread)
    if (report.avg_effective_spread > 0) {
        report.information_ratio = report.avg_realized_spread / report.avg_effective_spread;
    } else {
        report.information_ratio = 0.0;
    }
    
    return report;
}

} // namespace hft