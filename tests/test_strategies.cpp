#include <gtest/gtest.h>
#include "strategies/market_maker_strategy.h"
#include "core/order_book.h"
#include <memory>
#include <chrono>

using namespace hft;

class StrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize risk parameters
        risk_params = RiskParams{
            100000.0,   // max_position_value
            10000.0,    // buffer_amount
            0.0001,     // inventory_penalty
            0.0001,     // min_spread
            0.0010,     // max_spread
            100.0,      // default_size
            1.5,        // volatility_factor
            0.7,        // toxicity_threshold
            50000.0,    // max_drawdown
            2.0         // target_sharpe
        };
        
        // Create order book
        order_book = std::make_shared<OrderBook>();
        
        // Add some test orders
        auto bid1 = std::make_shared<Order>(1, Side::BUY, 99.95, 1000, 
            std::chrono::nanoseconds(1000));
        auto bid2 = std::make_shared<Order>(2, Side::BUY, 99.90, 2000, 
            std::chrono::nanoseconds(2000));
        auto ask1 = std::make_shared<Order>(3, Side::SELL, 100.05, 1500, 
            std::chrono::nanoseconds(3000));
        auto ask2 = std::make_shared<Order>(4, Side::SELL, 100.10, 2500, 
            std::chrono::nanoseconds(4000));
        
        order_book->addOrder(bid1);
        order_book->addOrder(bid2);
        order_book->addOrder(ask1);
        order_book->addOrder(ask2);
    }
    
    RiskParams risk_params;
    std::shared_ptr<OrderBook> order_book;
};

// Test Avellaneda-Stoikov Strategy
TEST_F(StrategyTest, AvellanedaStoikovBasicQuotes) {
    // Create strategy
    auto strategy = std::make_unique<AvellanedaStoikovStrategy>(
        risk_params, 0.1, 10.0, 300.0
    );
    
    // Create market state
    MarketState state;
    state.current_position = 0;
    state.unrealized_pnl = 0;
    state.realized_pnl = 0;
    state.current_volatility = 0.02;
    state.volume_imbalance = 0.1;
    state.order_flow_toxicity = 0.3;
    state.spread_percentile = 0.5;
    state.queue_position = 0;
    state.last_update = std::chrono::nanoseconds(5000);
    
    // Get quotes
    auto [bid_price, ask_price] = strategy->getQuotes(*order_book, state);
    
    // Verify quotes are generated
    ASSERT_TRUE(bid_price.has_value());
    ASSERT_TRUE(ask_price.has_value());
    
    // Verify spread constraints
    double spread = ask_price.value() - bid_price.value();
    EXPECT_GE(spread, risk_params.min_spread);
    EXPECT_LE(spread, risk_params.max_spread);
    
    // Verify quotes are inside the market
    EXPECT_GT(bid_price.value(), order_book->getBestBid());
    EXPECT_LT(ask_price.value(), order_book->getBestAsk());
}

TEST_F(StrategyTest, AvellanedaStoikovInventorySkew) {
    auto strategy = std::make_unique<AvellanedaStoikovStrategy>(
        risk_params, 0.1, 10.0, 300.0
    );
    
    // Test with positive inventory (should skew prices down)
    MarketState long_state;
    long_state.current_position = 1000;  // Long position
    long_state.unrealized_pnl = 100;
    long_state.realized_pnl = 0;
    long_state.current_volatility = 0.02;
    long_state.volume_imbalance = 0;
    long_state.order_flow_toxicity = 0.2;
    long_state.spread_percentile = 0.5;
    long_state.queue_position = 0;
    long_state.last_update = std::chrono::nanoseconds(5000);
    
    auto [long_bid, long_ask] = strategy->getQuotes(*order_book, long_state);
    
    // Test with negative inventory (should skew prices up)
    MarketState short_state = long_state;
    short_state.current_position = -1000;  // Short position
    
    auto [short_bid, short_ask] = strategy->getQuotes(*order_book, short_state);
    
    // Verify inventory skew
    ASSERT_TRUE(long_bid.has_value() && long_ask.has_value());
    ASSERT_TRUE(short_bid.has_value() && short_ask.has_value());
    
    // When long, we want to sell more aggressively (lower ask)
    EXPECT_LT(long_ask.value(), short_ask.value());
    // When short, we want to buy more aggressively (higher bid)
    EXPECT_GT(short_bid.value(), long_bid.value());
}

TEST_F(StrategyTest, AvellanedaStoikovRiskLimits) {
    auto strategy = std::make_unique<AvellanedaStoikovStrategy>(
        risk_params, 0.1, 10.0, 300.0
    );
    
    // Test with position at risk limit
    MarketState risk_state;
    risk_state.current_position = risk_params.max_position_value / 100.0;  // At limit
    risk_state.unrealized_pnl = -risk_params.max_drawdown * 0.9;  // Near drawdown limit
    risk_state.realized_pnl = 0;
    risk_state.current_volatility = 0.05;  // High volatility
    risk_state.volume_imbalance = 0;
    risk_state.order_flow_toxicity = 0.8;  // High toxicity
    risk_state.spread_percentile = 0.9;
    risk_state.queue_position = 0;
    risk_state.last_update = std::chrono::nanoseconds(5000);
    
    auto [bid_price, ask_price] = strategy->getQuotes(*order_book, risk_state);
    
    // Strategy might not quote on the long side when at position limit
    if (risk_state.current_position > 0) {
        // At long limit, might not provide ask quote
        EXPECT_TRUE(!ask_price.has_value() || 
                   ask_price.value() > order_book->getBestAsk());
    } else {
        // At short limit, might not provide bid quote
        EXPECT_TRUE(!bid_price.has_value() || 
                   bid_price.value() < order_book->getBestBid());
    }
}

// Test Alpha Market Maker Strategy
TEST_F(StrategyTest, AlphaMarketMakerBasicQuotes) {
    auto strategy = std::make_unique<AlphaMarketMaker>(risk_params);
    
    MarketState state;
    state.current_position = 0;
    state.unrealized_pnl = 0;
    state.realized_pnl = 0;
    state.current_volatility = 0.02;
    state.volume_imbalance = 0.2;  // Bullish imbalance
    state.order_flow_toxicity = 0.3;
    state.spread_percentile = 0.5;
    state.queue_position = 5;
    state.last_update = std::chrono::nanoseconds(5000);
    
    auto [bid_price, ask_price] = strategy->getQuotes(*order_book, state);
    
    ASSERT_TRUE(bid_price.has_value());
    ASSERT_TRUE(ask_price.has_value());
    
    // With bullish imbalance, alpha strategy should be more aggressive on bid
    double mid_price = order_book->getMidPrice();
    double bid_distance = mid_price - bid_price.value();
    double ask_distance = ask_price.value() - mid_price;
    
    // Bid should be closer to mid than ask (more aggressive buying)
    EXPECT_LT(bid_distance, ask_distance);
}

TEST_F(StrategyTest, AlphaMarketMakerToxicFlow) {
    auto strategy = std::make_unique<AlphaMarketMaker>(risk_params);
    
    MarketState toxic_state;
    toxic_state.current_position = 0;
    toxic_state.unrealized_pnl = 0;
    toxic_state.realized_pnl = 0;
    toxic_state.current_volatility = 0.02;
    toxic_state.volume_imbalance = 0;
    toxic_state.order_flow_toxicity = 0.9;  // Very toxic
    toxic_state.spread_percentile = 0.5;
    toxic_state.queue_position = 0;
    toxic_state.last_update = std::chrono::nanoseconds(5000);
    
    auto [bid_price, ask_price] = strategy->getQuotes(*order_book, toxic_state);
    
    // With toxic flow, strategy should either not quote or quote wide
    if (bid_price.has_value() && ask_price.has_value()) {
        double spread = ask_price.value() - bid_price.value();
        double normal_spread = order_book->getSpread();
        
        // Spread should be wider than normal market spread
        EXPECT_GT(spread, normal_spread * 1.5);
    }
}

// Test Strategy Position Management
TEST_F(StrategyTest, PositionSizingWithRisk) {
    auto strategy = std::make_unique<AvellanedaStoikovStrategy>(
        risk_params, 0.1, 10.0, 300.0
    );
    
    // Test position sizing calculation
    double signal_strength = 0.8;
    double current_exposure = 50000.0;
    
    double position_size = strategy->calculatePositionSize(
        signal_strength, current_exposure
    );
    
    // Verify position size respects limits
    EXPECT_GT(position_size, 0);
    EXPECT_LE(position_size * 100.0, risk_params.max_position_value);
    
    // Test with near-limit exposure
    double near_limit_exposure = risk_params.max_position_value * 0.95;
    double limited_size = strategy->calculatePositionSize(
        signal_strength, near_limit_exposure
    );
    
    // Should return smaller size when near limit
    EXPECT_LT(limited_size, position_size);
}

// Performance benchmarks
TEST_F(StrategyTest, StrategyPerformance) {
    auto strategy = std::make_unique<AvellanedaStoikovStrategy>(
        risk_params, 0.1, 10.0, 300.0
    );
    
    MarketState state;
    state.current_position = 0;
    state.unrealized_pnl = 0;
    state.realized_pnl = 0;
    state.current_volatility = 0.02;
    state.volume_imbalance = 0.1;
    state.order_flow_toxicity = 0.3;
    state.spread_percentile = 0.5;
    state.queue_position = 0;
    state.last_update = std::chrono::nanoseconds(5000);
    
    // Measure quote generation time
    const int iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto [bid, ask] = strategy->getQuotes(*order_book, state);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / static_cast<double>(iterations);
    
    // Quote generation should be fast (< 100 microseconds)
    EXPECT_LT(avg_time, 100.0);
    
    std::cout << "Average quote generation time: " << avg_time << " μs" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}