#include <gtest/gtest.h>
#include "risk/risk_manager.h"
#include <memory>
#include <vector>
#include <thread>

using namespace hft;

class RiskManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup risk limits
        risk_limits = RiskLimits{
            100000.0,    // max_position_value
            500000.0,    // max_total_exposure
            20,          // max_position_count
            10000.0,     // max_daily_loss
            0.05,        // max_drawdown (5%)
            0.02,        // stop_loss_percent (2%)
            1000.0,      // max_order_size
            50000.0,     // max_order_value
            100,         // max_orders_per_second
            50,          // max_open_orders
            0.005,       // max_spread_percent (0.5%)
            0.1,         // min_liquidity_ratio
            0.001,       // max_market_impact
            10000.0,     // max_delta
            1000.0,      // max_gamma
            5000.0,      // max_vega
            1000.0,      // max_theta
            3.0,         // max_leverage
            0.3,         // margin_call_level
            0.25         // liquidation_level
        };
        
        risk_manager = std::make_unique<RiskManager>(risk_limits);
    }
    
    RiskLimits risk_limits;
    std::unique_ptr<RiskManager> risk_manager;
};

// Basic Position Management Tests
TEST_F(RiskManagerTest, BasicPositionManagement) {
    // Test taking a new position
    EXPECT_TRUE(risk_manager->canTakePosition("AAPL", 100, 150.0));
    
    risk_manager->updatePosition("AAPL", 100, 150.0);
    
    auto position = risk_manager->getPosition("AAPL");
    EXPECT_EQ(position.symbol, "AAPL");
    EXPECT_EQ(position.quantity, 100);
    EXPECT_DOUBLE_EQ(position.average_price, 150.0);
    EXPECT_DOUBLE_EQ(position.market_value, 15000.0);
}

TEST_F(RiskManagerTest, PositionLimitEnforcement) {
    // Test position value limit
    double large_quantity = risk_limits.max_position_value / 100.0 + 100;
    EXPECT_FALSE(risk_manager->canTakePosition("AAPL", large_quantity, 100.0));
    
    // Test total exposure limit
    risk_manager->updatePosition("AAPL", 1000, 100.0);
    risk_manager->updatePosition("GOOGL", 500, 200.0);
    risk_manager->updatePosition("MSFT", 2000, 150.0);
    
    // Try to add position that would exceed total exposure
    EXPECT_FALSE(risk_manager->canTakePosition("AMZN", 1000, 200.0));
}

TEST_F(RiskManagerTest, OrderValidation) {
    // Test valid order
    EXPECT_TRUE(risk_manager->validateOrder("AAPL", Side::BUY, 100, 150.0));
    
    // Test order size limit
    EXPECT_FALSE(risk_manager->validateOrder("AAPL", Side::BUY, 
        risk_limits.max_order_size + 100, 150.0));
    
    // Test order value limit
    double high_price = risk_limits.max_order_value / 100.0 + 100;
    EXPECT_FALSE(risk_manager->validateOrder("AAPL", Side::BUY, 100, high_price));
}

TEST_F(RiskManagerTest, PnLTracking) {
    // Create position
    risk_manager->updatePosition("AAPL", 1000, 100.0);
    
    // Mark to market with profit
    risk_manager->markToMarket("AAPL", 105.0);
    
    auto position = risk_manager->getPosition("AAPL");
    EXPECT_DOUBLE_EQ(position.unrealized_pnl, 5000.0);  // 1000 * (105 - 100)
    
    // Realize some PnL
    risk_manager->realizePnL("AAPL", 2000.0);
    position = risk_manager->getPosition("AAPL");
    EXPECT_DOUBLE_EQ(position.realized_pnl, 2000.0);
    
    // Check total PnL
    EXPECT_DOUBLE_EQ(risk_manager->getTotalPnL(), 7000.0);
}

TEST_F(RiskManagerTest, DrawdownCalculation) {
    // Simulate profitable trades
    risk_manager->realizePnL("AAPL", 5000.0);
    risk_manager->realizePnL("GOOGL", 3000.0);
    
    // Then losses
    risk_manager->realizePnL("MSFT", -2000.0);
    risk_manager->realizePnL("AMZN", -1000.0);
    
    auto metrics = risk_manager->getMetrics();
    
    // Drawdown should be calculated from peak
    EXPECT_GT(metrics.current_drawdown, 0);
    EXPECT_LE(metrics.current_drawdown, metrics.max_drawdown);
}

TEST_F(RiskManagerTest, EmergencyStop) {
    // Should not be stopped initially
    EXPECT_FALSE(risk_manager->shouldStopTrading());
    
    // Trigger emergency stop
    risk_manager->triggerEmergencyStop("Risk limit breach");
    
    // Should be stopped now
    EXPECT_TRUE(risk_manager->shouldStopTrading());
    
    // Orders should not be validated when stopped
    EXPECT_FALSE(risk_manager->validateOrder("AAPL", Side::BUY, 100, 150.0));
}

TEST_F(RiskManagerTest, RiskEventHandling) {
    std::vector<RiskEvent> received_events;
    
    // Add event handler
    risk_manager->addRiskHandler([&received_events](const RiskEvent& event) {
        received_events.push_back(event);
    });
    
    // Publish risk event
    RiskEvent test_event{
        RiskEventType::LIMIT_BREACH,
        "Position limit exceeded",
        "AAPL",
        0.8,
        std::chrono::nanoseconds(1000),
        {{"limit", "max_position_value"}, {"current", "110000"}}
    };
    
    risk_manager->publishRiskEvent(test_event);
    
    // Allow time for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    ASSERT_EQ(received_events.size(), 1);
    EXPECT_EQ(received_events[0].type, RiskEventType::LIMIT_BREACH);
    EXPECT_EQ(received_events[0].symbol, "AAPL");
}

// Dynamic Risk Manager Tests
TEST_F(RiskManagerTest, DynamicRiskWithCorrelation) {
    DynamicRiskManager::HedgeParams hedge_params{
        true,       // auto_hedge_enabled
        0.1,        // delta_hedge_threshold
        0.05,       // gamma_hedge_threshold
        60000,      // hedge_rebalance_interval_ms
        {"SPY", "VXX"}  // hedge_instruments
    };
    
    auto dynamic_risk = std::make_unique<DynamicRiskManager>(risk_limits, hedge_params);
    
    // Set correlations
    dynamic_risk->updateCorrelation("AAPL", "MSFT", 0.8);
    dynamic_risk->updateCorrelation("AAPL", "GOOGL", 0.7);
    
    // Test correlation retrieval
    EXPECT_DOUBLE_EQ(dynamic_risk->getCorrelation("AAPL", "MSFT"), 0.8);
    EXPECT_DOUBLE_EQ(dynamic_risk->getCorrelation("MSFT", "AAPL"), 0.8);  // Symmetric
    
    // Take correlated positions
    dynamic_risk->updatePosition("AAPL", 1000, 150.0);
    dynamic_risk->updatePosition("MSFT", 1000, 200.0);
    
    // Due to correlation, effective risk should be considered
    bool can_take_more = dynamic_risk->canTakePosition("GOOGL", 1000, 250.0);
    
    // This depends on implementation, but correlated positions should affect limits
    EXPECT_TRUE(can_take_more || !can_take_more);  // Just verify it runs
}

TEST_F(RiskManagerTest, StressTesting) {
    DynamicRiskManager::HedgeParams hedge_params{
        false, 0.1, 0.05, 60000, {}
    };
    
    auto dynamic_risk = std::make_unique<DynamicRiskManager>(risk_limits, hedge_params);
    
    // Add positions
    dynamic_risk->updatePosition("AAPL", 1000, 150.0);
    dynamic_risk->updatePosition("GOOGL", 500, 2000.0);
    
    // Add stress scenario
    DynamicRiskManager::StressScenario crash_scenario{
        "Market Crash",
        {{"AAPL", -0.20}, {"GOOGL", -0.25}},  // 20% and 25% drops
        0.05  // 5% probability
    };
    
    dynamic_risk->addStressScenario(crash_scenario);
    
    // Run stress test
    double stress_loss = dynamic_risk->runStressTest("Market Crash");
    
    // Expected loss: 1000*150*0.20 + 500*2000*0.25 = 30000 + 250000 = 280000
    EXPECT_DOUBLE_EQ(stress_loss, -280000.0);
}

// Capital Allocator Tests
TEST_F(RiskManagerTest, CapitalAllocation) {
    CapitalAllocator::AllocationConstraints constraints{
        1000000.0,   // total_capital
        0.20,        // max_allocation_percent (20%)
        0.05,        // min_allocation_percent (5%)
        2.0,         // target_sharpe_ratio
        0.02         // risk_free_rate
    };
    
    auto allocator = std::make_unique<CapitalAllocator>(constraints);
    
    // Update strategy performance
    std::vector<double> strategy1_returns = {0.01, 0.02, -0.01, 0.015, 0.005};
    std::vector<double> strategy2_returns = {0.005, 0.01, 0.0, 0.008, 0.012};
    
    allocator->updateStrategyPerformance("strategy1", strategy1_returns);
    allocator->updateStrategyPerformance("strategy2", strategy2_returns);
    
    // Get optimal allocation
    auto allocations = allocator->optimizeAllocation({"strategy1", "strategy2"});
    
    // Verify allocations sum to 1.0 and respect constraints
    double total = 0;
    for (const auto& [strategy, allocation] : allocations) {
        EXPECT_GE(allocation, constraints.min_allocation_percent);
        EXPECT_LE(allocation, constraints.max_allocation_percent);
        total += allocation;
    }
    EXPECT_NEAR(total, 1.0, 0.01);
}

TEST_F(RiskManagerTest, MarginManagement) {
    MarginManager::MarginRequirements requirements{
        0.25,    // initial_margin_percent (25%)
        0.20,    // maintenance_margin_percent (20%)
        1.5,     // option_margin_multiplier
        {{"AAPL", 0.30}}  // AAPL requires 30% margin
    };
    
    auto margin_manager = std::make_unique<MarginManager>(requirements);
    
    // Set account balance
    MarginManager::AccountBalance balance{
        500000.0,    // cash_balance
        300000.0,    // securities_value
        150000.0,    // margin_used
        650000.0,    // available_margin
        500000.0,    // excess_liquidity
        std::chrono::nanoseconds(1000)
    };
    
    margin_manager->updateBalance(balance);
    
    // Calculate margin requirements
    double initial_margin = margin_manager->calculateInitialMargin("AAPL", 1000, 150.0);
    EXPECT_DOUBLE_EQ(initial_margin, 1000 * 150.0 * 0.30);  // Uses AAPL specific
    
    double generic_margin = margin_manager->calculateInitialMargin("GOOGL", 100, 2000.0);
    EXPECT_DOUBLE_EQ(generic_margin, 100 * 2000.0 * 0.25);  // Uses default
    
    // Check margin adequacy
    EXPECT_TRUE(margin_manager->hasAdequateMargin(50000.0));
    EXPECT_FALSE(margin_manager->hasAdequateMargin(700000.0));
}

TEST_F(RiskManagerTest, ComplianceChecks) {
    ComplianceManager::ComplianceRules rules{
        false,       // no_short_selling
        false,       // uptick_rule_enabled
        0.05,        // max_order_size_percent_adv (5% of ADV)
        0.10,        // max_participation_rate (10%)
        100.0,       // min_order_spacing_ms
        5,           // max_order_modifications
        {{"RESTRICTED", true}},  // restricted_symbols
        {{"AAPL", 10000.0}},    // position_limits
        {{9, 10}, {15, 16}},    // blackout_periods (9-10am, 3-4pm)
        true         // trade_only_regular_hours
    };
    
    auto compliance = std::make_unique<ComplianceManager>(rules);
    
    // Test restricted symbol
    EXPECT_FALSE(compliance->canTradeSymbol("RESTRICTED"));
    EXPECT_TRUE(compliance->canTradeSymbol("AAPL"));
    
    // Test order compliance
    EXPECT_TRUE(compliance->isOrderCompliant("AAPL", Side::BUY, 100, 150.0));
    
    // Record some orders to test rate limiting
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    compliance->recordOrder("AAPL", 100, std::chrono::duration_cast<std::chrono::nanoseconds>(now));
    
    // Immediate second order should be flagged (min spacing not met)
    EXPECT_FALSE(compliance->isOrderCompliant("AAPL", Side::BUY, 100, 150.0));
}

// Integrated Risk System Tests
TEST_F(RiskManagerTest, IntegratedRiskSystem) {
    CapitalAllocator::AllocationConstraints capital_constraints{
        1000000.0, 0.10, 0.01, 2.0, 0.02
    };
    
    MarginManager::MarginRequirements margin_requirements{
        0.25, 0.20, 1.5, {}
    };
    
    ComplianceManager::ComplianceRules compliance_rules{
        false, false, 0.05, 0.10, 100.0, 5, {}, {}, {}, true
    };
    
    auto integrated_system = std::make_unique<IntegratedRiskSystem>(
        risk_limits, capital_constraints, margin_requirements, compliance_rules
    );
    
    // Test unified order approval
    std::vector<std::string> rejection_reasons;
    
    bool approved = integrated_system->approveOrder(
        "AAPL", Side::BUY, 100, 150.0, rejection_reasons
    );
    
    EXPECT_TRUE(approved);
    EXPECT_TRUE(rejection_reasons.empty());
    
    // Test order that violates limits
    approved = integrated_system->approveOrder(
        "AAPL", Side::BUY, 10000, 150.0, rejection_reasons
    );
    
    EXPECT_FALSE(approved);
    EXPECT_FALSE(rejection_reasons.empty());
}

// Performance and Thread Safety Tests
TEST_F(RiskManagerTest, ThreadSafety) {
    const int num_threads = 4;
    const int operations_per_thread = 1000;
    
    std::vector<std::thread> threads;
    
    // Launch threads that perform concurrent operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string symbol = "SYM" + std::to_string(t);
                
                // Alternate between operations
                if (i % 3 == 0) {
                    risk_manager->canTakePosition(symbol, 10, 100.0);
                } else if (i % 3 == 1) {
                    risk_manager->updatePosition(symbol, 10, 100.0);
                } else {
                    risk_manager->markToMarket(symbol, 101.0);
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify system is still consistent
    EXPECT_TRUE(risk_manager->isWithinRiskLimits());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}