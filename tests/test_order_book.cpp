#include "core/order_book.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <random>

using namespace hft;

void test_basic_operations() {
    std::cout << "Testing basic order book operations..." << std::endl;
    
    OrderBook book;
    
    // Test adding orders
    auto order1 = std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0));
    auto order2 = std::make_shared<Order>(2, Side::SELL, 101.0, 1500, Timestamp(0));
    auto order3 = std::make_shared<Order>(3, Side::BUY, 99.5, 500, Timestamp(0));
    
    book.addOrder(order1);
    book.addOrder(order2);
    book.addOrder(order3);
    
    // Test best bid/ask
    assert(book.getBestBid() == 100.0);
    assert(book.getBestAsk() == 101.0);
    assert(book.getSpread() == 1.0);
    assert(book.getMidPrice() == 100.5);
    
    // Test order retrieval
    auto retrieved = book.getOrder(1);
    assert(retrieved != nullptr);
    assert(retrieved->price == 100.0);
    
    // Test order cancellation
    book.cancelOrder(1);
    assert(book.getBestBid() == 99.5);
    assert(book.getOrder(1) == nullptr);
    
    // Test order modification
    book.modifyOrder(3, 750);
    auto modified = book.getOrder(3);
    assert(modified->quantity == 750);
    
    std::cout << "✓ Basic operations test passed" << std::endl;
}

void test_microprice_calculation() {
    std::cout << "Testing microprice calculation..." << std::endl;
    
    OrderBook book;
    
    // Add multiple levels
    book.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(2, Side::BUY, 99.9, 2000, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(3, Side::BUY, 99.8, 3000, Timestamp(0)));
    
    book.addOrder(std::make_shared<Order>(4, Side::SELL, 100.1, 1500, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(5, Side::SELL, 100.2, 2500, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(6, Side::SELL, 100.3, 3500, Timestamp(0)));
    
    double microprice = book.calculateMicroprice(3);
    double mid_price = book.getMidPrice();
    
    // Microprice should be different from mid price due to size imbalance
    assert(microprice != mid_price);
    
    // With more size on the ask side, microprice should be below mid
    assert(microprice < mid_price);
    
    std::cout << "✓ Microprice calculation test passed" << std::endl;
    std::cout << "  Mid price: " << mid_price << ", Microprice: " << microprice << std::endl;
}

void test_order_flow_imbalance() {
    std::cout << "Testing order flow imbalance..." << std::endl;
    
    OrderBook book;
    
    // Create imbalanced book
    book.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 5000, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(2, Side::SELL, 100.1, 1000, Timestamp(0)));
    
    double imbalance = book.getOrderFlowImbalance();
    
    // Should show positive imbalance (more buy volume)
    assert(imbalance > 0);
    assert(imbalance < 1.0);  // Should be normalized between -1 and 1
    
    std::cout << "✓ Order flow imbalance test passed" << std::endl;
    std::cout << "  Imbalance: " << imbalance << std::endl;
}

void test_snapshot_restore() {
    std::cout << "Testing snapshot and restore..." << std::endl;
    
    OrderBook book1;
    
    // Build a book
    book1.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0)));
    book1.addOrder(std::make_shared<Order>(2, Side::BUY, 99.9, 2000, Timestamp(0)));
    book1.addOrder(std::make_shared<Order>(3, Side::SELL, 100.1, 1500, Timestamp(0)));
    book1.addOrder(std::make_shared<Order>(4, Side::SELL, 100.2, 2500, Timestamp(0)));
    
    // Take snapshot
    auto snapshot = book1.getSnapshot(10);
    
    // Create new book from snapshot
    OrderBook book2;
    book2.restoreFromSnapshot(snapshot);
    
    // Verify books are equivalent
    assert(book2.getBestBid() == book1.getBestBid());
    assert(book2.getBestAsk() == book1.getBestAsk());
    assert(book2.getMidPrice() == book1.getMidPrice());
    
    std::cout << "✓ Snapshot/restore test passed" << std::endl;
}

void test_performance() {
    std::cout << "Testing order book performance..." << std::endl;
    
    OrderBook book;
    const int num_orders = 100000;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> price_dist(99.0, 101.0);
    std::uniform_int_distribution<> size_dist(100, 10000);
    std::uniform_int_distribution<> side_dist(0, 1);
    
    // Measure add performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_orders; ++i) {
        auto order = std::make_shared<Order>(
            i,
            side_dist(gen) == 0 ? Side::BUY : Side::SELL,
            price_dist(gen),
            size_dist(gen),
            Timestamp(0)
        );
        book.addOrder(order);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double adds_per_second = (num_orders * 1000000.0) / add_duration.count();
    std::cout << "  Add performance: " << adds_per_second << " orders/second" << std::endl;
    
    // Measure cancel performance
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_orders / 2; ++i) {
        book.cancelOrder(i * 2);  // Cancel every other order
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto cancel_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double cancels_per_second = ((num_orders / 2) * 1000000.0) / cancel_duration.count();
    std::cout << "  Cancel performance: " << cancels_per_second << " orders/second" << std::endl;
    
    // Measure metric calculation performance
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10000; ++i) {
        double microprice = book.calculateMicroprice();
        double imbalance = book.getOrderFlowImbalance();
        double pressure = book.getBookPressure();
        (void)microprice; (void)imbalance; (void)pressure;  // Avoid unused warnings
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto metric_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double metrics_per_second = (10000 * 1000000.0) / metric_duration.count();
    std::cout << "  Metric calculations: " << metrics_per_second << " calculations/second" << std::endl;
    
    std::cout << "✓ Performance test passed" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    OrderBook book;
    
    // Empty book
    assert(book.getBestBid() == 0.0);
    assert(book.getBestAsk() == 0.0);
    assert(book.getSpread() == 0.0);
    assert(book.getMidPrice() == 0.0);
    
    // Single order
    book.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0)));
    assert(book.getBestBid() == 100.0);
    assert(book.getBestAsk() == 0.0);
    assert(book.getMidPrice() == 0.0);  // No ask, so no mid
    
    // Cancel non-existent order
    book.cancelOrder(999);  // Should not crash
    
    // Modify non-existent order
    book.modifyOrder(999, 100);  // Should not crash
    
    // Zero quantity order (should still work)
    book.addOrder(std::make_shared<Order>(2, Side::SELL, 101.0, 0, Timestamp(0)));
    
    std::cout << "✓ Edge cases test passed" << std::endl;
}

void test_level_aggregation() {
    std::cout << "Testing price level aggregation..." << std::endl;
    
    OrderBook book;
    
    // Add multiple orders at same price
    book.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(2, Side::BUY, 100.0, 2000, Timestamp(0)));
    book.addOrder(std::make_shared<Order>(3, Side::BUY, 100.0, 3000, Timestamp(0)));
    
    auto bid_levels = book.getBidLevels(1);
    assert(bid_levels.size() == 1);
    assert(bid_levels[0].first == 100.0);
    assert(bid_levels[0].second == 6000);  // Total quantity at level
    
    // Cancel one order
    book.cancelOrder(2);
    bid_levels = book.getBidLevels(1);
    assert(bid_levels[0].second == 4000);  // Updated total
    
    std::cout << "✓ Level aggregation test passed" << std::endl;
}

void test_persistence() {
    std::cout << "Testing file persistence..." << std::endl;
    
    const std::string filename = "test_orderbook.dat";
    
    {
        OrderBook book1;
        
        // Create a book with some orders
        book1.addOrder(std::make_shared<Order>(1, Side::BUY, 100.0, 1000, Timestamp(0)));
        book1.addOrder(std::make_shared<Order>(2, Side::BUY, 99.9, 2000, Timestamp(0)));
        book1.addOrder(std::make_shared<Order>(3, Side::SELL, 100.1, 1500, Timestamp(0)));
        book1.addOrder(std::make_shared<Order>(4, Side::SELL, 100.2, 2500, Timestamp(0)));
        
        // Save to file
        book1.saveToFile(filename);
    }
    
    {
        // Load from file in new scope
        OrderBook book2;
        book2.loadFromFile(filename);
        
        // Verify data
        assert(book2.getBestBid() == 100.0);
        assert(book2.getBestAsk() == 100.1);
        
        auto bid_levels = book2.getBidLevels(2);
        assert(bid_levels.size() == 2);
        assert(bid_levels[0].first == 100.0);
        assert(bid_levels[1].first == 99.9);
    }
    
    // Cleanup
    std::remove(filename.c_str());
    
    std::cout << "✓ Persistence test passed" << std::endl;
}

int main() {
    std::cout << "Running Order Book Tests\n" << std::endl;
    
    try {
        test_basic_operations();
        test_microprice_calculation();
        test_order_flow_imbalance();
        test_snapshot_restore();
        test_level_aggregation();
        test_edge_cases();
        test_persistence();
        test_performance();
        
        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}