#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>

namespace hft {

using OrderId = uint64_t;
using Price = double;
using Quantity = uint64_t;
using Timestamp = std::chrono::nanoseconds;

enum class Side {
    BUY,
    SELL
};

enum class OrderType {
    LIMIT,
    MARKET,
    STOP
};

struct Order {
    OrderId id;
    Side side;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    Timestamp timestamp;
    OrderType type;
    
    Order(OrderId id, Side side, Price price, Quantity qty, Timestamp ts, OrderType type = OrderType::LIMIT)
        : id(id), side(side), price(price), quantity(qty), filled_quantity(0), timestamp(ts), type(type) {}
};

struct Level {
    Price price;
    Quantity total_quantity;
    std::vector<std::shared_ptr<Order>> orders;
    
    Level(Price p) : price(p), total_quantity(0) {}
};

class OrderBook {
private:
    std::map<Price, Level, std::greater<Price>> bids_;  // Descending order
    std::map<Price, Level, std::less<Price>> asks_;     // Ascending order
    std::unordered_map<OrderId, std::shared_ptr<Order>> order_map_;
    
    mutable std::mutex mutex_;
    std::atomic<uint64_t> last_update_sequence_;
    
    // Microstructure metrics
    struct MarketMetrics {
        double bid_ask_spread;
        double weighted_mid_price;
        double order_flow_imbalance;
        double book_pressure;
        Timestamp last_update;
    } metrics_;
    
    void updateMetrics();
    
public:
    OrderBook();
    ~OrderBook() = default;
    
    // Core operations - O(1) for order lookup, O(log n) for price level operations
    void addOrder(std::shared_ptr<Order> order);
    void cancelOrder(OrderId order_id);
    void modifyOrder(OrderId order_id, Quantity new_quantity);
    std::shared_ptr<Order> getOrder(OrderId order_id) const;
    
    // Market data access
    double getBestBid() const;
    double getBestAsk() const;
    double getSpread() const;
    double getMidPrice() const;
    
    // Advanced metrics
    double calculateMicroprice(int levels = 5) const;
    double getOrderFlowImbalance(int time_window_ms = 1000) const;
    double getBookPressure(int levels = 10) const;
    double getKyleLambda() const;
    
    // Level 2 data
    std::vector<std::pair<Price, Quantity>> getBidLevels(int depth = 10) const;
    std::vector<std::pair<Price, Quantity>> getAskLevels(int depth = 10) const;
    
    // Snapshot for backtesting
    struct Snapshot {
        std::vector<std::pair<Price, Quantity>> bids;
        std::vector<std::pair<Price, Quantity>> asks;
        Timestamp timestamp;
        uint64_t sequence;
    };
    
    Snapshot getSnapshot(int depth = 20) const;
    void restoreFromSnapshot(const Snapshot& snapshot);
    
    // Performance metrics
    uint64_t getUpdateCount() const { return last_update_sequence_.load(); }
    size_t getOrderCount() const { return order_map_.size(); }
    
    // Memory-mapped persistence
    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
};

// Order book event for event sourcing
struct OrderBookEvent {
    enum Type {
        ADD,
        CANCEL,
        MODIFY,
        TRADE
    };
    
    Type type;
    OrderId order_id;
    Side side;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
};

} // namespace hft