#include "core/order_book.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace hft {

OrderBook::OrderBook() : last_update_sequence_(0) {
    metrics_ = {0.0, 0.0, 0.0, 0.0, Timestamp(0)};
}

void OrderBook::addOrder(std::shared_ptr<Order> order) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // O(1) insertion into order map
    order_map_[order->id] = order;
    
    // O(log n) insertion into price level
    if (order->side == Side::BUY) {
        auto& level = bids_[order->price];
        level.price = order->price;
        level.orders.push_back(order);
        level.total_quantity += order->quantity;
    } else {
        auto& level = asks_[order->price];
        level.price = order->price;
        level.orders.push_back(order);
        level.total_quantity += order->quantity;
    }
    
    last_update_sequence_++;
    updateMetrics();
}

void OrderBook::cancelOrder(OrderId order_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = order_map_.find(order_id);
    if (it == order_map_.end()) {
        return;
    }
    
    auto order = it->second;
    order_map_.erase(it);
    
    // Remove from price level
    auto& book = (order->side == Side::BUY) ? 
        static_cast<std::map<Price, Level>&>(bids_) : 
        static_cast<std::map<Price, Level>&>(asks_);
    
    auto level_it = book.find(order->price);
    if (level_it != book.end()) {
        auto& level = level_it->second;
        auto order_it = std::find(level.orders.begin(), level.orders.end(), order);
        if (order_it != level.orders.end()) {
            level.orders.erase(order_it);
            level.total_quantity -= order->quantity;
            
            if (level.orders.empty()) {
                book.erase(level_it);
            }
        }
    }
    
    last_update_sequence_++;
    updateMetrics();
}

void OrderBook::modifyOrder(OrderId order_id, Quantity new_quantity) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = order_map_.find(order_id);
    if (it == order_map_.end()) {
        return;
    }
    
    auto order = it->second;
    Quantity old_quantity = order->quantity;
    order->quantity = new_quantity;
    
    // Update level quantity
    auto& book = (order->side == Side::BUY) ? 
        static_cast<std::map<Price, Level>&>(bids_) : 
        static_cast<std::map<Price, Level>&>(asks_);
    
    auto level_it = book.find(order->price);
    if (level_it != book.end()) {
        level_it->second.total_quantity += (new_quantity - old_quantity);
    }
    
    last_update_sequence_++;
    updateMetrics();
}

std::shared_ptr<Order> OrderBook::getOrder(OrderId order_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = order_map_.find(order_id);
    return (it != order_map_.end()) ? it->second : nullptr;
}

double OrderBook::getBestBid() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bids_.empty() ? 0.0 : bids_.begin()->first;
}

double OrderBook::getBestAsk() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asks_.empty() ? 0.0 : asks_.begin()->first;
}

double OrderBook::getSpread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return asks_.begin()->first - bids_.begin()->first;
}

double OrderBook::getMidPrice() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bids_.empty() || asks_.empty()) {
        return 0.0;
    }
    return (asks_.begin()->first + bids_.begin()->first) / 2.0;
}

double OrderBook::calculateMicroprice(int levels) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double weighted_bid_price = 0.0;
    double weighted_ask_price = 0.0;
    double total_bid_qty = 0.0;
    double total_ask_qty = 0.0;
    
    // Calculate weighted prices for bids
    int count = 0;
    for (const auto& [price, level] : bids_) {
        if (count++ >= levels) break;
        weighted_bid_price += price * level.total_quantity;
        total_bid_qty += level.total_quantity;
    }
    
    // Calculate weighted prices for asks
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count++ >= levels) break;
        weighted_ask_price += price * level.total_quantity;
        total_ask_qty += level.total_quantity;
    }
    
    if (total_bid_qty == 0 || total_ask_qty == 0) {
        return getMidPrice();
    }
    
    // Microprice formula: (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
    double avg_bid_price = weighted_bid_price / total_bid_qty;
    double avg_ask_price = weighted_ask_price / total_ask_qty;
    
    return (avg_bid_price * total_ask_qty + avg_ask_price * total_bid_qty) / 
           (total_bid_qty + total_ask_qty);
}

double OrderBook::getOrderFlowImbalance(int time_window_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Calculate volume imbalance at top levels
    double bid_volume = 0.0;
    double ask_volume = 0.0;
    
    int levels = 5;
    int count = 0;
    
    for (const auto& [price, level] : bids_) {
        if (count++ >= levels) break;
        bid_volume += level.total_quantity;
    }
    
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count++ >= levels) break;
        ask_volume += level.total_quantity;
    }
    
    double total_volume = bid_volume + ask_volume;
    if (total_volume == 0) return 0.0;
    
    // Order flow imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
    return (bid_volume - ask_volume) / total_volume;
}

double OrderBook::getBookPressure(int levels) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double bid_pressure = 0.0;
    double ask_pressure = 0.0;
    
    // Calculate pressure as sum of (quantity / distance_from_mid)
    double mid_price = getMidPrice();
    if (mid_price == 0) return 0.0;
    
    int count = 0;
    for (const auto& [price, level] : bids_) {
        if (count++ >= levels) break;
        double distance = std::abs(mid_price - price);
        if (distance > 0) {
            bid_pressure += level.total_quantity / distance;
        }
    }
    
    count = 0;
    for (const auto& [price, level] : asks_) {
        if (count++ >= levels) break;
        double distance = std::abs(price - mid_price);
        if (distance > 0) {
            ask_pressure += level.total_quantity / distance;
        }
    }
    
    return bid_pressure - ask_pressure;
}

double OrderBook::getKyleLambda() const {
    // Simplified Kyle's lambda calculation
    // In practice, this would require trade data and price impact analysis
    double spread = getSpread();
    double depth = 0.0;
    
    // Average depth at top 5 levels
    auto bid_levels = getBidLevels(5);
    auto ask_levels = getAskLevels(5);
    
    for (const auto& [price, qty] : bid_levels) {
        depth += qty;
    }
    for (const auto& [price, qty] : ask_levels) {
        depth += qty;
    }
    
    if (depth == 0) return 0.0;
    
    // Simplified lambda = spread / sqrt(depth)
    return spread / std::sqrt(depth);
}

std::vector<std::pair<Price, Quantity>> OrderBook::getBidLevels(int depth) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<Price, Quantity>> levels;
    
    int count = 0;
    for (const auto& [price, level] : bids_) {
        if (count++ >= depth) break;
        levels.emplace_back(price, level.total_quantity);
    }
    
    return levels;
}

std::vector<std::pair<Price, Quantity>> OrderBook::getAskLevels(int depth) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<Price, Quantity>> levels;
    
    int count = 0;
    for (const auto& [price, level] : asks_) {
        if (count++ >= depth) break;
        levels.emplace_back(price, level.total_quantity);
    }
    
    return levels;
}

OrderBook::Snapshot OrderBook::getSnapshot(int depth) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Snapshot snapshot;
    snapshot.bids = getBidLevels(depth);
    snapshot.asks = getAskLevels(depth);
    snapshot.timestamp = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    snapshot.sequence = last_update_sequence_.load();
    
    return snapshot;
}

void OrderBook::restoreFromSnapshot(const Snapshot& snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear existing data
    bids_.clear();
    asks_.clear();
    order_map_.clear();
    
    // Restore bid levels
    OrderId synthetic_id = 1;
    for (const auto& [price, quantity] : snapshot.bids) {
        auto order = std::make_shared<Order>(
            synthetic_id++, Side::BUY, price, quantity, snapshot.timestamp
        );
        order_map_[order->id] = order;
        
        auto& level = bids_[price];
        level.price = price;
        level.orders.push_back(order);
        level.total_quantity = quantity;
    }
    
    // Restore ask levels
    for (const auto& [price, quantity] : snapshot.asks) {
        auto order = std::make_shared<Order>(
            synthetic_id++, Side::SELL, price, quantity, snapshot.timestamp
        );
        order_map_[order->id] = order;
        
        auto& level = asks_[price];
        level.price = price;
        level.orders.push_back(order);
        level.total_quantity = quantity;
    }
    
    last_update_sequence_ = snapshot.sequence;
    updateMetrics();
}

void OrderBook::updateMetrics() {
    metrics_.bid_ask_spread = getSpread();
    metrics_.weighted_mid_price = calculateMicroprice();
    metrics_.order_flow_imbalance = getOrderFlowImbalance();
    metrics_.book_pressure = getBookPressure();
    metrics_.last_update = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
}

void OrderBook::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing");
    }
    
    auto snapshot = getSnapshot(100);  // Save top 100 levels
    
    // Write header
    uint32_t bid_count = snapshot.bids.size();
    uint32_t ask_count = snapshot.asks.size();
    file.write(reinterpret_cast<const char*>(&bid_count), sizeof(bid_count));
    file.write(reinterpret_cast<const char*>(&ask_count), sizeof(ask_count));
    
    // Write bid levels
    for (const auto& [price, quantity] : snapshot.bids) {
        file.write(reinterpret_cast<const char*>(&price), sizeof(price));
        file.write(reinterpret_cast<const char*>(&quantity), sizeof(quantity));
    }
    
    // Write ask levels
    for (const auto& [price, quantity] : snapshot.asks) {
        file.write(reinterpret_cast<const char*>(&price), sizeof(price));
        file.write(reinterpret_cast<const char*>(&quantity), sizeof(quantity));
    }
}

void OrderBook::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading");
    }
    
    // Read header
    uint32_t bid_count, ask_count;
    file.read(reinterpret_cast<char*>(&bid_count), sizeof(bid_count));
    file.read(reinterpret_cast<char*>(&ask_count), sizeof(ask_count));
    
    Snapshot snapshot;
    
    // Read bid levels
    for (uint32_t i = 0; i < bid_count; ++i) {
        Price price;
        Quantity quantity;
        file.read(reinterpret_cast<char*>(&price), sizeof(price));
        file.read(reinterpret_cast<char*>(&quantity), sizeof(quantity));
        snapshot.bids.emplace_back(price, quantity);
    }
    
    // Read ask levels
    for (uint32_t i = 0; i < ask_count; ++i) {
        Price price;
        Quantity quantity;
        file.read(reinterpret_cast<char*>(&price), sizeof(price));
        file.read(reinterpret_cast<char*>(&quantity), sizeof(quantity));
        snapshot.asks.emplace_back(price, quantity);
    }
    
    restoreFromSnapshot(snapshot);
}

} // namespace hft