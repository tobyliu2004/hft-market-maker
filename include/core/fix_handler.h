#pragma once

#include <quickfix/Application.h>
#include <quickfix/MessageCracker.h>
#include <quickfix/SessionID.h>
#include <quickfix/Message.h>
#include <quickfix/fix44/MarketDataSnapshotFullRefresh.h>
#include <quickfix/fix44/MarketDataIncrementalRefresh.h>
#include <quickfix/fix44/MarketDataRequest.h>
#include <quickfix/fix44/NewOrderSingle.h>
#include <quickfix/fix44/ExecutionReport.h>
#include <quickfix/fix44/OrderCancelRequest.h>
#include <quickfix/fix44/OrderCancelReplaceRequest.h>

#include "core/order_book.h"
#include <memory>
#include <unordered_map>
#include <chrono>
#include <atomic>

namespace hft {

class MarketDataHandler {
public:
    virtual ~MarketDataHandler() = default;
    virtual void onMarketDataUpdate(const std::string& symbol, std::shared_ptr<OrderBook> book) = 0;
    virtual void onTrade(const std::string& symbol, Price price, Quantity quantity, Side side) = 0;
};

class OrderHandler {
public:
    virtual ~OrderHandler() = default;
    virtual void onOrderAccepted(const std::string& cl_ord_id, const std::string& order_id) = 0;
    virtual void onOrderRejected(const std::string& cl_ord_id, const std::string& reason) = 0;
    virtual void onOrderFilled(const std::string& cl_ord_id, Price price, Quantity quantity) = 0;
    virtual void onOrderCanceled(const std::string& cl_ord_id) = 0;
};

class FIXMarketDataHandler : public FIX::Application, public FIX::MessageCracker {
private:
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> order_books_;
    std::shared_ptr<MarketDataHandler> market_handler_;
    std::shared_ptr<OrderHandler> order_handler_;
    
    // Performance metrics
    std::atomic<uint64_t> message_count_{0};
    std::atomic<uint64_t> total_latency_ns_{0};
    std::chrono::high_resolution_clock::time_point last_message_time_;
    
    // Session management
    FIX::SessionID market_data_session_;
    FIX::SessionID order_session_;
    
public:
    FIXMarketDataHandler(std::shared_ptr<MarketDataHandler> market_handler,
                         std::shared_ptr<OrderHandler> order_handler);
    
    // FIX::Application interface
    void onCreate(const FIX::SessionID& sessionID) override;
    void onLogon(const FIX::SessionID& sessionID) override;
    void onLogout(const FIX::SessionID& sessionID) override;
    void toAdmin(FIX::Message& message, const FIX::SessionID& sessionID) override;
    void toApp(FIX::Message& message, const FIX::SessionID& sessionID) override;
    void fromAdmin(const FIX::Message& message, const FIX::SessionID& sessionID) override;
    void fromApp(const FIX::Message& message, const FIX::SessionID& sessionID) override;
    
    // Market data message handlers
    void onMessage(const FIX44::MarketDataSnapshotFullRefresh& message, const FIX::SessionID& sessionID);
    void onMessage(const FIX44::MarketDataIncrementalRefresh& message, const FIX::SessionID& sessionID);
    
    // Order management message handlers
    void onMessage(const FIX44::ExecutionReport& message, const FIX::SessionID& sessionID);
    
    // Outgoing message methods
    void subscribeMarketData(const std::string& symbol, int depth = 10);
    void unsubscribeMarketData(const std::string& symbol);
    
    void sendOrder(const std::string& symbol, Side side, Price price, Quantity quantity, 
                   const std::string& cl_ord_id);
    void cancelOrder(const std::string& cl_ord_id, const std::string& orig_cl_ord_id);
    void replaceOrder(const std::string& cl_ord_id, const std::string& orig_cl_ord_id,
                      Price new_price, Quantity new_quantity);
    
    // Performance metrics
    double getAverageLatencyMicros() const;
    uint64_t getMessageCount() const { return message_count_.load(); }
    
    // Order book access
    std::shared_ptr<OrderBook> getOrderBook(const std::string& symbol) const;
    
private:
    void updateLatencyMetrics(const std::chrono::high_resolution_clock::time_point& receive_time);
    void processMarketDataEntry(const FIX::Group& group, const std::string& symbol);
    std::string generateOrderID();
};

// FIX message utilities
class FIXMessageBuilder {
public:
    static FIX44::MarketDataRequest buildMarketDataRequest(
        const std::string& md_req_id,
        const std::string& symbol,
        int market_depth,
        bool subscribe = true
    );
    
    static FIX44::NewOrderSingle buildNewOrderSingle(
        const std::string& cl_ord_id,
        const std::string& symbol,
        Side side,
        Price price,
        Quantity quantity
    );
    
    static FIX44::OrderCancelRequest buildOrderCancelRequest(
        const std::string& cl_ord_id,
        const std::string& orig_cl_ord_id,
        const std::string& symbol,
        Side side
    );
    
    static FIX44::OrderCancelReplaceRequest buildOrderCancelReplaceRequest(
        const std::string& cl_ord_id,
        const std::string& orig_cl_ord_id,
        const std::string& symbol,
        Side side,
        Price new_price,
        Quantity new_quantity
    );
};

// Latency tracker for performance monitoring
class LatencyTracker {
private:
    struct LatencyStats {
        double min_us;
        double max_us;
        double avg_us;
        double p50_us;
        double p95_us;
        double p99_us;
        uint64_t sample_count;
    };
    
    std::vector<double> latencies_;
    mutable std::mutex mutex_;
    
public:
    void recordLatency(double latency_us);
    LatencyStats getStats() const;
    void reset();
};

} // namespace hft