#include "core/fix_handler.h"
#include <quickfix/SessionSettings.h>
#include <quickfix/field/MDReqID.h>
#include <quickfix/field/SubscriptionRequestType.h>
#include <quickfix/field/MarketDepth.h>
#include <quickfix/field/MDUpdateType.h>
#include <quickfix/field/Symbol.h>
#include <quickfix/field/NoMDEntryTypes.h>
#include <quickfix/field/MDEntryType.h>
#include <quickfix/field/NoRelatedSym.h>
#include <quickfix/field/MDEntryPx.h>
#include <quickfix/field/MDEntrySize.h>
#include <quickfix/field/ClOrdID.h>
#include <quickfix/field/Side.h>
#include <quickfix/field/OrderQty.h>
#include <quickfix/field/Price.h>
#include <quickfix/field/OrdType.h>
#include <quickfix/field/TimeInForce.h>
#include <quickfix/field/TransactTime.h>
#include <quickfix/field/ExecType.h>
#include <quickfix/field/OrdStatus.h>
#include <quickfix/field/OrderID.h>
#include <quickfix/field/ExecID.h>
#include <quickfix/field/LeavesQty.h>
#include <quickfix/field/CumQty.h>
#include <quickfix/field/AvgPx.h>
#include <quickfix/field/Text.h>
#include <quickfix/field/OrdRejReason.h>
#include <quickfix/field/OrigClOrdID.h>

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace hft {

FIXMarketDataHandler::FIXMarketDataHandler(
    std::shared_ptr<MarketDataHandler> market_handler,
    std::shared_ptr<OrderHandler> order_handler)
    : market_handler_(market_handler), 
      order_handler_(order_handler),
      last_message_time_(std::chrono::high_resolution_clock::now()) {
}

void FIXMarketDataHandler::onCreate(const FIX::SessionID& sessionID) {
    std::cout << "Session created: " << sessionID << std::endl;
}

void FIXMarketDataHandler::onLogon(const FIX::SessionID& sessionID) {
    std::cout << "Logon: " << sessionID << std::endl;
    
    // Determine session type based on target comp ID or other criteria
    std::string targetCompID = sessionID.getTargetCompID();
    if (targetCompID.find("MARKET") != std::string::npos) {
        market_data_session_ = sessionID;
    } else if (targetCompID.find("ORDER") != std::string::npos) {
        order_session_ = sessionID;
    }
}

void FIXMarketDataHandler::onLogout(const FIX::SessionID& sessionID) {
    std::cout << "Logout: " << sessionID << std::endl;
}

void FIXMarketDataHandler::toAdmin(FIX::Message& message, const FIX::SessionID& sessionID) {
    // Handle administrative messages (logon, heartbeat, etc.)
    FIX::MsgType msgType;
    message.getHeader().getField(msgType);
    
    if (msgType == FIX::MsgType_Logon) {
        // Add any custom logon fields if needed
        // message.setField(FIX::Username("username"));
        // message.setField(FIX::Password("password"));
    }
}

void FIXMarketDataHandler::toApp(FIX::Message& message, const FIX::SessionID& sessionID) {
    // Log outgoing application messages
    std::cout << "Outgoing: " << message << std::endl;
}

void FIXMarketDataHandler::fromAdmin(const FIX::Message& message, const FIX::SessionID& sessionID) {
    // Handle incoming administrative messages
}

void FIXMarketDataHandler::fromApp(const FIX::Message& message, const FIX::SessionID& sessionID) {
    auto receive_time = std::chrono::high_resolution_clock::now();
    updateLatencyMetrics(receive_time);
    
    try {
        crack(message, sessionID);
    } catch (const FIX::UnsupportedMessageType& e) {
        std::cerr << "Unsupported message type: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::onMessage(const FIX44::MarketDataSnapshotFullRefresh& message, 
                                      const FIX::SessionID& sessionID) {
    try {
        FIX::Symbol symbol;
        message.get(symbol);
        
        auto book = std::make_shared<OrderBook>();
        
        // Process market data entries
        FIX::NoMDEntries noMDEntries;
        message.get(noMDEntries);
        
        for (int i = 1; i <= noMDEntries; ++i) {
            FIX44::MarketDataSnapshotFullRefresh::NoMDEntries group;
            message.getGroup(i, group);
            
            FIX::MDEntryType entryType;
            FIX::MDEntryPx price;
            FIX::MDEntrySize size;
            
            group.get(entryType);
            group.get(price);
            group.get(size);
            
            // Create order for the book
            static std::atomic<OrderId> order_id_gen{1};
            auto order = std::make_shared<Order>(
                order_id_gen++,
                (entryType == FIX::MDEntryType_BID) ? Side::BUY : Side::SELL,
                price,
                static_cast<Quantity>(size),
                std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count())
            );
            
            book->addOrder(order);
        }
        
        // Update order book map
        order_books_[symbol] = book;
        
        // Notify handler
        if (market_handler_) {
            market_handler_->onMarketDataUpdate(symbol, book);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing market data snapshot: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::onMessage(const FIX44::MarketDataIncrementalRefresh& message, 
                                      const FIX::SessionID& sessionID) {
    try {
        FIX::NoMDEntries noMDEntries;
        message.get(noMDEntries);
        
        for (int i = 1; i <= noMDEntries; ++i) {
            FIX44::MarketDataIncrementalRefresh::NoMDEntries group;
            message.getGroup(i, group);
            
            FIX::Symbol symbol;
            FIX::MDUpdateAction action;
            FIX::MDEntryType entryType;
            
            group.get(symbol);
            group.get(action);
            group.get(entryType);
            
            auto book_it = order_books_.find(symbol);
            if (book_it == order_books_.end()) {
                // Create new book if doesn't exist
                order_books_[symbol] = std::make_shared<OrderBook>();
                book_it = order_books_.find(symbol);
            }
            
            auto book = book_it->second;
            
            if (action == FIX::MDUpdateAction_NEW) {
                FIX::MDEntryPx price;
                FIX::MDEntrySize size;
                group.get(price);
                group.get(size);
                
                static std::atomic<OrderId> order_id_gen{1000000};
                auto order = std::make_shared<Order>(
                    order_id_gen++,
                    (entryType == FIX::MDEntryType_BID) ? Side::BUY : Side::SELL,
                    price,
                    static_cast<Quantity>(size),
                    std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch().count())
                );
                
                book->addOrder(order);
                
            } else if (action == FIX::MDUpdateAction_CHANGE) {
                // Handle order modifications
                // In practice, we'd need an order ID to modify specific orders
                
            } else if (action == FIX::MDUpdateAction_DELETE) {
                // Handle order deletions
                // In practice, we'd need an order ID to delete specific orders
            }
            
            // Check for trades
            if (entryType == FIX::MDEntryType_TRADE) {
                FIX::MDEntryPx price;
                FIX::MDEntrySize size;
                group.get(price);
                group.get(size);
                
                // Determine trade side based on price relative to spread
                Side trade_side = Side::BUY;  // Default, would need more info
                
                if (market_handler_) {
                    market_handler_->onTrade(symbol, price, static_cast<Quantity>(size), trade_side);
                }
            }
            
            // Notify handler of book update
            if (market_handler_ && (entryType == FIX::MDEntryType_BID || 
                                   entryType == FIX::MDEntryType_OFFER)) {
                market_handler_->onMarketDataUpdate(symbol, book);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing incremental refresh: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::onMessage(const FIX44::ExecutionReport& message, 
                                      const FIX::SessionID& sessionID) {
    try {
        FIX::ClOrdID clOrdID;
        FIX::ExecType execType;
        FIX::OrdStatus ordStatus;
        
        message.get(clOrdID);
        message.get(execType);
        message.get(ordStatus);
        
        if (execType == FIX::ExecType_NEW) {
            FIX::OrderID orderID;
            message.get(orderID);
            
            if (order_handler_) {
                order_handler_->onOrderAccepted(clOrdID, orderID);
            }
            
        } else if (execType == FIX::ExecType_REJECTED) {
            FIX::Text text;
            std::string reason = "Unknown";
            
            if (message.isSetField(text)) {
                message.get(text);
                reason = text;
            } else if (message.isSetField(FIX::FIELD::OrdRejReason)) {
                FIX::OrdRejReason rejReason;
                message.get(rejReason);
                reason = std::to_string(rejReason);
            }
            
            if (order_handler_) {
                order_handler_->onOrderRejected(clOrdID, reason);
            }
            
        } else if (execType == FIX::ExecType_FILL || execType == FIX::ExecType_PARTIAL_FILL) {
            FIX::LastPx lastPx;
            FIX::LastQty lastQty;
            
            message.get(lastPx);
            message.get(lastQty);
            
            if (order_handler_) {
                order_handler_->onOrderFilled(clOrdID, lastPx, static_cast<Quantity>(lastQty));
            }
            
        } else if (execType == FIX::ExecType_CANCELED) {
            if (order_handler_) {
                order_handler_->onOrderCanceled(clOrdID);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing execution report: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::subscribeMarketData(const std::string& symbol, int depth) {
    try {
        auto request = FIXMessageBuilder::buildMarketDataRequest(
            generateOrderID(), symbol, depth, true
        );
        
        FIX::Session::sendToTarget(request, market_data_session_);
        
    } catch (const std::exception& e) {
        std::cerr << "Error subscribing to market data: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::unsubscribeMarketData(const std::string& symbol) {
    try {
        auto request = FIXMessageBuilder::buildMarketDataRequest(
            generateOrderID(), symbol, 0, false
        );
        
        FIX::Session::sendToTarget(request, market_data_session_);
        
    } catch (const std::exception& e) {
        std::cerr << "Error unsubscribing from market data: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::sendOrder(const std::string& symbol, Side side, Price price, 
                                     Quantity quantity, const std::string& cl_ord_id) {
    try {
        auto order = FIXMessageBuilder::buildNewOrderSingle(
            cl_ord_id, symbol, side, price, quantity
        );
        
        FIX::Session::sendToTarget(order, order_session_);
        
    } catch (const std::exception& e) {
        std::cerr << "Error sending order: " << e.what() << std::endl;
    }
}

void FIXMarketDataHandler::cancelOrder(const std::string& cl_ord_id, 
                                       const std::string& orig_cl_ord_id) {
    // Implementation would require tracking original order details
    // For now, this is a placeholder
}

void FIXMarketDataHandler::replaceOrder(const std::string& cl_ord_id, 
                                        const std::string& orig_cl_ord_id,
                                        Price new_price, Quantity new_quantity) {
    // Implementation would require tracking original order details
    // For now, this is a placeholder
}

double FIXMarketDataHandler::getAverageLatencyMicros() const {
    uint64_t count = message_count_.load();
    if (count == 0) return 0.0;
    
    uint64_t total_ns = total_latency_ns_.load();
    return static_cast<double>(total_ns) / (count * 1000.0);
}

std::shared_ptr<OrderBook> FIXMarketDataHandler::getOrderBook(const std::string& symbol) const {
    auto it = order_books_.find(symbol);
    return (it != order_books_.end()) ? it->second : nullptr;
}

void FIXMarketDataHandler::updateLatencyMetrics(
    const std::chrono::high_resolution_clock::time_point& receive_time) {
    
    auto latency = receive_time - last_message_time_;
    total_latency_ns_ += latency.count();
    message_count_++;
    last_message_time_ = receive_time;
}

std::string FIXMarketDataHandler::generateOrderID() {
    static std::atomic<uint64_t> id_counter{1};
    
    std::stringstream ss;
    ss << "MD" << std::setfill('0') << std::setw(12) << id_counter++;
    return ss.str();
}

// FIXMessageBuilder implementations
FIX44::MarketDataRequest FIXMessageBuilder::buildMarketDataRequest(
    const std::string& md_req_id,
    const std::string& symbol,
    int market_depth,
    bool subscribe) {
    
    FIX44::MarketDataRequest request;
    
    request.set(FIX::MDReqID(md_req_id));
    request.set(FIX::SubscriptionRequestType(
        subscribe ? FIX::SubscriptionRequestType_SNAPSHOT_PLUS_UPDATES : 
                   FIX::SubscriptionRequestType_DISABLE_PREVIOUS_SNAPSHOT_PLUS_UPDATE_REQUEST
    ));
    request.set(FIX::MarketDepth(market_depth));
    request.set(FIX::MDUpdateType(FIX::MDUpdateType_INCREMENTAL_REFRESH));
    
    // Add entry types
    request.set(FIX::NoMDEntryTypes(2));
    
    FIX44::MarketDataRequest::NoMDEntryTypes entryGroup1;
    entryGroup1.set(FIX::MDEntryType(FIX::MDEntryType_BID));
    request.addGroup(entryGroup1);
    
    FIX44::MarketDataRequest::NoMDEntryTypes entryGroup2;
    entryGroup2.set(FIX::MDEntryType(FIX::MDEntryType_OFFER));
    request.addGroup(entryGroup2);
    
    // Add symbol
    request.set(FIX::NoRelatedSym(1));
    FIX44::MarketDataRequest::NoRelatedSym symGroup;
    symGroup.set(FIX::Symbol(symbol));
    request.addGroup(symGroup);
    
    return request;
}

FIX44::NewOrderSingle FIXMessageBuilder::buildNewOrderSingle(
    const std::string& cl_ord_id,
    const std::string& symbol,
    Side side,
    Price price,
    Quantity quantity) {
    
    FIX44::NewOrderSingle order;
    
    order.set(FIX::ClOrdID(cl_ord_id));
    order.set(FIX::Symbol(symbol));
    order.set(FIX::Side(side == Side::BUY ? FIX::Side_BUY : FIX::Side_SELL));
    order.set(FIX::OrderQty(static_cast<double>(quantity)));
    order.set(FIX::Price(price));
    order.set(FIX::OrdType(FIX::OrdType_LIMIT));
    order.set(FIX::TimeInForce(FIX::TimeInForce_DAY));
    order.set(FIX::TransactTime());
    
    return order;
}

// LatencyTracker implementations
void LatencyTracker::recordLatency(double latency_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.push_back(latency_us);
    
    // Keep only last 100k samples to prevent unbounded growth
    if (latencies_.size() > 100000) {
        latencies_.erase(latencies_.begin(), latencies_.begin() + 50000);
    }
}

LatencyTracker::LatencyStats LatencyTracker::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (latencies_.empty()) {
        return {0, 0, 0, 0, 0, 0, 0};
    }
    
    std::vector<double> sorted_latencies = latencies_;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    
    LatencyStats stats;
    stats.min_us = sorted_latencies.front();
    stats.max_us = sorted_latencies.back();
    stats.avg_us = std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0.0) / 
                   sorted_latencies.size();
    
    size_t size = sorted_latencies.size();
    stats.p50_us = sorted_latencies[size * 0.50];
    stats.p95_us = sorted_latencies[size * 0.95];
    stats.p99_us = sorted_latencies[size * 0.99];
    stats.sample_count = size;
    
    return stats;
}

void LatencyTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.clear();
}

} // namespace hft