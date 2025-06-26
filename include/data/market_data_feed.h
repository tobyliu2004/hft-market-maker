#pragma once

#include "core/order_book.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <chrono>

// WebSocket++ headers
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

// JSON parsing
#include <nlohmann/json.hpp>

namespace hft {

// Market data message types
enum class MarketDataType {
    QUOTE,
    TRADE,
    ORDER_BOOK_SNAPSHOT,
    ORDER_BOOK_UPDATE,
    TRADING_STATUS,
    STATISTICS
};

// Generic market data message
struct MarketDataMessage {
    MarketDataType type;
    std::string symbol;
    Timestamp timestamp;
    nlohmann::json data;
};

// Trade data structure
struct TradeData {
    std::string symbol;
    Price price;
    Quantity size;
    Side side;
    std::string trade_id;
    Timestamp exchange_timestamp;
    Timestamp local_timestamp;
    std::vector<std::string> conditions;  // Trade conditions/flags
};

// Quote data structure
struct QuoteData {
    std::string symbol;
    Price bid_price;
    Quantity bid_size;
    Price ask_price;
    Quantity ask_size;
    std::string bid_exchange;
    std::string ask_exchange;
    Timestamp exchange_timestamp;
    Timestamp local_timestamp;
};

// Market data callback interfaces
class MarketDataListener {
public:
    virtual ~MarketDataListener() = default;
    
    virtual void onQuote(const QuoteData& quote) = 0;
    virtual void onTrade(const TradeData& trade) = 0;
    virtual void onOrderBookUpdate(const std::string& symbol, 
                                  const OrderBook::Snapshot& snapshot) = 0;
    virtual void onTradingStatus(const std::string& symbol, 
                                const std::string& status) = 0;
    virtual void onError(const std::string& error) = 0;
};

// Base class for market data feeds
class MarketDataFeed {
protected:
    std::vector<std::shared_ptr<MarketDataListener>> listeners_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    // Message queue for async processing
    std::queue<MarketDataMessage> message_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Performance metrics
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_processed_{0};
    std::atomic<uint64_t> total_latency_us_{0};
    
    // Notify all listeners
    void notifyQuote(const QuoteData& quote);
    void notifyTrade(const TradeData& trade);
    void notifyOrderBookUpdate(const std::string& symbol, 
                              const OrderBook::Snapshot& snapshot);
    void notifyTradingStatus(const std::string& symbol, const std::string& status);
    void notifyError(const std::string& error);
    
    // Message processing
    virtual void processMessage(const MarketDataMessage& message) = 0;
    void messageProcessingLoop();
    
public:
    MarketDataFeed();
    virtual ~MarketDataFeed();
    
    // Connection management
    virtual bool connect(const std::string& connection_string) = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    
    // Subscription management
    virtual void subscribe(const std::string& symbol, int depth = 10) = 0;
    virtual void unsubscribe(const std::string& symbol) = 0;
    virtual void subscribeAll(const std::vector<std::string>& symbols) = 0;
    
    // Listener management
    void addListener(std::shared_ptr<MarketDataListener> listener);
    void removeListener(std::shared_ptr<MarketDataListener> listener);
    
    // Performance metrics
    double getAverageLatencyMicros() const;
    uint64_t getMessagesReceived() const { return messages_received_.load(); }
    uint64_t getMessagesProcessed() const { return messages_processed_.load(); }
    double getThroughput() const;  // Messages per second
};

// Polygon.io WebSocket feed implementation
class PolygonWebSocketFeed : public MarketDataFeed {
private:
    typedef websocketpp::client<websocketpp::config::asio_tls_client> client;
    typedef websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> context_ptr;
    
    client ws_client_;
    websocketpp::connection_hdl connection_;
    std::string api_key_;
    std::thread ws_thread_;
    
    // Subscription tracking
    std::unordered_map<std::string, int> subscribed_symbols_;
    std::mutex subscription_mutex_;
    
    // WebSocket callbacks
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, client::message_ptr msg);
    void on_fail(websocketpp::connection_hdl hdl);
    
    // TLS context
    context_ptr on_tls_init(websocketpp::connection_hdl);
    
    // Message parsing
    void parsePolygonMessage(const std::string& message);
    TradeData parsePolygonTrade(const nlohmann::json& trade_json);
    QuoteData parsePolygonQuote(const nlohmann::json& quote_json);
    OrderBook::Snapshot parsePolygonSnapshot(const nlohmann::json& snapshot_json);
    
    // Authentication
    void authenticate();
    
protected:
    void processMessage(const MarketDataMessage& message) override;
    
public:
    explicit PolygonWebSocketFeed(const std::string& api_key);
    ~PolygonWebSocketFeed();
    
    bool connect(const std::string& connection_string) override;
    void disconnect() override;
    bool isConnected() const override;
    
    void subscribe(const std::string& symbol, int depth = 10) override;
    void unsubscribe(const std::string& symbol) override;
    void subscribeAll(const std::vector<std::string>& symbols) override;
};

// NYSE TAQ historical data reader
class NYSETAQDataReader {
private:
    struct TAQHeader {
        char version[8];
        uint32_t num_records;
        uint64_t start_timestamp;
        uint64_t end_timestamp;
        char symbol[12];
    };
    
    struct TAQTradeRecord {
        uint64_t timestamp;
        double price;
        uint32_t size;
        char exchange;
        char sale_condition[4];
        char trade_id[16];
    };
    
    struct TAQQuoteRecord {
        uint64_t timestamp;
        double bid_price;
        uint32_t bid_size;
        double ask_price;
        uint32_t ask_size;
        char bid_exchange;
        char ask_exchange;
        char quote_condition;
    };
    
    std::string base_path_;
    std::unordered_map<std::string, std::vector<TradeData>> trades_cache_;
    std::unordered_map<std::string, std::vector<QuoteData>> quotes_cache_;
    
    // Binary file parsing
    std::vector<TradeData> readTradeFile(const std::string& filename);
    std::vector<QuoteData> readQuoteFile(const std::string& filename);
    
public:
    explicit NYSETAQDataReader(const std::string& base_path);
    
    // Load historical data for a date range
    void loadData(const std::string& symbol, 
                  const std::string& start_date, 
                  const std::string& end_date);
    
    // Get loaded data
    std::vector<TradeData> getTrades(const std::string& symbol, 
                                    Timestamp start_time, 
                                    Timestamp end_time);
    
    std::vector<QuoteData> getQuotes(const std::string& symbol,
                                    Timestamp start_time,
                                    Timestamp end_time);
    
    // Replay historical data through listeners
    void replay(const std::string& symbol,
                Timestamp start_time,
                Timestamp end_time,
                double speed_multiplier,
                std::shared_ptr<MarketDataListener> listener);
};

// Simulated market data feed for backtesting
class SimulatedMarketDataFeed : public MarketDataFeed {
private:
    NYSETAQDataReader taq_reader_;
    std::thread replay_thread_;
    std::atomic<bool> replaying_;
    double replay_speed_;
    
    // Current replay state
    std::string current_symbol_;
    Timestamp current_time_;
    Timestamp end_time_;
    
    // Order book reconstruction
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> order_books_;
    
    void replayLoop();
    void reconstructOrderBook(const std::string& symbol, const QuoteData& quote);
    
protected:
    void processMessage(const MarketDataMessage& message) override;
    
public:
    explicit SimulatedMarketDataFeed(const std::string& historical_data_path);
    ~SimulatedMarketDataFeed();
    
    bool connect(const std::string& connection_string) override;
    void disconnect() override;
    bool isConnected() const override { return replaying_.load(); }
    
    void subscribe(const std::string& symbol, int depth = 10) override;
    void unsubscribe(const std::string& symbol) override;
    void subscribeAll(const std::vector<std::string>& symbols) override;
    
    // Backtesting controls
    void startReplay(const std::string& symbol,
                    const std::string& date,
                    double speed_multiplier = 1.0);
    void pauseReplay();
    void resumeReplay();
    void stopReplay();
    
    // Jump to specific time
    void seekTo(Timestamp target_time);
    
    // Get current replay time
    Timestamp getCurrentTime() const { return current_time_; }
};

// Market data recorder for creating datasets
class MarketDataRecorder : public MarketDataListener {
private:
    std::string output_directory_;
    std::ofstream trade_file_;
    std::ofstream quote_file_;
    std::ofstream book_file_;
    
    std::mutex file_mutex_;
    std::atomic<uint64_t> records_written_{0};
    
    // Compression
    bool use_compression_;
    size_t buffer_size_;
    std::vector<uint8_t> write_buffer_;
    
    void writeTradeRecord(const TradeData& trade);
    void writeQuoteRecord(const QuoteData& quote);
    void writeBookSnapshot(const std::string& symbol, 
                          const OrderBook::Snapshot& snapshot);
    
public:
    MarketDataRecorder(const std::string& output_directory, 
                      bool use_compression = true);
    ~MarketDataRecorder();
    
    // MarketDataListener interface
    void onQuote(const QuoteData& quote) override;
    void onTrade(const TradeData& trade) override;
    void onOrderBookUpdate(const std::string& symbol, 
                          const OrderBook::Snapshot& snapshot) override;
    void onTradingStatus(const std::string& symbol, 
                        const std::string& status) override;
    void onError(const std::string& error) override;
    
    // Control recording
    void startRecording();
    void stopRecording();
    void flush();
    
    uint64_t getRecordsWritten() const { return records_written_.load(); }
};

// Market data aggregator for multiple feeds
class MarketDataAggregator : public MarketDataListener {
private:
    std::vector<std::shared_ptr<MarketDataFeed>> feeds_;
    std::vector<std::shared_ptr<MarketDataListener>> listeners_;
    
    // Best bid/offer tracking
    struct BBO {
        Price bid_price;
        Quantity bid_size;
        std::string bid_source;
        Price ask_price;
        Quantity ask_size;
        std::string ask_source;
        Timestamp last_update;
    };
    
    std::unordered_map<std::string, BBO> nbbo_;
    mutable std::mutex nbbo_mutex_;
    
    // Consolidated order book
    std::unordered_map<std::string, std::shared_ptr<OrderBook>> consolidated_books_;
    
    void updateNBBO(const std::string& symbol, const QuoteData& quote);
    
public:
    MarketDataAggregator();
    
    // Feed management
    void addFeed(std::shared_ptr<MarketDataFeed> feed, const std::string& name);
    void removeFeed(const std::string& name);
    
    // Listener management
    void addListener(std::shared_ptr<MarketDataListener> listener);
    void removeListener(std::shared_ptr<MarketDataListener> listener);
    
    // MarketDataListener interface (from individual feeds)
    void onQuote(const QuoteData& quote) override;
    void onTrade(const TradeData& trade) override;
    void onOrderBookUpdate(const std::string& symbol, 
                          const OrderBook::Snapshot& snapshot) override;
    void onTradingStatus(const std::string& symbol, 
                        const std::string& status) override;
    void onError(const std::string& error) override;
    
    // Get consolidated data
    BBO getNBBO(const std::string& symbol) const;
    std::shared_ptr<OrderBook> getConsolidatedBook(const std::string& symbol) const;
};

// Market data normalizer for different formats
class MarketDataNormalizer {
private:
    // Symbol mapping
    std::unordered_map<std::string, std::string> symbol_map_;
    
    // Price/size adjustments
    std::unordered_map<std::string, double> price_multipliers_;
    std::unordered_map<std::string, double> size_multipliers_;
    
public:
    MarketDataNormalizer();
    
    // Configure mappings
    void addSymbolMapping(const std::string& from, const std::string& to);
    void setPriceMultiplier(const std::string& symbol, double multiplier);
    void setSizeMultiplier(const std::string& symbol, double multiplier);
    
    // Normalize data
    TradeData normalizeTrade(const TradeData& trade);
    QuoteData normalizeQuote(const QuoteData& quote);
    
    // Load configuration from file
    void loadConfiguration(const std::string& config_file);
};

} // namespace hft