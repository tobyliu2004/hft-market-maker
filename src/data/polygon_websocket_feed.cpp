#include "data/market_data_feed.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace hft {

// Base MarketDataFeed implementation
MarketDataFeed::MarketDataFeed() : running_(false) {}

MarketDataFeed::~MarketDataFeed() {
    running_ = false;
    queue_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void MarketDataFeed::notifyQuote(const QuoteData& quote) {
    for (auto& listener : listeners_) {
        listener->onQuote(quote);
    }
}

void MarketDataFeed::notifyTrade(const TradeData& trade) {
    for (auto& listener : listeners_) {
        listener->onTrade(trade);
    }
}

void MarketDataFeed::notifyOrderBookUpdate(const std::string& symbol, 
                                          const OrderBook::Snapshot& snapshot) {
    for (auto& listener : listeners_) {
        listener->onOrderBookUpdate(symbol, snapshot);
    }
}

void MarketDataFeed::notifyTradingStatus(const std::string& symbol, 
                                        const std::string& status) {
    for (auto& listener : listeners_) {
        listener->onTradingStatus(symbol, status);
    }
}

void MarketDataFeed::notifyError(const std::string& error) {
    for (auto& listener : listeners_) {
        listener->onError(error);
    }
}

void MarketDataFeed::messageProcessingLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !message_queue_.empty() || !running_; });
        
        while (!message_queue_.empty()) {
            MarketDataMessage message = std::move(message_queue_.front());
            message_queue_.pop();
            lock.unlock();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            processMessage(message);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();
            total_latency_us_ += latency;
            messages_processed_++;
            
            lock.lock();
        }
    }
}

void MarketDataFeed::addListener(std::shared_ptr<MarketDataListener> listener) {
    listeners_.push_back(listener);
}

void MarketDataFeed::removeListener(std::shared_ptr<MarketDataListener> listener) {
    listeners_.erase(
        std::remove(listeners_.begin(), listeners_.end(), listener),
        listeners_.end()
    );
}

double MarketDataFeed::getAverageLatencyMicros() const {
    uint64_t processed = messages_processed_.load();
    if (processed == 0) return 0.0;
    
    return static_cast<double>(total_latency_us_.load()) / processed;
}

double MarketDataFeed::getThroughput() const {
    // TODO: Implement proper throughput calculation with time tracking
    return static_cast<double>(messages_processed_.load());
}

// PolygonWebSocketFeed implementation
PolygonWebSocketFeed::PolygonWebSocketFeed(const std::string& api_key) 
    : api_key_(api_key) {
    
    // Initialize WebSocket client
    ws_client_.init_asio();
    
    // Set up handlers
    ws_client_.set_open_handler(
        std::bind(&PolygonWebSocketFeed::on_open, this, std::placeholders::_1)
    );
    ws_client_.set_close_handler(
        std::bind(&PolygonWebSocketFeed::on_close, this, std::placeholders::_1)
    );
    ws_client_.set_message_handler(
        std::bind(&PolygonWebSocketFeed::on_message, this, 
                  std::placeholders::_1, std::placeholders::_2)
    );
    ws_client_.set_fail_handler(
        std::bind(&PolygonWebSocketFeed::on_fail, this, std::placeholders::_1)
    );
    ws_client_.set_tls_init_handler(
        std::bind(&PolygonWebSocketFeed::on_tls_init, this, std::placeholders::_1)
    );
    
    // Configure WebSocket settings
    ws_client_.set_access_channels(websocketpp::log::alevel::none);
    ws_client_.set_error_channels(websocketpp::log::elevel::all);
}

PolygonWebSocketFeed::~PolygonWebSocketFeed() {
    disconnect();
}

bool PolygonWebSocketFeed::connect(const std::string& connection_string) {
    try {
        websocketpp::lib::error_code ec;
        client::connection_ptr con = ws_client_.get_connection(connection_string, ec);
        
        if (ec) {
            notifyError("Connection initialization error: " + ec.message());
            return false;
        }
        
        ws_client_.connect(con);
        
        // Start WebSocket thread
        ws_thread_ = std::thread([this]() {
            ws_client_.run();
        });
        
        // Start message processing thread
        running_ = true;
        worker_thread_ = std::thread(&PolygonWebSocketFeed::messageProcessingLoop, this);
        
        return true;
        
    } catch (const std::exception& e) {
        notifyError("Connection error: " + std::string(e.what()));
        return false;
    }
}

void PolygonWebSocketFeed::disconnect() {
    running_ = false;
    
    if (isConnected()) {
        websocketpp::lib::error_code ec;
        ws_client_.close(connection_, websocketpp::close::status::going_away, "", ec);
    }
    
    ws_client_.stop();
    
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
    
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

bool PolygonWebSocketFeed::isConnected() const {
    if (!connection_.expired()) {
        auto con = ws_client_.get_con_from_hdl(connection_);
        return con->get_state() == websocketpp::session::state::open;
    }
    return false;
}

void PolygonWebSocketFeed::subscribe(const std::string& symbol, int depth) {
    if (!isConnected()) {
        notifyError("Cannot subscribe: not connected");
        return;
    }
    
    nlohmann::json subscribe_msg = {
        {"action", "subscribe"},
        {"params", {
            {"T", {symbol}},    // Trades
            {"Q", {symbol}},    // Quotes
            {"A", {symbol}}     // Aggregates
        }}
    };
    
    websocketpp::lib::error_code ec;
    ws_client_.send(connection_, subscribe_msg.dump(), 
                    websocketpp::frame::opcode::text, ec);
    
    if (ec) {
        notifyError("Subscribe error: " + ec.message());
    } else {
        std::lock_guard<std::mutex> lock(subscription_mutex_);
        subscribed_symbols_[symbol] = depth;
    }
}

void PolygonWebSocketFeed::unsubscribe(const std::string& symbol) {
    if (!isConnected()) {
        return;
    }
    
    nlohmann::json unsubscribe_msg = {
        {"action", "unsubscribe"},
        {"params", {
            {"T", {symbol}},
            {"Q", {symbol}},
            {"A", {symbol}}
        }}
    };
    
    websocketpp::lib::error_code ec;
    ws_client_.send(connection_, unsubscribe_msg.dump(), 
                    websocketpp::frame::opcode::text, ec);
    
    if (!ec) {
        std::lock_guard<std::mutex> lock(subscription_mutex_);
        subscribed_symbols_.erase(symbol);
    }
}

void PolygonWebSocketFeed::subscribeAll(const std::vector<std::string>& symbols) {
    for (const auto& symbol : symbols) {
        subscribe(symbol);
    }
}

void PolygonWebSocketFeed::on_open(websocketpp::connection_hdl hdl) {
    connection_ = hdl;
    authenticate();
}

void PolygonWebSocketFeed::on_close(websocketpp::connection_hdl hdl) {
    notifyError("WebSocket connection closed");
}

void PolygonWebSocketFeed::on_message(websocketpp::connection_hdl hdl, 
                                      client::message_ptr msg) {
    messages_received_++;
    
    try {
        parsePolygonMessage(msg->get_payload());
    } catch (const std::exception& e) {
        notifyError("Message parsing error: " + std::string(e.what()));
    }
}

void PolygonWebSocketFeed::on_fail(websocketpp::connection_hdl hdl) {
    notifyError("WebSocket connection failed");
}

PolygonWebSocketFeed::context_ptr PolygonWebSocketFeed::on_tls_init(
    websocketpp::connection_hdl) {
    
    context_ptr ctx = websocketpp::lib::make_shared<
        websocketpp::lib::asio::ssl::context>(
            websocketpp::lib::asio::ssl::context::tlsv12
    );
    
    try {
        ctx->set_options(
            websocketpp::lib::asio::ssl::context::default_workarounds |
            websocketpp::lib::asio::ssl::context::no_sslv2 |
            websocketpp::lib::asio::ssl::context::no_sslv3 |
            websocketpp::lib::asio::ssl::context::single_dh_use
        );
    } catch (std::exception& e) {
        std::cerr << "TLS initialization error: " << e.what() << std::endl;
    }
    
    return ctx;
}

void PolygonWebSocketFeed::authenticate() {
    nlohmann::json auth_msg = {
        {"action", "auth"},
        {"params", api_key_}
    };
    
    websocketpp::lib::error_code ec;
    ws_client_.send(connection_, auth_msg.dump(), 
                    websocketpp::frame::opcode::text, ec);
    
    if (ec) {
        notifyError("Authentication error: " + ec.message());
    }
}

void PolygonWebSocketFeed::parsePolygonMessage(const std::string& message) {
    auto json_msg = nlohmann::json::parse(message);
    
    // Handle different message types
    if (json_msg.contains("ev")) {
        std::string event = json_msg["ev"];
        
        MarketDataMessage msg;
        msg.timestamp = std::chrono::nanoseconds(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        msg.data = json_msg;
        
        if (event == "T") {
            // Trade message
            msg.type = MarketDataType::TRADE;
            msg.symbol = json_msg["sym"];
        } else if (event == "Q") {
            // Quote message
            msg.type = MarketDataType::QUOTE;
            msg.symbol = json_msg["sym"];
        } else if (event == "A") {
            // Aggregate/bar message
            msg.type = MarketDataType::STATISTICS;
            msg.symbol = json_msg["sym"];
        } else if (event == "status") {
            // Status message
            msg.type = MarketDataType::TRADING_STATUS;
            msg.symbol = json_msg.value("sym", "");
        }
        
        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            message_queue_.push(std::move(msg));
        }
        queue_cv_.notify_one();
    }
}

TradeData PolygonWebSocketFeed::parsePolygonTrade(const nlohmann::json& trade_json) {
    TradeData trade;
    
    trade.symbol = trade_json["sym"];
    trade.price = trade_json["p"];
    trade.size = trade_json["s"];
    trade.trade_id = std::to_string(trade_json.value("i", 0));
    
    // Polygon uses Unix timestamp in nanoseconds
    uint64_t timestamp_ns = trade_json["t"];
    trade.exchange_timestamp = Timestamp(timestamp_ns);
    trade.local_timestamp = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    
    // Parse conditions if present
    if (trade_json.contains("c")) {
        for (const auto& condition : trade_json["c"]) {
            trade.conditions.push_back(std::to_string(condition));
        }
    }
    
    // Determine trade side from conditions or price movement
    // This is simplified - real implementation would track bid/ask
    trade.side = Side::BUY;  // Default
    
    return trade;
}

QuoteData PolygonWebSocketFeed::parsePolygonQuote(const nlohmann::json& quote_json) {
    QuoteData quote;
    
    quote.symbol = quote_json["sym"];
    quote.bid_price = quote_json["bp"];
    quote.bid_size = quote_json["bs"];
    quote.ask_price = quote_json["ap"];
    quote.ask_size = quote_json["as"];
    
    // Exchange codes
    quote.bid_exchange = std::to_string(quote_json.value("bx", 0));
    quote.ask_exchange = std::to_string(quote_json.value("ax", 0));
    
    // Timestamps
    uint64_t timestamp_ns = quote_json["t"];
    quote.exchange_timestamp = Timestamp(timestamp_ns);
    quote.local_timestamp = std::chrono::nanoseconds(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    
    return quote;
}

void PolygonWebSocketFeed::processMessage(const MarketDataMessage& message) {
    switch (message.type) {
        case MarketDataType::TRADE: {
            TradeData trade = parsePolygonTrade(message.data);
            notifyTrade(trade);
            break;
        }
        
        case MarketDataType::QUOTE: {
            QuoteData quote = parsePolygonQuote(message.data);
            notifyQuote(quote);
            
            // Update order book with quote
            // In a real implementation, we'd maintain a proper L2 book
            OrderBook::Snapshot snapshot;
            snapshot.bids = {{quote.bid_price, quote.bid_size}};
            snapshot.asks = {{quote.ask_price, quote.ask_size}};
            snapshot.timestamp = quote.exchange_timestamp;
            snapshot.sequence = messages_processed_;
            
            notifyOrderBookUpdate(message.symbol, snapshot);
            break;
        }
        
        case MarketDataType::TRADING_STATUS: {
            std::string status = message.data.value("status", "unknown");
            notifyTradingStatus(message.symbol, status);
            break;
        }
        
        default:
            break;
    }
}

// SimulatedMarketDataFeed implementation
SimulatedMarketDataFeed::SimulatedMarketDataFeed(const std::string& historical_data_path)
    : taq_reader_(historical_data_path), replaying_(false), replay_speed_(1.0) {
}

SimulatedMarketDataFeed::~SimulatedMarketDataFeed() {
    stopReplay();
}

bool SimulatedMarketDataFeed::connect(const std::string& connection_string) {
    // For simulated feed, connection string could be a config file
    // or just return true for compatibility
    running_ = true;
    worker_thread_ = std::thread(&SimulatedMarketDataFeed::messageProcessingLoop, this);
    return true;
}

void SimulatedMarketDataFeed::disconnect() {
    stopReplay();
    running_ = false;
    queue_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void SimulatedMarketDataFeed::subscribe(const std::string& symbol, int depth) {
    // For simulated feed, just track subscription
    order_books_[symbol] = std::make_shared<OrderBook>();
}

void SimulatedMarketDataFeed::unsubscribe(const std::string& symbol) {
    order_books_.erase(symbol);
}

void SimulatedMarketDataFeed::subscribeAll(const std::vector<std::string>& symbols) {
    for (const auto& symbol : symbols) {
        subscribe(symbol);
    }
}

void SimulatedMarketDataFeed::startReplay(const std::string& symbol,
                                         const std::string& date,
                                         double speed_multiplier) {
    if (replaying_) {
        stopReplay();
    }
    
    current_symbol_ = symbol;
    replay_speed_ = speed_multiplier;
    replaying_ = true;
    
    // Load data for the specified date
    taq_reader_.loadData(symbol, date, date);
    
    // Start replay thread
    replay_thread_ = std::thread(&SimulatedMarketDataFeed::replayLoop, this);
}

void SimulatedMarketDataFeed::stopReplay() {
    replaying_ = false;
    
    if (replay_thread_.joinable()) {
        replay_thread_.join();
    }
}

void SimulatedMarketDataFeed::replayLoop() {
    // Get the full day's data
    auto start_time = Timestamp(0);  // Beginning of day
    auto end_time = Timestamp(std::chrono::hours(24).count());  // End of day
    
    auto trades = taq_reader_.getTrades(current_symbol_, start_time, end_time);
    auto quotes = taq_reader_.getQuotes(current_symbol_, start_time, end_time);
    
    size_t trade_idx = 0;
    size_t quote_idx = 0;
    
    auto replay_start_time = std::chrono::high_resolution_clock::now();
    Timestamp first_timestamp = std::min(
        trades.empty() ? Timestamp::max() : trades[0].exchange_timestamp,
        quotes.empty() ? Timestamp::max() : quotes[0].exchange_timestamp
    );
    
    while (replaying_ && (trade_idx < trades.size() || quote_idx < quotes.size())) {
        // Determine next event
        bool process_trade = false;
        
        if (trade_idx < trades.size() && quote_idx < quotes.size()) {
            process_trade = trades[trade_idx].exchange_timestamp <= 
                          quotes[quote_idx].exchange_timestamp;
        } else if (trade_idx < trades.size()) {
            process_trade = true;
        }
        
        if (process_trade && trade_idx < trades.size()) {
            // Process trade
            const auto& trade = trades[trade_idx];
            current_time_ = trade.exchange_timestamp;
            
            // Wait for appropriate time based on replay speed
            auto elapsed_real = std::chrono::high_resolution_clock::now() - replay_start_time;
            auto elapsed_sim = current_time_ - first_timestamp;
            auto target_real = std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed_sim) / replay_speed_;
            
            if (target_real > elapsed_real) {
                std::this_thread::sleep_for(target_real - elapsed_real);
            }
            
            // Create message and add to queue
            MarketDataMessage msg;
            msg.type = MarketDataType::TRADE;
            msg.symbol = trade.symbol;
            msg.timestamp = trade.exchange_timestamp;
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                message_queue_.push(msg);
            }
            queue_cv_.notify_one();
            
            trade_idx++;
            
        } else if (quote_idx < quotes.size()) {
            // Process quote
            const auto& quote = quotes[quote_idx];
            current_time_ = quote.exchange_timestamp;
            
            // Wait for appropriate time
            auto elapsed_real = std::chrono::high_resolution_clock::now() - replay_start_time;
            auto elapsed_sim = current_time_ - first_timestamp;
            auto target_real = std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed_sim) / replay_speed_;
            
            if (target_real > elapsed_real) {
                std::this_thread::sleep_for(target_real - elapsed_real);
            }
            
            // Update order book
            reconstructOrderBook(quote.symbol, quote);
            
            // Create message and add to queue
            MarketDataMessage msg;
            msg.type = MarketDataType::QUOTE;
            msg.symbol = quote.symbol;
            msg.timestamp = quote.exchange_timestamp;
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                message_queue_.push(msg);
            }
            queue_cv_.notify_one();
            
            quote_idx++;
        }
    }
}

void SimulatedMarketDataFeed::reconstructOrderBook(const std::string& symbol, 
                                                  const QuoteData& quote) {
    auto it = order_books_.find(symbol);
    if (it == order_books_.end()) {
        return;
    }
    
    auto& book = it->second;
    
    // Simple order book update - replace top of book
    // In reality, you'd need full L2 data to properly reconstruct
    book->cancelOrder(1);  // Remove old best bid
    book->cancelOrder(2);  // Remove old best ask
    
    // Add new best bid and ask
    auto bid_order = std::make_shared<Order>(
        1, Side::BUY, quote.bid_price, quote.bid_size,
        quote.exchange_timestamp
    );
    auto ask_order = std::make_shared<Order>(
        2, Side::SELL, quote.ask_price, quote.ask_size,
        quote.exchange_timestamp
    );
    
    book->addOrder(bid_order);
    book->addOrder(ask_order);
}

void SimulatedMarketDataFeed::processMessage(const MarketDataMessage& message) {
    // For simulated feed, we've already processed in replayLoop
    // Just forward to listeners
    
    if (message.type == MarketDataType::TRADE) {
        // Reconstruct TradeData from message
        // In real implementation, would store full data in message
        TradeData trade;
        trade.symbol = message.symbol;
        trade.exchange_timestamp = message.timestamp;
        trade.local_timestamp = std::chrono::nanoseconds(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        
        notifyTrade(trade);
        
    } else if (message.type == MarketDataType::QUOTE) {
        // Get current order book snapshot
        auto it = order_books_.find(message.symbol);
        if (it != order_books_.end()) {
            auto snapshot = it->second->getSnapshot();
            notifyOrderBookUpdate(message.symbol, snapshot);
        }
    }
}

// MarketDataRecorder implementation
MarketDataRecorder::MarketDataRecorder(const std::string& output_directory,
                                     bool use_compression)
    : output_directory_(output_directory), 
      use_compression_(use_compression),
      buffer_size_(1024 * 1024) {  // 1MB buffer
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_directory);
    
    // Reserve buffer space
    if (use_compression_) {
        write_buffer_.reserve(buffer_size_);
    }
}

MarketDataRecorder::~MarketDataRecorder() {
    stopRecording();
}

void MarketDataRecorder::startRecording() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    // Open files with timestamp in filename
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    std::string base_name = output_directory_ + "/" + timestamp.str();
    
    trade_file_.open(base_name + "_trades.dat", std::ios::binary);
    quote_file_.open(base_name + "_quotes.dat", std::ios::binary);
    book_file_.open(base_name + "_books.dat", std::ios::binary);
    
    if (!trade_file_ || !quote_file_ || !book_file_) {
        throw std::runtime_error("Failed to open output files");
    }
}

void MarketDataRecorder::stopRecording() {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    flush();
    
    trade_file_.close();
    quote_file_.close();
    book_file_.close();
}

void MarketDataRecorder::flush() {
    // Flush all streams
    if (trade_file_.is_open()) trade_file_.flush();
    if (quote_file_.is_open()) quote_file_.flush();
    if (book_file_.is_open()) book_file_.flush();
}

void MarketDataRecorder::onQuote(const QuoteData& quote) {
    writeQuoteRecord(quote);
}

void MarketDataRecorder::onTrade(const TradeData& trade) {
    writeTradeRecord(trade);
}

void MarketDataRecorder::onOrderBookUpdate(const std::string& symbol,
                                          const OrderBook::Snapshot& snapshot) {
    writeBookSnapshot(symbol, snapshot);
}

void MarketDataRecorder::onTradingStatus(const std::string& symbol,
                                        const std::string& status) {
    // Could write to a separate status file
}

void MarketDataRecorder::onError(const std::string& error) {
    std::cerr << "Market data error: " << error << std::endl;
}

void MarketDataRecorder::writeTradeRecord(const TradeData& trade) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (!trade_file_.is_open()) return;
    
    // Binary format for efficiency
    // Format: [symbol_len][symbol][timestamp][price][size][side][conditions_count][conditions...]
    
    uint8_t symbol_len = trade.symbol.length();
    trade_file_.write(reinterpret_cast<const char*>(&symbol_len), sizeof(symbol_len));
    trade_file_.write(trade.symbol.c_str(), symbol_len);
    
    uint64_t timestamp = trade.exchange_timestamp.count();
    trade_file_.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    
    trade_file_.write(reinterpret_cast<const char*>(&trade.price), sizeof(trade.price));
    trade_file_.write(reinterpret_cast<const char*>(&trade.size), sizeof(trade.size));
    
    uint8_t side = (trade.side == Side::BUY) ? 1 : 0;
    trade_file_.write(reinterpret_cast<const char*>(&side), sizeof(side));
    
    uint16_t conditions_count = trade.conditions.size();
    trade_file_.write(reinterpret_cast<const char*>(&conditions_count), sizeof(conditions_count));
    
    for (const auto& condition : trade.conditions) {
        uint8_t cond_len = condition.length();
        trade_file_.write(reinterpret_cast<const char*>(&cond_len), sizeof(cond_len));
        trade_file_.write(condition.c_str(), cond_len);
    }
    
    records_written_++;
}

void MarketDataRecorder::writeQuoteRecord(const QuoteData& quote) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (!quote_file_.is_open()) return;
    
    // Binary format
    // Format: [symbol_len][symbol][timestamp][bid_price][bid_size][ask_price][ask_size]
    
    uint8_t symbol_len = quote.symbol.length();
    quote_file_.write(reinterpret_cast<const char*>(&symbol_len), sizeof(symbol_len));
    quote_file_.write(quote.symbol.c_str(), symbol_len);
    
    uint64_t timestamp = quote.exchange_timestamp.count();
    quote_file_.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    
    quote_file_.write(reinterpret_cast<const char*>(&quote.bid_price), sizeof(quote.bid_price));
    quote_file_.write(reinterpret_cast<const char*>(&quote.bid_size), sizeof(quote.bid_size));
    quote_file_.write(reinterpret_cast<const char*>(&quote.ask_price), sizeof(quote.ask_price));
    quote_file_.write(reinterpret_cast<const char*>(&quote.ask_size), sizeof(quote.ask_size));
    
    records_written_++;
}

void MarketDataRecorder::writeBookSnapshot(const std::string& symbol,
                                         const OrderBook::Snapshot& snapshot) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    
    if (!book_file_.is_open()) return;
    
    // Binary format
    // Format: [symbol_len][symbol][timestamp][bid_count][bids...][ask_count][asks...]
    
    uint8_t symbol_len = symbol.length();
    book_file_.write(reinterpret_cast<const char*>(&symbol_len), sizeof(symbol_len));
    book_file_.write(symbol.c_str(), symbol_len);
    
    uint64_t timestamp = snapshot.timestamp.count();
    book_file_.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    
    // Write bids
    uint16_t bid_count = snapshot.bids.size();
    book_file_.write(reinterpret_cast<const char*>(&bid_count), sizeof(bid_count));
    
    for (const auto& [price, size] : snapshot.bids) {
        book_file_.write(reinterpret_cast<const char*>(&price), sizeof(price));
        book_file_.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    
    // Write asks
    uint16_t ask_count = snapshot.asks.size();
    book_file_.write(reinterpret_cast<const char*>(&ask_count), sizeof(ask_count));
    
    for (const auto& [price, size] : snapshot.asks) {
        book_file_.write(reinterpret_cast<const char*>(&price), sizeof(price));
        book_file_.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    
    records_written_++;
}

} // namespace hft