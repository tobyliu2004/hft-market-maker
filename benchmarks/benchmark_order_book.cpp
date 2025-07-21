#include "core/order_book.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace hft;

class OrderBookBenchmark : public benchmark::Fixture {
protected:
    std::shared_ptr<OrderBook> book;
    std::vector<std::shared_ptr<Order>> orders;
    std::mt19937 rng{42};
    std::uniform_real_distribution<double> price_dist{99.0, 101.0};
    std::uniform_real_distribution<double> size_dist{100.0, 1000.0};
    std::uniform_int_distribution<int> side_dist{0, 1};
    
    void SetUp(const ::benchmark::State& state) override {
        book = std::make_shared<OrderBook>();
        orders.clear();
        
        // Pre-generate orders
        int num_orders = state.range(0);
        for (int i = 0; i < num_orders; ++i) {
            auto order = std::make_shared<Order>(
                i + 1,
                side_dist(rng) == 0 ? Side::BUY : Side::SELL,
                price_dist(rng),
                size_dist(rng),
                std::chrono::high_resolution_clock::now()
            );
            orders.push_back(order);
        }
    }
};

BENCHMARK_DEFINE_F(OrderBookBenchmark, AddOrder)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        book = std::make_shared<OrderBook>();
        state.ResumeTiming();
        
        for (const auto& order : orders) {
            book->addOrder(order);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * orders.size());
}

BENCHMARK_DEFINE_F(OrderBookBenchmark, CancelOrder)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        // Reset book and add all orders
        book = std::make_shared<OrderBook>();
        for (const auto& order : orders) {
            book->addOrder(order);
        }
        state.ResumeTiming();
        
        // Cancel all orders
        for (const auto& order : orders) {
            book->cancelOrder(order->order_id);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * orders.size());
}

BENCHMARK_DEFINE_F(OrderBookBenchmark, GetBestBidAsk)(benchmark::State& state) {
    // Populate book once
    for (const auto& order : orders) {
        book->addOrder(order);
    }
    
    for (auto _ : state) {
        auto [best_bid, best_ask] = book->getBestBidAsk();
        benchmark::DoNotOptimize(best_bid);
        benchmark::DoNotOptimize(best_ask);
    }
}

BENCHMARK_DEFINE_F(OrderBookBenchmark, GetSnapshot)(benchmark::State& state) {
    // Populate book once
    for (const auto& order : orders) {
        book->addOrder(order);
    }
    
    int depth = state.range(1);
    
    for (auto _ : state) {
        auto snapshot = book->getSnapshot(depth);
        benchmark::DoNotOptimize(snapshot);
    }
}

BENCHMARK_DEFINE_F(OrderBookBenchmark, MatchOrders)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        book = std::make_shared<OrderBook>();
        
        // Add half as limit orders
        for (size_t i = 0; i < orders.size() / 2; ++i) {
            book->addOrder(orders[i]);
        }
        state.ResumeTiming();
        
        // Add remaining as market orders (will match)
        for (size_t i = orders.size() / 2; i < orders.size(); ++i) {
            auto market_order = std::make_shared<Order>(
                orders[i]->order_id,
                orders[i]->side == Side::BUY ? Side::SELL : Side::BUY,
                orders[i]->price,
                orders[i]->quantity,
                orders[i]->timestamp
            );
            book->addOrder(market_order);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * orders.size() / 2);
}

// Benchmark mixed operations (realistic scenario)
static void BM_OrderBookMixedOperations(benchmark::State& state) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> op_dist(0, 3);
    std::uniform_real_distribution<double> price_dist(99.0, 101.0);
    std::uniform_real_distribution<double> size_dist(100.0, 1000.0);
    std::uniform_int_distribution<int> side_dist(0, 1);
    
    auto book = std::make_shared<OrderBook>();
    std::vector<OrderID> active_orders;
    OrderID next_order_id = 1;
    
    // Pre-populate book
    for (int i = 0; i < 1000; ++i) {
        auto order = std::make_shared<Order>(
            next_order_id++,
            side_dist(rng) == 0 ? Side::BUY : Side::SELL,
            price_dist(rng),
            size_dist(rng),
            std::chrono::high_resolution_clock::now()
        );
        book->addOrder(order);
        active_orders.push_back(order->order_id);
    }
    
    for (auto _ : state) {
        int operation = op_dist(rng);
        
        switch (operation) {
            case 0: {  // Add order
                auto order = std::make_shared<Order>(
                    next_order_id++,
                    side_dist(rng) == 0 ? Side::BUY : Side::SELL,
                    price_dist(rng),
                    size_dist(rng),
                    std::chrono::high_resolution_clock::now()
                );
                book->addOrder(order);
                active_orders.push_back(order->order_id);
                break;
            }
            case 1: {  // Cancel order
                if (!active_orders.empty()) {
                    std::uniform_int_distribution<size_t> idx_dist(0, active_orders.size() - 1);
                    size_t idx = idx_dist(rng);
                    book->cancelOrder(active_orders[idx]);
                    active_orders.erase(active_orders.begin() + idx);
                }
                break;
            }
            case 2: {  // Get best bid/ask
                auto [best_bid, best_ask] = book->getBestBidAsk();
                benchmark::DoNotOptimize(best_bid);
                benchmark::DoNotOptimize(best_ask);
                break;
            }
            case 3: {  // Get snapshot
                auto snapshot = book->getSnapshot(10);
                benchmark::DoNotOptimize(snapshot);
                break;
            }
        }
    }
}

// Register benchmarks
BENCHMARK_REGISTER_F(OrderBookBenchmark, AddOrder)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(OrderBookBenchmark, CancelOrder)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(OrderBookBenchmark, GetBestBidAsk)
    ->Arg(1000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(OrderBookBenchmark, GetSnapshot)
    ->ArgsProduct({
        {1000, 10000},  // Number of orders
        {5, 10, 20}     // Depth
    })
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(OrderBookBenchmark, MatchOrders)
    ->RangeMultiplier(10)
    ->Range(10, 1000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_OrderBookMixedOperations)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(1000000);

// Memory usage benchmark
static void BM_OrderBookMemoryUsage(benchmark::State& state) {
    int num_orders = state.range(0);
    
    for (auto _ : state) {
        auto book = std::make_shared<OrderBook>();
        
        for (int i = 0; i < num_orders; ++i) {
            auto order = std::make_shared<Order>(
                i + 1,
                i % 2 == 0 ? Side::BUY : Side::SELL,
                100.0 + (i % 100) * 0.01,
                100.0 + (i % 10) * 100,
                std::chrono::high_resolution_clock::now()
            );
            book->addOrder(order);
        }
        
        benchmark::DoNotOptimize(book);
    }
    
    state.SetBytesProcessed(state.iterations() * num_orders * sizeof(Order));
}

BENCHMARK(BM_OrderBookMemoryUsage)
    ->RangeMultiplier(10)
    ->Range(100, 100000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();