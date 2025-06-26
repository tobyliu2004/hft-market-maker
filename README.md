# High-Frequency Market Making Simulator

A production-grade C++ market making system with Level 2 order book dynamics, FIX protocol support, and sophisticated trading strategies integrated with Python-based statistical models.

## Features

### Core Components

- **Order Book Engine**: O(1) order insertion/cancellation with memory-mapped persistence
- **FIX Protocol Handler**: QuickFIX implementation for market data and order management
- **Market Making Strategies**: 
  - Avellaneda-Stoikov optimal market making
  - Alpha-based market making with microstructure signals
  - Statistical arbitrage and pairs trading
- **Risk Management**: Dynamic position limits, VaR calculations, and compliance checks
- **Data Pipeline**: Real-time feeds from Polygon.io and historical NYSE TAQ data

### Technical Highlights

- **Performance**: Handles 1M+ order updates per second with microsecond latency
- **Statistical Models**: Python/C++ hybrid for ML predictions and adverse selection detection
- **Risk Controls**: Integrated margin, compliance, and capital allocation systems
- **Backtesting**: Full historical simulation with realistic market impact modeling

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Market Data    │     │   Order Book     │     │   Strategies    │
│  (Polygon/TAQ)  │────▶│   Engine (C++)   │────▶│  (Avellaneda)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FIX Protocol   │     │ Statistical      │     │ Risk Manager    │
│    Handler      │◀────│ Models (Python)  │────▶│   (Dynamic)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Building

### Prerequisites

- C++20 compatible compiler
- CMake 3.16+
- Boost 1.70+
- Python 3.8+ with NumPy, SciPy, statsmodels
- QuickFIX 1.15+
- WebSocket++ 0.8+
- nlohmann/json 3.2+

### Build Instructions

```bash
mkdir build && cd build
cmake ..
make -j8
```

### Running Tests

```bash
make test
```

## Configuration

Edit `config/config.json` to configure:

- FIX connection settings
- Market data sources and API keys
- Trading strategy parameters
- Risk limits and position sizing
- Execution settings

Example configuration:

```json
{
    "strategy": {
        "type": "avellaneda_stoikov",
        "risk_aversion": 0.1,
        "order_arrival_rate": 10.0,
        "time_horizon": 300.0
    },
    "risk_limits": {
        "max_position_value": 100000.0,
        "max_daily_loss": 10000.0,
        "max_drawdown": 0.05
    }
}
```

## Usage

### Live Trading

```bash
./bin/market_maker config/config.json
```

### Backtesting

```bash
./bin/backtest --config config/backtest.json --date 2024-01-15
```

### Performance Monitoring

The system exposes Prometheus metrics on port 9090:

- Order latency percentiles
- Fill rates and spreads captured
- P&L and risk metrics
- System performance counters

## Strategy Details

### Avellaneda-Stoikov Model

Implements the optimal market-making framework with:
- Reservation price calculation based on inventory risk
- Dynamic spread adjustment using volatility estimates
- Inventory penalty functions to manage position risk

### Microstructure Alpha Signals

- **Microprice**: Size-weighted mid-price for better valuation
- **Order Flow Imbalance**: Directional pressure indicators
- **Queue Position**: Smart order placement optimization
- **Kyle's Lambda**: Price impact estimation

### Risk Management

- **Position Limits**: Hard limits with buffer zones
- **Dynamic Hedging**: Automatic delta/gamma hedging
- **Stress Testing**: Configurable market scenarios
- **Compliance**: Regulatory checks and audit trails

## Python Integration

Statistical models are implemented in Python for flexibility:

```python
# Adverse selection detection
detector = AdverseSelectionDetector()
toxicity_score = detector.detect_toxic_flow(trades, orderbook)

# Volatility forecasting
estimator = VolatilityEstimator()
garch_vol = estimator.estimate_garch_volatility(returns)

# Price prediction
predictor = PricePredictionModel()
prediction = predictor.predict(orderbook_features)
```

## Performance Metrics

Measured on Intel Xeon E5-2698 v4:

- Order book update latency: < 500ns
- FIX message processing: < 10μs
- Strategy calculation: < 50μs
- End-to-end tick-to-trade: < 100μs

## Safety and Compliance

- Kill switches for emergency stops
- Position and loss limits
- Anti-manipulation checks
- Full audit logging
- Regulatory reporting support

## Future Enhancements

- [ ] GPU acceleration for ML models
- [ ] Additional exchange connectivity
- [ ] Options market making
- [ ] Cryptocurrency support
- [ ] Cloud deployment templates

## Contributing

Please read CONTRIBUTING.md for development guidelines.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments carries risk. Always test thoroughly in simulation before any live deployment.