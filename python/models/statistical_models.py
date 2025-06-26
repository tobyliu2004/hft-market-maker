"""
Statistical models for market microstructure analysis and prediction.
These models integrate with the C++ engine via Python bindings.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import het_arch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MicrostructureFeatures:
    """Extract microstructure features from order book data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_order_flow_imbalance(self, bid_sizes, ask_sizes, levels=5):
        """Calculate order flow imbalance across multiple levels."""
        bid_total = np.sum(bid_sizes[:levels])
        ask_total = np.sum(ask_sizes[:levels])
        
        if bid_total + ask_total == 0:
            return 0
            
        return (bid_total - ask_total) / (bid_total + ask_total)
    
    def calculate_book_pressure(self, bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price):
        """Calculate book pressure weighted by distance from mid."""
        bid_pressure = 0
        ask_pressure = 0
        
        for price, size in zip(bid_prices, bid_sizes):
            distance = abs(mid_price - price)
            if distance > 0:
                bid_pressure += size / distance
                
        for price, size in zip(ask_prices, ask_sizes):
            distance = abs(price - mid_price)
            if distance > 0:
                ask_pressure += size / distance
                
        return bid_pressure - ask_pressure
    
    def calculate_microprice(self, best_bid, best_ask, bid_size, ask_size):
        """Calculate microprice using size-weighted mid."""
        if bid_size + ask_size == 0:
            return (best_bid + best_ask) / 2
            
        return (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    
    def calculate_kyle_lambda(self, trades_df, time_window='1min'):
        """Estimate Kyle's lambda (price impact coefficient)."""
        # Aggregate net order flow and price changes
        resampled = trades_df.resample(time_window).agg({
            'price': ['first', 'last'],
            'size': lambda x: x[trades_df.loc[x.index, 'side'] == 'buy'].sum() - 
                             x[trades_df.loc[x.index, 'side'] == 'sell'].sum()
        })
        
        price_changes = resampled['price']['last'] - resampled['price']['first']
        net_flow = resampled['size']['<lambda>']
        
        # Remove NaN values
        mask = ~(np.isnan(price_changes) | np.isnan(net_flow))
        price_changes = price_changes[mask]
        net_flow = net_flow[mask]
        
        if len(price_changes) < 10:
            return 0
            
        # Linear regression
        slope, _, _, _, _ = stats.linregress(net_flow, price_changes)
        return slope
    
    def calculate_roll_spread(self, prices):
        """Calculate Roll's implied spread from price changes."""
        price_changes = np.diff(prices)
        
        if len(price_changes) < 2:
            return 0
            
        autocovariance = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        if autocovariance >= 0:
            return 0
            
        return 2 * np.sqrt(-autocovariance)
    
    def extract_features(self, order_book_snapshot, trades_df=None):
        """Extract all microstructure features from order book snapshot."""
        features = {}
        
        # Basic features
        features['spread'] = order_book_snapshot['best_ask'] - order_book_snapshot['best_bid']
        features['mid_price'] = (order_book_snapshot['best_ask'] + order_book_snapshot['best_bid']) / 2
        features['relative_spread'] = features['spread'] / features['mid_price']
        
        # Order flow imbalance at different levels
        for levels in [1, 3, 5, 10]:
            features[f'ofi_{levels}'] = self.calculate_order_flow_imbalance(
                order_book_snapshot['bid_sizes'],
                order_book_snapshot['ask_sizes'],
                levels
            )
        
        # Book pressure
        features['book_pressure'] = self.calculate_book_pressure(
            order_book_snapshot['bid_prices'],
            order_book_snapshot['bid_sizes'],
            order_book_snapshot['ask_prices'],
            order_book_snapshot['ask_sizes'],
            features['mid_price']
        )
        
        # Microprice
        features['microprice'] = self.calculate_microprice(
            order_book_snapshot['best_bid'],
            order_book_snapshot['best_ask'],
            order_book_snapshot['bid_sizes'][0],
            order_book_snapshot['ask_sizes'][0]
        )
        
        features['microprice_deviation'] = (features['microprice'] - features['mid_price']) / features['mid_price']
        
        # Volume features
        features['total_bid_volume'] = np.sum(order_book_snapshot['bid_sizes'])
        features['total_ask_volume'] = np.sum(order_book_snapshot['ask_sizes'])
        features['volume_imbalance'] = (features['total_bid_volume'] - features['total_ask_volume']) / \
                                      (features['total_bid_volume'] + features['total_ask_volume'])
        
        # Depth features
        features['bid_depth_weighted_price'] = np.average(
            order_book_snapshot['bid_prices'][:10],
            weights=order_book_snapshot['bid_sizes'][:10]
        )
        features['ask_depth_weighted_price'] = np.average(
            order_book_snapshot['ask_prices'][:10],
            weights=order_book_snapshot['ask_sizes'][:10]
        )
        
        # Trade-based features if available
        if trades_df is not None and len(trades_df) > 0:
            features['kyle_lambda'] = self.calculate_kyle_lambda(trades_df)
            features['roll_spread'] = self.calculate_roll_spread(trades_df['price'].values)
            features['trade_intensity'] = len(trades_df) / (trades_df.index[-1] - trades_df.index[0]).total_seconds()
        
        return features


class AdverseSelectionDetector:
    """Detect toxic order flow using statistical models."""
    
    def __init__(self, lookback_window=1000):
        self.lookback_window = lookback_window
        self.price_impact_history = []
        self.toxicity_threshold = 2.0  # Standard deviations
        
    def calculate_price_impact(self, pre_trade_mid, post_trade_mid, trade_side, trade_size):
        """Calculate realized price impact of a trade."""
        if trade_side == 'buy':
            impact = (post_trade_mid - pre_trade_mid) / pre_trade_mid
        else:
            impact = (pre_trade_mid - post_trade_mid) / pre_trade_mid
            
        # Normalize by trade size (in standard deviations of typical trade size)
        return impact * np.sqrt(trade_size)
    
    def update_impact_history(self, impact):
        """Update rolling history of price impacts."""
        self.price_impact_history.append(impact)
        
        if len(self.price_impact_history) > self.lookback_window:
            self.price_impact_history.pop(0)
    
    def detect_toxic_flow(self, recent_trades, order_book_snapshots):
        """Detect whether recent order flow appears toxic."""
        if len(recent_trades) < 10:
            return 0.0, "Insufficient data"
            
        # Calculate price impacts for recent trades
        impacts = []
        for i, trade in recent_trades.iterrows():
            # Find pre and post trade snapshots
            pre_snapshot = order_book_snapshots[order_book_snapshots.index < trade.name].iloc[-1]
            post_snapshot = order_book_snapshots[order_book_snapshots.index > trade.name].iloc[0]
            
            pre_mid = (pre_snapshot['best_bid'] + pre_snapshot['best_ask']) / 2
            post_mid = (post_snapshot['best_bid'] + post_snapshot['best_ask']) / 2
            
            impact = self.calculate_price_impact(
                pre_mid, post_mid, trade['side'], trade['size']
            )
            impacts.append(impact)
            self.update_impact_history(impact)
        
        # Statistical tests for adverse selection
        recent_mean = np.mean(impacts)
        recent_std = np.std(impacts)
        
        if len(self.price_impact_history) < 100:
            return 0.0, "Building history"
            
        historical_mean = np.mean(self.price_impact_history)
        historical_std = np.std(self.price_impact_history)
        
        # Z-score of recent impacts
        if historical_std > 0:
            z_score = (recent_mean - historical_mean) / historical_std
        else:
            z_score = 0
            
        # Additional tests
        tests_failed = []
        
        # Test 1: Abnormally high average impact
        if abs(z_score) > self.toxicity_threshold:
            tests_failed.append("high_impact")
            
        # Test 2: Consistently one-sided impacts
        same_sign_ratio = sum(1 for x in impacts if x * recent_mean > 0) / len(impacts)
        if same_sign_ratio > 0.8:
            tests_failed.append("directional")
            
        # Test 3: Increasing impact magnitude (momentum)
        first_half_mean = np.mean(impacts[:len(impacts)//2])
        second_half_mean = np.mean(impacts[len(impacts)//2:])
        if abs(second_half_mean) > abs(first_half_mean) * 1.5:
            tests_failed.append("accelerating")
            
        # Calculate toxicity score
        toxicity_score = min(1.0, abs(z_score) / self.toxicity_threshold)
        
        if len(tests_failed) > 0:
            toxicity_score = max(toxicity_score, len(tests_failed) / 3.0)
            
        return toxicity_score, tests_failed


class VolatilityEstimator:
    """Estimate and predict volatility using GARCH and other models."""
    
    def __init__(self):
        self.returns_history = []
        self.volatility_estimates = []
        
    def calculate_realized_volatility(self, prices, window=20):
        """Calculate realized volatility from price series."""
        returns = np.diff(np.log(prices))
        
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252 * 6.5 * 60)  # Annualized
            
        # Rolling window volatility
        volatilities = []
        for i in range(window, len(returns) + 1):
            vol = np.std(returns[i-window:i]) * np.sqrt(252 * 6.5 * 60)
            volatilities.append(vol)
            
        return volatilities[-1] if volatilities else 0
    
    def calculate_parkinson_volatility(self, high_prices, low_prices, window=20):
        """Calculate Parkinson volatility using high-low prices."""
        if len(high_prices) < window:
            return 0
            
        log_hl = np.log(high_prices / low_prices)
        factor = 1 / (4 * np.log(2))
        
        volatilities = []
        for i in range(window, len(log_hl) + 1):
            vol = np.sqrt(factor * np.mean(log_hl[i-window:i]**2)) * np.sqrt(252)
            volatilities.append(vol)
            
        return volatilities[-1] if volatilities else 0
    
    def estimate_garch_volatility(self, returns, p=1, q=1):
        """Estimate volatility using GARCH(p,q) model."""
        from arch import arch_model
        
        # Scale returns to percentage
        scaled_returns = returns * 100
        
        try:
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q)
            result = model.fit(disp='off')
            
            # Get conditional volatility forecast
            forecast = result.forecast(horizon=1)
            next_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
            
            return next_vol * np.sqrt(252 * 6.5 * 60)  # Annualized
            
        except:
            # Fallback to simple volatility
            return np.std(returns) * np.sqrt(252 * 6.5 * 60)
    
    def calculate_yang_zhang_volatility(self, open_prices, high_prices, low_prices, close_prices, window=20):
        """Calculate Yang-Zhang volatility estimator."""
        if len(open_prices) < window + 1:
            return 0
            
        # Overnight volatility
        log_co = np.log(open_prices[1:] / close_prices[:-1])
        overnight_var = np.var(log_co)
        
        # Open-to-close volatility
        log_oc = np.log(close_prices / open_prices)
        oc_var = np.var(log_oc)
        
        # Rogers-Satchell volatility
        log_hl = np.log(high_prices / low_prices)
        log_hc = np.log(high_prices / close_prices)
        log_lc = np.log(low_prices / close_prices)
        rs_var = np.mean(log_hc * log_hl)
        
        # Yang-Zhang combination
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * 252)


class PricePredictionModel:
    """Machine learning models for short-term price prediction."""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'arima': None
        }
        self.feature_extractor = MicrostructureFeatures()
        self.is_trained = False
        
    def prepare_features(self, order_book_snapshots, trades_df, lookback=50):
        """Prepare features for ML models."""
        features_list = []
        targets = []
        
        for i in range(lookback, len(order_book_snapshots) - 1):
            # Extract features from current snapshot
            snapshot = order_book_snapshots.iloc[i]
            
            # Get recent trades
            snapshot_time = order_book_snapshots.index[i]
            recent_trades = trades_df[
                (trades_df.index > snapshot_time - pd.Timedelta(seconds=10)) &
                (trades_df.index <= snapshot_time)
            ]
            
            features = self.feature_extractor.extract_features(snapshot, recent_trades)
            
            # Add lagged features
            for lag in [1, 5, 10]:
                if i - lag >= 0:
                    lag_snapshot = order_book_snapshots.iloc[i - lag]
                    lag_mid = (lag_snapshot['best_bid'] + lag_snapshot['best_ask']) / 2
                    features[f'return_lag_{lag}'] = (features['mid_price'] - lag_mid) / lag_mid
                    
            features_list.append(features)
            
            # Target: next period return
            next_snapshot = order_book_snapshots.iloc[i + 1]
            next_mid = (next_snapshot['best_bid'] + next_snapshot['best_ask']) / 2
            target = (next_mid - features['mid_price']) / features['mid_price']
            targets.append(target)
            
        return pd.DataFrame(features_list), np.array(targets)
    
    def train(self, order_book_snapshots, trades_df):
        """Train prediction models."""
        print("Preparing features...")
        X, y = self.prepare_features(order_book_snapshots, trades_df)
        
        # Remove any NaN values
        mask = ~(X.isna().any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            print("Insufficient data for training")
            return
            
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_extractor.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.models['rf'].fit(X_train_scaled, y_train)
        rf_score = self.models['rf'].score(X_test_scaled, y_test)
        print(f"Random Forest R²: {rf_score:.4f}")
        
        # Train ARIMA on returns
        print("Training ARIMA...")
        try:
            self.models['arima'] = ARIMA(y_train, order=(2, 0, 2))
            arima_result = self.models['arima'].fit()
            arima_forecast = arima_result.forecast(steps=len(y_test))
            arima_mse = np.mean((arima_forecast - y_test)**2)
            print(f"ARIMA MSE: {arima_mse:.6f}")
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            
        self.is_trained = True
        
        # Feature importance
        if hasattr(self.models['rf'], 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': self.models['rf'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(importances.head(10))
            
        return rf_score
    
    def predict(self, current_snapshot, recent_trades):
        """Make price prediction for next period."""
        if not self.is_trained:
            return 0.0, 0.0  # No prediction available
            
        # Extract features
        features = self.feature_extractor.extract_features(current_snapshot, recent_trades)
        features_df = pd.DataFrame([features])
        
        # Scale features
        try:
            features_scaled = self.feature_extractor.scaler.transform(features_df)
        except:
            return 0.0, 0.0
            
        # Make predictions
        rf_pred = self.models['rf'].predict(features_scaled)[0]
        
        # Ensemble prediction (could add more models)
        ensemble_pred = rf_pred
        
        # Estimate prediction confidence based on feature values
        # High volatility or unusual feature values -> lower confidence
        feature_z_scores = np.abs(features_scaled[0])
        confidence = 1.0 / (1.0 + np.mean(feature_z_scores))
        
        return ensemble_pred, confidence


class CointegrationAnalyzer:
    """Analyze cointegration relationships for pairs trading."""
    
    def __init__(self):
        self.pairs = {}
        
    def test_stationarity(self, series, significance=0.05):
        """Test if a series is stationary using ADF test."""
        result = adfuller(series, autolag='AIC')
        return result[1] < significance  # p-value < significance level
    
    def find_cointegrated_pairs(self, price_data, significance=0.05):
        """Find cointegrated pairs from price data."""
        symbols = price_data.columns
        cointegrated_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Test for cointegration
                score, pvalue, _ = coint(price_data[symbol1], price_data[symbol2])
                
                if pvalue < significance:
                    # Calculate hedge ratio
                    hedge_ratio = self.calculate_hedge_ratio(
                        price_data[symbol1].values,
                        price_data[symbol2].values
                    )
                    
                    # Calculate spread statistics
                    spread = price_data[symbol1] - hedge_ratio * price_data[symbol2]
                    
                    if self.test_stationarity(spread):
                        cointegrated_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'hedge_ratio': hedge_ratio,
                            'pvalue': pvalue,
                            'spread_mean': spread.mean(),
                            'spread_std': spread.std(),
                            'half_life': self.calculate_half_life(spread)
                        })
                        
        return pd.DataFrame(cointegrated_pairs)
    
    def calculate_hedge_ratio(self, prices1, prices2):
        """Calculate optimal hedge ratio using OLS."""
        # Simple linear regression
        X = prices2.reshape(-1, 1)
        y = prices1
        
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS solution
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        return beta[1]  # Slope coefficient
    
    def calculate_half_life(self, spread):
        """Calculate mean reversion half-life of spread."""
        # AR(1) model: spread_t = alpha + beta * spread_{t-1} + epsilon
        lagged_spread = spread[:-1]
        current_spread = spread[1:]
        
        # Remove NaN values
        mask = ~(np.isnan(lagged_spread) | np.isnan(current_spread))
        lagged_spread = lagged_spread[mask]
        current_spread = current_spread[mask]
        
        if len(lagged_spread) < 10:
            return np.inf
            
        # Linear regression
        X = np.column_stack([np.ones(len(lagged_spread)), lagged_spread])
        beta = np.linalg.lstsq(X, current_spread, rcond=None)[0]
        
        # Half-life = -log(2) / log(beta[1])
        if beta[1] > 0 and beta[1] < 1:
            half_life = -np.log(2) / np.log(beta[1])
            return half_life
        else:
            return np.inf
    
    def generate_trading_signals(self, symbol1_price, symbol2_price, pair_params, z_entry=2.0, z_exit=0.5):
        """Generate trading signals for a cointegrated pair."""
        # Calculate current spread
        spread = symbol1_price - pair_params['hedge_ratio'] * symbol2_price
        
        # Calculate z-score
        z_score = (spread - pair_params['spread_mean']) / pair_params['spread_std']
        
        # Generate signals
        if z_score > z_entry:
            # Spread too high: short symbol1, long symbol2
            return {'signal': 'short_spread', 'z_score': z_score}
        elif z_score < -z_entry:
            # Spread too low: long symbol1, short symbol2
            return {'signal': 'long_spread', 'z_score': z_score}
        elif abs(z_score) < z_exit:
            # Close position
            return {'signal': 'close', 'z_score': z_score}
        else:
            # Hold position
            return {'signal': 'hold', 'z_score': z_score}


# Integration functions for C++ binding
def create_feature_extractor():
    """Create feature extractor instance for C++ integration."""
    return MicrostructureFeatures()

def create_adverse_selection_detector(lookback=1000):
    """Create adverse selection detector for C++ integration."""
    return AdverseSelectionDetector(lookback)

def create_volatility_estimator():
    """Create volatility estimator for C++ integration."""
    return VolatilityEstimator()

def create_price_predictor():
    """Create price prediction model for C++ integration."""
    return PricePredictionModel()

def create_cointegration_analyzer():
    """Create cointegration analyzer for C++ integration."""
    return CointegrationAnalyzer()


# Example usage and testing
if __name__ == "__main__":
    # Test microstructure features
    print("Testing Microstructure Feature Extraction...")
    
    # Mock order book snapshot
    snapshot = {
        'best_bid': 100.00,
        'best_ask': 100.02,
        'bid_prices': np.array([100.00, 99.99, 99.98, 99.97, 99.96]),
        'bid_sizes': np.array([1000, 2000, 3000, 1500, 2500]),
        'ask_prices': np.array([100.02, 100.03, 100.04, 100.05, 100.06]),
        'ask_sizes': np.array([1200, 2200, 2800, 1800, 2000])
    }
    
    extractor = MicrostructureFeatures()
    features = extractor.extract_features(snapshot)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.6f}")
    
    # Test volatility estimation
    print("\n\nTesting Volatility Estimation...")
    
    # Generate mock price data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, 1000)))
    
    vol_estimator = VolatilityEstimator()
    realized_vol = vol_estimator.calculate_realized_volatility(prices)
    
    print(f"Realized Volatility: {realized_vol:.2%}")
    
    # Test GARCH volatility
    returns = np.diff(np.log(prices))
    garch_vol = vol_estimator.estimate_garch_volatility(returns)
    print(f"GARCH Volatility: {garch_vol:.2%}")