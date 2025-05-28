import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt
from arch import arch_model

class PowerMarketStatistics:
    def __init__(self, price_data, volume_data, fundamental_data):
        """
        Initialize with market data
        :param price_data: DataFrame with price time series
        :param volume_data: DataFrame with trading volumes
        :param fundamental_data: DataFrame with fundamental drivers
        """
        self.prices = price_data
        self.volumes = volume_data
        self.fundamentals = fundamental_data
        
    def compute_basic_stats(self, window='1D'):
        """Compute rolling statistical properties of prices"""
        stats = pd.DataFrame(index=self.prices.index)
        stats['mean'] = self.prices.rolling(window).mean()
        stats['std'] = self.prices.rolling(window).std()
        stats['skew'] = self.prices.rolling(window).apply(skew)
        stats['kurtosis'] = self.prices.rolling(window).apply(kurtosis)
        stats['median'] = self.prices.rolling(window).median()
        stats['iqr'] = self.prices.rolling(window).quantile(0.75) - self.prices.rolling(window).quantile(0.25)
        return stats.dropna()
    
    def analyze_jumps(self, threshold=3):
        """Identify price jumps using standard deviation thresholds"""
        returns = np.log(self.prices).diff()
        jump_indicator = np.abs(returns) > threshold * returns.std()
        return returns[jump_indicator]
    
    def estimate_volatility(self, model='GARCH', p=1, q=1):
        """Estimate time-varying volatility"""
        returns = np.log(self.prices).diff().dropna()
        
        if model == 'GARCH':
            garch = arch_model(returns, vol='Garch', p=p, q=q)
            res = garch.fit(disp='off')
            return res.conditional_volatility
        elif model == 'EWMA':
            return returns.ewm(span=20).std()
        else:
            return returns.rolling('1H').std()
    
    def compute_cross_correlations(self, other_series, max_lags=12):
        """Cross-correlation analysis with fundamentals"""
        ccf = []
        for lag in range(-max_lags, max_lags+1):
            if lag < 0:
                ccf.append(self.prices.corr(other_series.shift(-lag)))
            else:
                ccf.append(self.prices.shift(lag).corr(other_series))
        return ccf

class FundamentalModel:
    def __init__(self, price_data, fundamental_data):
        self.X = fundamental_data
        self.y = price_data
        
    def train_ols_model(self):
        """Ordinary Least Squares regression model"""
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()
        return model
        
    def train_random_forest(self, n_estimators=100):
        """Machine learning approach to price prediction"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(X_train, y_train)
        return rf, rf.score(X_test, y_test)
    
    def compute_fundamental_zscore(self, lookback=30):
        """Standardized measure of fundamental drivers"""
        rolling_mean = self.X.rolling(lookback).mean()
        rolling_std = self.X.rolling(lookback).std()
        return (self.X - rolling_mean) / rolling_std

class PowerTradingStrategies:
    def __init__(self, price_data, stats_data, fundamental_model):
        self.prices = price_data
        self.stats = stats_data
        self.model = fundamental_model
        
    def mean_reversion_strategy(self, entry_z=1.5, exit_z=0.5):
        """
        Statistical arbitrage based on z-score normalization
        :param entry_z: z-score threshold for entering trades
        :param exit_z: z-score threshold for exiting trades
        """
        signals = pd.DataFrame(index=self.prices.index)
        signals['z_score'] = (self.prices - self.stats['mean']) / self.stats['std']
        
        # Generate signals
        signals['position'] = 0
        signals.loc[signals['z_score'] > entry_z, 'position'] = -1  # Overpriced - sell
        signals.loc[signals['z_score'] < -entry_z, 'position'] = 1   # Underpriced - buy
        signals.loc[abs(signals['z_score']) < exit_z, 'position'] = 0  # Exit
        
        return signals
    
    def volatility_breakout_strategy(self, volatility_window='4H', multiplier=2):
        """
        Breakout strategy based on volatility bands
        :param volatility_window: lookback window for volatility calculation
        :param multiplier: width of volatility bands
        """
        signals = pd.DataFrame(index=self.prices.index)
        rolling_std = self.prices.rolling(volatility_window).std()
        
        # Create bands
        signals['upper_band'] = self.prices.rolling(volatility_window).mean() + multiplier * rolling_std
        signals['lower_band'] = self.prices.rolling(volatility_window).mean() - multiplier * rolling_std
        
        # Generate signals
        signals['position'] = 0
        signals.loc[self.prices > signals['upper_band'], 'position'] = -1  # Sell at high volatility
        signals.loc[self.prices < signals['lower_band'], 'position'] = 1    # Buy at low volatility
        
        return signals
    
    def fundamental_mispricing_strategy(self, threshold=1.5):
        """
        Trade based on deviations from fundamental value
        :param threshold: standard deviation threshold for action
        """
        pred_prices = self.model.train_ols_model().predict()
        residuals = self.prices - pred_prices
        z_scores = (residuals - residuals.mean()) / residuals.std()
        
        signals = pd.DataFrame(index=self.prices.index)
        signals['position'] = 0
        signals.loc[z_scores > threshold, 'position'] = -1  # Overpriced
        signals.loc[z_scores < -threshold, 'position'] = 1   # Underpriced
        
        return signals
    
    def intraday_seasonality_strategy(self, seasonal_pattern):
        """
        Exploit recurring intraday patterns
        :param seasonal_pattern: pre-calculated seasonal component
        """
        deviation = self.prices - seasonal_pattern
        std_dev = deviation.rolling('7D').std()
        
        signals = pd.DataFrame(index=self.prices.index)
        signals['position'] = np.where(
            deviation > 1.5 * std_dev, -1, np.where(
                deviation < -1.5 * std_dev, 1, 0))
        
        return signals

class BacktestingEngine:
    def __init__(self, prices, signals, transaction_cost=0.0001):
        self.prices = prices
        self.signals = signals
        self.tc = transaction_cost
        
    def run_backtest(self):
        portfolio = pd.DataFrame(index=self.prices.index)
        portfolio['price'] = self.prices
        portfolio['position'] = self.signals['position']
        
        # Calculate PnL
        portfolio['returns'] = portfolio['price'].pct_change()
        portfolio['strategy_returns'] = portfolio['position'].shift(1) * portfolio['returns']
        
        # Account for transaction costs
        trades = portfolio['position'].diff().abs()
        portfolio['strategy_returns'] -= trades * self.tc
        
        # Cumulative performance
        portfolio['cumulative_strategy'] = (1 + portfolio['strategy_returns']).cumprod()
        portfolio['cumulative_buy_hold'] = (1 + portfolio['returns']).cumprod()
        
        return portfolio
    
    def compute_performance_metrics(self, portfolio):
        metrics = {}
        returns = portfolio['strategy_returns']
        
        # Basic metrics
        metrics['total_return'] = portfolio['cumulative_strategy'].iloc[-1] - 1
        metrics['annualized_return'] = (1 + metrics['total_return'])**(252/len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Risk metrics
        metrics['max_drawdown'] = (portfolio['cumulative_strategy'].cummax() - 
                                  portfolio['cumulative_strategy']).max()
        metrics['win_rate'] = (returns > 0).mean()
        metrics['profit_factor'] = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
        
        return metrics

    def plot_performance(self, portfolio):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price and positions
        ax[0].plot(portfolio['price'], label='Price')
        ax[0].plot(portfolio.loc[portfolio['position'] > 0, 'price'], '^', markersize=5, color='g', label='Buy')
        ax[0].plot(portfolio.loc[portfolio['position'] < 0, 'price'], 'v', markersize=5, color='r', label='Sell')
        ax[0].set_title('Trading Signals')
        ax[0].legend()
        
        # Cumulative returns
        ax[1].plot(portfolio['cumulative_strategy'], label='Strategy')
        ax[1].plot(portfolio['cumulative_buy_hold'], label='Buy & Hold')
        ax[1].set_title('Cumulative Returns')
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()

