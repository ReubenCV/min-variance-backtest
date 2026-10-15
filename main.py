import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_mock_data(n_assets=10, n_periods=250):
    np.random.seed(42)
    mean_returns = np.random.uniform(0.0002, 0.0008, n_assets)
    cov_matrix = np.random.uniform(-0.0001, 0.0003, (n_assets, n_assets))
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    tickers = [f"ASSET_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=tickers)

def min_variance_objective(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def optimize_minimum_variance(returns_df):
    cov_matrix = returns_df.cov().values
    n_assets = len(returns_df.columns)
    initial_weights = np.ones(n_assets) / n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    result = minimize(
        min_variance_objective,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds = bounds,
        constraints=constraints
    )
    return pd.Series(result.x, index=returns_df.columns)

def calculate_metrics(returns_series, risk_free_rate=0.02/252):
    ann_return = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_return - (risk_free_rate * 252)) / ann_vol if ann_vol > 0 else 0
    return ann_return, ann_vol, sharpe

if __name__ == "__main__":
    print("--- Running Minimum Variance Portfolio MVO ---\n")
    returns = generate_mock_data()
    print(f"Loaded returns data for {returns.shape} assets over {returns.shape} periods.")
    optimal_weights = optimize_minimum_variance(returns)
    print("\nOptimal Portfolio Weights (Minimum Variance):")
    print(optimal_weights.round(4))
    min_var_returns = (returns * optimal_weights).sum(axis=1)
    equal_weight_returns = returns.mean(axis=1)
    mv_ret, mv_vol, mv_sharpe = calculate_metrics(min_var_returns)
    eq_ret, eq_vol, eq_sharpe = calculate_metrics(equal_weight_returns)
    print("\n--- Performance Summary ---")
    print(f"Min Variance Portfolio  -> Ann. Return: {mv_ret:.2}, Ann. Vol: {mv_vol:.2%}, Sharpe: {mv_sharpe:.2f}")
    print(f"Equal-Weight Benchmark -> Ann. Return: {eq_ret:.2}, Ann. Vol: {eq_vol:.2%}, Sharpe: {eq_sharpe:.2f}")