import numpy as np
import pandas as pd
from scipy.optimize import minimize


def generate_mock_data(n_assets=10, n_periods=250):
    """Generates synthetic daily return data for MVP testing."""
    np.random.seed(42)
    mean_returns = np.random.uniform(0.0002, 0.0008, n_assets)
    cov_matrix = np.random.uniform(-0.0001, 0.0003, (n_assets, n_assets))
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    tickers = [f"ASSET_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=tickers)


def min_variance_objective(weights, cov_matrix):
    """Objective function: Portfolio Variance = w^T * Sigma * w"""
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def optimize_minimum_variance(returns_df):
    """Solves quadratic programming for long-only minimum variance weights."""
    cov_matrix = returns_df.cov().values
    n_assets = len(returns_df.columns)
    initial_weights = np.ones(n_assets) / n_assets

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    result = minimize(
        min_variance_objective,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return pd.Series(result.x, index=returns_df.columns)


def calculate_metrics(returns_series, risk_free_rate=0.02/252):
    """Calculates annualized return, annualized volatility, and Sharpe Ratio."""
    ann_return = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_return - (risk_free_rate * 252)) / ann_vol if ann_vol > 0 else 0
    return ann_return, ann_vol, sharpe


if __name__ == "__main__":
    print("--- Running Minimum Variance Portfolio MVP ---\n")
    returns = generate_mock_data()
    print(f"Loaded returns data for
