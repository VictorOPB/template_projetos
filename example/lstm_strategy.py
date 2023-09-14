import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random
from modules.get_retornos_sp import get_retornos_sp
from modules.initial_weights import get_uniform_noneg

def _sel_stocks(returns, size):
    """
    Select top stocks based on stocks' momentum scores analyzed for 3 periods: 1-month momentum, 3-month and
    6-month momentum, compounded with low volatility filtering. A 'low-volatility-momentum' strategy, per say.

    Args:
        returns (DataFrame): DataFrame containing returns data.
        size (int): Number of stocks to select.

    Returns:
        DataFrame: Selected returns data.
    """
    if (size is None) or (size > len(returns.columns)):
        size = len(returns.columns)

    low_volatility_period = 63  # Assuming daily data

    # Initialize an empty DataFrame to store momentum scores
    momentum_scores_df = pd.DataFrame(index=returns.index)
    vol_df = pd.DataFrame(index=returns.index)
    volatilities = []
    for column in returns:
        # Calculate the rolling standard deviation to measure mean volatility for this stock
        vol_df[f'{low_volatility_period}-Day Low Volatility of {column} Stock'] = returns[column].rolling(low_volatility_period).std()
        vol_values = np.nan_to_num(returns[column].rolling(low_volatility_period).std().values)
        mean_vol = np.mean(vol_values)
        volatilities.append(mean_vol)

        # Define multiple momentum periods in terms of trading days
        momentum_periods = [21, 63, 126]  # 1-month, 3-month, and 6-month

        # Calculate momentum scores for each period and store in the DataFrame
        for period in momentum_periods:
            momentum_scores = returns[column].rolling(period).sum()
            momentum_scores = momentum_scores.shift(-period)
            momentum_scores_df[f'{period}-Day Momentum of {column} Stock'] = momentum_scores

    momentum_scores_df.fillna(0)
    vol_df.fillna(0)

    # Define thresholds for low volatility and momentum
    low_volatility_threshold = 0.015  # Adjust as needed
    momentum_threshold = 0.8  # Adjust as needed

    # Create boolean DataFrames for low volatility and momentum
    low_volatility_condition = vol_df < low_volatility_threshold
    momentum_condition = momentum_scores_df > momentum_threshold

    # Combine the conditions to select assets
    selected_assets_df = low_volatility_condition & momentum_condition

    return selected_assets_df # must filter correctly; end result is WRONG!

def _calculate_risk_stat(weights, returns):
    """
    Calculate the risk statistic based on portfolio weights and returns' covariance.

    Args:
        weights (array-like): Portfolio weights.
        returns (DataFrame): DataFrame containing returns data.

    Returns:
        float: Calculated risk statistic.
    """
    return (weights @ returns.cov() @ weights * 252) ** 0.5

def strategy_minRisk(data, t, size=30, window_size=500):
    """
    Implement a [INPUT MY STRATEGY] strategy

    Args:
        data (dict): Data dictionary containing 'sp' and 'prices' DataFrames.
        t (int): The desired time.
        size (int, optional): Number of stocks to consider. Defaults to 30.
        window_size (int, optional): Size of the window for calculations. Defaults to 500.

    Returns:
        DataFrame: Optimal weights for the selected stocks.
    """
    returns = get_retornos_sp(data, t, window_size)
    returns_sel = _sel_stocks(returns, size)
    initial_weights = get_uniform_noneg(size)

    # Define optimization constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights must equal 1
    ]

    # Define bounds for weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(size))

    # Perform optimization
    result = minimize(
        _calculate_risk_stat,
        initial_weights,
        args=(returns_sel,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    opt_weights = pd.DataFrame({
        'date': [data['prices'].index[t]] * len(result.x),
        'ticker': returns_sel.columns,
        'weights': result.x,
    })

    return opt_weights