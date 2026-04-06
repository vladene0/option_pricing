"""
This module contains helper functions for the backtest.py script, responsible with loading and
cleaning historical market data, and obtaining certain model parameters from the historical data.
"""
from .option_pricing import StockHeston
import numpy as np
import pandas as pd
import yfinance as yf
import os

def load_data(test_params):
    """
    Function that loads historical option chain and interest rate data from the .csv files.

    Parameters:
        test_params (dict): Dictionary of backtest parameters. See config.json5.

    Returns:
        pd.DataFrame: DataFrame of the complete option chain.
        pd.DataFrame: DataFrame of the option chain for the selected test interval, accounting for calibration
                      and trade tracking to expiration. This DataFrame also contains the interest rate data,
                      as well as the distance from the "at-the-money" level for each option.
        pd.Series: Series of the unique trading dates in our option chain. Useful to properly account for
                   the missing weekend days.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    datasets_path = os.path.join(current_dir, "..", "..", "datasets")

    option_chain_paths = {"SPY": os.path.join(datasets_path, "spy_2020_2022_30dte.csv"),
                          "AAPL": os.path.join(datasets_path, "aapl_2021_2023_30dte.csv"),
                          "QQQ": os.path.join(datasets_path, "qqq_2021_2022_30dte.csv")}

    interest_rates_path = os.path.join(datasets_path, "interest_rates_2020_2022.csv")

    ticker = test_params["ticker"]
    start_date, end_date = pd.to_datetime(test_params["start_date"]), pd.to_datetime(test_params["end_date"])
    n_days_calibration = test_params["n_days_calibration"]

    # Read historical option chain data
    option_chain_full = pd.read_csv(option_chain_paths[ticker])
    option_chain_full["quote_date"] = pd.to_datetime(option_chain_full["quote_date"], format="%Y-%m-%d")
    option_chain_full["expire_date"] = pd.to_datetime(option_chain_full["expire_date"], format="%Y-%m-%d")

    # Read historical risk-free interest rate data
    interest_rates = pd.read_csv(interest_rates_path)
    interest_rates.rename(columns={"Date": "quote_date"}, inplace=True)
    interest_rates["quote_date"] = pd.to_datetime(interest_rates["quote_date"], format="%m/%d/%y")

    # We are going to use 1 month interest rates for option pricing over medium-length time periods
    interest_rates["rfr"] = interest_rates["1 Mo"] / 100

    # Check we have data on the start and end dates (they might be weekend days)
    unique_dates = option_chain_full["quote_date"].drop_duplicates().sort_values().reset_index(drop=True)
    if start_date not in unique_dates.values:
        raise ValueError("Invalid backtesting start date! Please choose a working day of the week!")
    if end_date not in unique_dates.values:
        raise ValueError("Invalid backtesting end date! Please choose a working day of the week!")

    # From the original option chain, select data starting from start_date, up to end_date;
    # Find the last expiry date for options on the end_date, in order to have enough data to track up to expiry options that have dte remaining on the end_date
    mask = (option_chain_full["quote_date"] >= start_date) & (option_chain_full["quote_date"] <= end_date)
    option_chain_first_selection = option_chain_full[mask]
    max_expire = option_chain_first_selection["expire_date"].max()
    if max_expire not in unique_dates.values:
        raise ValueError("Please choose an earlier end date; not enough data to track all options to expiration!")

    # Obtain the final option chain data, starting from n_days_calibration before start_date (for model calibration data), and up to max_expire
    start_date_idx = unique_dates[unique_dates == start_date].index[0]
    calibration_start = unique_dates.loc[start_date_idx - n_days_calibration]
    mask = (option_chain_full["quote_date"] >= calibration_start) & (option_chain_full["quote_date"] <= max_expire)
    option_chain = option_chain_full[mask].copy()

    # Merge risk-free interest rate data into the option chain
    option_chain = option_chain.merge(interest_rates[["quote_date", "rfr"]], on="quote_date", how="left")
    option_chain["rfr"] = option_chain["rfr"].ffill()

    # For each option, calculate the distance from the "at-the-money" level
    option_chain["distance_atm"] = abs(option_chain["strike"] - option_chain["underlying_last"])

    return option_chain_full, option_chain, unique_dates

def clean_data(option_chain_dirty):
    """
    Function that does some minor cleaning of the option chain. Currently, this function only corrects
    some missing option price data and rounds the few non-integer "days-to-expiry" values.

    Parameters:
        option_chain_dirty (pd.DataFrame): Initial option chain.

    Returns:
        pd.DataFrame: Cleaned option chain.
    """
    option_chain_clean = option_chain_dirty.copy()

    # Fix some missing price data
    option_chain_clean.loc[option_chain_clean["c_last"] == 0, "c_last"] = 0.01
    option_chain_clean.loc[option_chain_clean["p_last"] == 0, "p_last"] = 0.01

    # Fix some non-integer dte values (ex. 5.99, and so on)
    option_chain_clean["dte"] = option_chain_clean["dte"].round().astype(int)

    return option_chain_clean

def historical_params(option_chain_full, unique_dates, test_params, dt=1/365):
    """
    Function used for calculating historical model parameters from past market data for the underlying stock.
    We compute the volatility, variance, volatility of variance, as well as the values which describe stock jumps.

    Parameters:
        option_chain_full (pd.DataFrame): The complete option chain, from which we extract the price evolution of the underlying stock.
        unique_dates (pd.Series): The series of business dates in our option chain.
        test_params (dict): Dictionary of backtest parameters. See config.json5.
        dt (float): Time step for volatility calculations. Expressed in years.

    Returns:
        pd.DataFrame: A DataFrame containing the daily values of the calculated parameters.
    """
    ticker = test_params["ticker"]
    start_date, end_date = pd.to_datetime(test_params["start_date"]), pd.to_datetime(test_params["end_date"])
    n_days_calibration = test_params["n_days_calibration"]
    window_vol = test_params["window_vol"]
    n_years_jump_data, min_jump_size = test_params["n_years_jump_data"], test_params["min_jump_size"]

    # Find the right date before start_date to have enough data for computing volatility and vol of variance, as well as for Heston calibration;
    # Lookback -> n_days_calibration for calibration data, window - 1 for volatility, window - 1 for vol_of_variance
    start_date_idx = unique_dates[unique_dates == start_date].index[0]
    lookback_days = n_days_calibration + 2 * (window_vol - 1)
    if start_date_idx < lookback_days:
        raise ValueError("Please choose a later start date; not enough data to compute volatility or vol of variance!")
    lookback_idx = start_date_idx - lookback_days
    lookback_start = unique_dates.iloc[lookback_idx]

    # Create separate dataframe to store daily data for closing prices
    daily_data = option_chain_full.groupby("quote_date")["underlying_last"].first().reset_index()
    daily_data["quote_date"] = pd.to_datetime(daily_data["quote_date"])
    daily_data = daily_data[(daily_data["quote_date"] >= lookback_start) & (daily_data["quote_date"] <= end_date)]

    # Download underlying price data from YahooFinance for the Jump-Diffusion model, going back quite a few years
    jump_start = start_date - pd.DateOffset(years=n_years_jump_data)
    jump_end = end_date + pd.DateOffset(days=1)

    # We only need the closing prices for each date
    jump_data = yf.Ticker(ticker).history(start=str(jump_start.date()), end=str(jump_end.date()), interval="1d", auto_adjust=False)
    jump_data = jump_data["Close"].reset_index()
    jump_data.rename(columns={"Date": "quote_date", "Close": "underlying_last"}, inplace=True)
    jump_data["quote_date"] = pd.to_datetime(jump_data["quote_date"]).dt.tz_localize(None)

    # The number of past days over which we count jumps for a given date - equal to the number of trading days in the "n_years_jump_data" past years
    jump_window_days = len(jump_data[jump_data["quote_date"] < start_date])

    # Combine the jump_data with the original daily_data
    jump_data = jump_data[jump_data["quote_date"] < lookback_start]
    daily_data = pd.concat([jump_data, daily_data]).reset_index(drop=True)

    # Calculate log returns - the logarithm of the ratio of consecutive closing prices
    daily_data["log_returns"] = np.log(daily_data["underlying_last"] / daily_data["underlying_last"].shift(1))

    # Find days where jumps happened and for each day count the number of jumps that happened in the past window
    jump_bool = (daily_data["log_returns"] > np.log(1 + min_jump_size)) | (daily_data["log_returns"] < np.log(1 - min_jump_size))
    daily_data["jump_count"] = jump_bool.rolling(window=jump_window_days, min_periods=1).sum().shift(1)
    daily_data["jumps_per_year"] = daily_data["jump_count"] / n_years_jump_data

    # Obtain the mean and standard deviation of the log-jumps in the given window
    log_returns_roll = daily_data["log_returns"].where(jump_bool).rolling(window=jump_window_days, min_periods=1)
    daily_data["log_jump_avg"] = log_returns_roll.mean().shift(1)
    daily_data["log_jump_std"] = log_returns_roll.std().shift(1)

    # Filter once again to get rid of unnecessary jump data
    daily_data = daily_data[daily_data["quote_date"] >= lookback_start]

    # Obtain 30-day volatility, 30-day variance, and 30-day volatility of variance
    daily_data["volatility_30d"] = daily_data["log_returns"].rolling(window=window_vol).std() * np.sqrt(1/dt)
    daily_data["variance_30d"] = daily_data["volatility_30d"]**2
    daily_data["vol_of_variance_30d"] = daily_data["variance_30d"].rolling(window=window_vol).std() * np.sqrt(1/dt)
    daily_data.dropna(subset=["vol_of_variance_30d"], inplace=True)

    return daily_data

def get_model_params(option_chain, daily_options, unique_dates, test_params):
    """
    In backtest.py, we iterate through a selected date range to trade options. This function is used to retrieve the daily values
    of the different model parameters from the daily option chain (to which we add these parameters beforehand with the function above).

    Parameters:
        option_chain (pd.DataFrame): Complete option chain for the selected test interval, used to get the calibration data for the Heston model.
        daily_options (pd.DataFrame): Daily option chain from which we extract the daily values of the model parameters.
        unique_dates (pd.Series): The series of business days in the full option chain.
        test_params (dict): Dictionary of backtest parameters. See config.json5.

    Returns:
        dict: A dictionary of the daily model parameters we need.
    """
    stock_model = test_params["stock_model"]
    n_days_calibration = test_params["n_days_calibration"]
    n_dte_calibration = test_params["n_dte_calibration"]
    quote_date = daily_options["quote_date"].iloc[0]

    # The stock price, the risk-free rate, and the volatility are constant for a given quote date - store them in a dictionary
    model_params = {"quote_date": quote_date,
                    "stock_price": daily_options["underlying_last"].iloc[0],
                    "rfr": daily_options["rfr"].iloc[0],
                    "volatility": daily_options["volatility_30d"].iloc[0]}

    # If we use the Heston model for the stock, we need to extract its parameters; the mean_variance and vol_of_variance are calculated
    # beforehand from the historical data, while the var_return_rate and correlation are obtained with our calibration function
    if stock_model == "heston":
        model_params["mean_variance"] = daily_options["variance_30d"].iloc[0]
        model_params["vol_of_variance"] = daily_options["vol_of_variance_30d"].iloc[0]

        # Get options from the n_days_calibration previous days for model calibration
        calib_start_date_idx = unique_dates[unique_dates == quote_date].index[0] - n_days_calibration
        calib_end_date_idx = unique_dates[unique_dates == quote_date].index[0] - 1
        calib_start_date = unique_dates.loc[calib_start_date_idx]
        calib_end_date = unique_dates.loc[calib_end_date_idx]
        calibration_data = option_chain[(option_chain["quote_date"] >= calib_start_date) & (option_chain["quote_date"] <= calib_end_date)]

        # Select a subset of the calibration data: for each quote_date and for each dte select the ATM options, then select n_dte_calibration rows for each day
        calibration_data = calibration_data.loc[calibration_data.groupby(["quote_date", "dte"])["distance_atm"].idxmin()]
        calibration_data = calibration_data.groupby("quote_date").head(n_dte_calibration)

        model_params["var_return_rate"], model_params["correlation"] = StockHeston.calibrate(calibration_data)

    # Get model parameters for Jump-Diffusion
    if stock_model == "jd":
        model_params["n_jumps_avg"] = daily_options["jumps_per_year"].iloc[0]
        model_params["log_jump_size_avg"] = daily_options["log_jump_avg"].iloc[0]
        model_params["log_jump_size_std"] = daily_options["log_jump_std"].iloc[0]

    return model_params
