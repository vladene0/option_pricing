"""
This module contains helper functions for the backtest.py script, responsible with loading
and cleaning historical market data, obtaining certain model parameters from the historical
data, implementing the stat-arb and delta-hedging strategies, and analyzing the results.
"""
import option_pricing as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from joblib import Parallel, delayed

project_root_path = os.path.dirname(os.path.abspath(__file__))

option_chain_paths = {"SPY": os.path.join(project_root_path, "spy_2020_2022_30dte.csv"),
                      "AAPL": os.path.join(project_root_path, "aapl_2021_2023_30dte.csv")}

interest_rates_path = os.path.join(project_root_path, "par-yield-curve-rates-2020-2022.csv")

results_save_path = os.path.join(project_root_path, "Results")

if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)

def load_data(test_params):
    """
    Function that loads historical option chain and interest rate data from the .csv files.

    Parameters:
        test_params (dict): Dictionary containing the ticker of the underlying stock, the start and end dates of the backtest,
                            and the number of days for the calibration of the Heston model.

    Returns:
        pd.DataFrame: DataFrame of the complete option chain.
        pd.DataFrame: DataFrame of the option chain for the selected test interval, accounting for calibration
                      and trade tracking to expiration. This DataFrame also contains the interest rate data,
                      as well as the distance from the "at-the-money" level for each option.
        pd.Series: Series of the unique trading dates in our option chain. Useful to properly account for
                   the missing weekend days.
    """
    ticker = test_params["ticker"]
    start_date, end_date = test_params["start_date"], test_params["end_date"]
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

def historical_params(test_params, option_chain_full, unique_dates, dt=1/365):
    """
    Function used for calculating historical model parameters from past market data for the underlying stock.
    We compute the volatility, variance, volatility of variance, as well as the values which describe stock jumps.

    Parameters:
        test_params (dict): A dictionary containing some parameters necessary for the computation (see below).
        option_chain_full (pd.DataFrame): The complete option chain, from which we extract the price evolution of the underlying stock.
        unique_dates (pd.Series): The series of business dates in our option chain.
        dt (float): Time step for volatility calculations. Expressed in years.

    Returns:
        pd.DataFrame: A DataFrame containing the daily values of the calculated parameters.
    """
    ticker = test_params["ticker"]
    start_date, end_date = test_params["start_date"], test_params["end_date"]
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

def get_model_params(option_chain, daily_options, unique_dates, stock_model, n_days_calibration, n_dte_calibration):
    """
    In backtest.py, we iterate through a selected date range to trade options. This function is used to retrieve the daily values
    of the different model parameters from the daily option chain (to which we add these parameters beforehand with the function above).

    Parameters:
        option_chain (pd.DataFrame): Complete option chain for the selected test interval, used to get the calibration data for the Heston model.
        daily_options (pd.DataFrame): Daily option chain from which we extract the daily values of the model parameters.
        unique_dates (pd.Series): The series of business days in the full option chain.
        stock_model (str): The model used for the underlying stock simulation. We need different parameters for different models.
        n_days_calibration (int): Number of days prior to current date from which we use data for Heston calibration.
        n_dte_calibration (int): Number of dte values per day for Heston calibration.

    Returns:
        dict: A dictionary of the daily model parameters we need.
    """
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

        model_params["var_return_rate"], model_params["correlation"] = op.StockHeston.calibrate(calibration_data)

    # Get model parameters for Jump-Diffusion
    if stock_model == "jd":
        model_params["n_jumps_avg"] = daily_options["jumps_per_year"].iloc[0]
        model_params["log_jump_size_avg"] = daily_options["log_jump_avg"].iloc[0]
        model_params["log_jump_size_std"] = daily_options["log_jump_std"].iloc[0]

    return model_params

def arbitrage(test_params, model_params, active_trades, daily_dte_options, dte):
    """
    This function implements the statistical arbitrage strategy. It iterates through the set of options with the same quote date and dte,
    the only variable being the strike price. The function bundles together all call and put options which are not currently active and
    does a batch evaluation. Trades are executed where appropriate, and the execution data is returned as a list of dictionaries.

    Parameters:
        test_params (dict): A dictionary of backtest parameters; see backtest.py or below for its components.
        model_params (dict): A dictionary of model parameters; see the function above.
        active_trades (set[str]): The set of currently active option ids.
        daily_dte_options (pd.DataFrame): DataFrame representing option chain data for fixed quote date and dte.
        dte (int): Number of days to expiration of the options. This value is the same for all options in the daily_dte_options DataFrame.

    Returns:
        int: Number of total call and put options evaluated.
        list[dict]: Each dictionary describes an executed trade.
        set[str]: The set of option ids for the executed trades in order to keep track of active positions more easily.
    """
    # Unpack some backtest parameters into local variables
    min_dte, n_below_atm, n_above_atm = test_params["min_dte"], test_params["n_below_atm"], test_params["n_above_atm"]
    min_rel_diff, max_rel_diff = test_params["min_rel_diff"], test_params["max_rel_diff"]

    n_calls, n_puts = 0, 0                              # Number of call and put options evaluated
    call_strikes, put_strikes = [], []                  # Strike prices at which to evaluate the call and put options
    call_market_values, put_market_values = [], []      # The market values of the options to evaluate
    call_ids, put_ids = [], []                          # The identification strings of the options to evaluate
    executed_trades_local = []                          # A local subset of the trades executed by a given parallel worker
    active_trades_local = set()                         # A local subset of the trades currently active from a parallel worker

    # The expiration date is fixed for a given function call, since the dte and quote_date are fixed
    expiration_date = str(daily_dte_options["expire_date"].iloc[0].date())

    # We only choose to trade options with a minimum number of days to expiration remaining
    if dte > min_dte:

        # Find the option which is "at-the-money"
        atm_index = daily_dte_options["distance_atm"].idxmin()

        # We evaluate only the atm option, n_below_atm, and n_above_atm options
        options_to_evaluate = daily_dte_options.loc[atm_index - n_below_atm : atm_index + n_above_atm + 1]

        # Iterate over each row, which contains data for both a call and a put at a given strike price
        for _, row in options_to_evaluate.iterrows():
            strike_price = row["strike"]

            # An option is uniquely identified through its type, expiration date, and strike price
            call_id = "call_" + expiration_date + "_" + str(strike_price)
            put_id = "put_" + expiration_date + "_" + str(strike_price)

            if call_id not in active_trades:
                n_calls += 1
                call_strikes.append(row["strike"])
                call_market_values.append(row["c_last"])
                call_ids.append(call_id)

            if put_id not in active_trades:
                n_puts += 1
                put_strikes.append(row["strike"])
                put_market_values.append(row["p_last"])
                put_ids.append(put_id)

        # Merge the arrays for calls and puts, then evaluate the selected options
        strike_prices = np.array(call_strikes + put_strikes)
        market_values = np.array(call_market_values + put_market_values)
        option_ids = call_ids + put_ids

        # Call our function for predicting prices with a stochastic model
        option_values = op.stochastic_option(test_params, model_params, dte, strike_prices, n_calls, n_puts)

        expected_profits = option_values - market_values
        rel_diffs = expected_profits / market_values

        for index, rel_diff in enumerate(rel_diffs):

            # For each of the evaluated options, check the relative difference between the market price and our model's price
            if min_rel_diff < abs(rel_diff) < max_rel_diff:

                # If a trade meets the criteria to be executed, build a dictionary with the transaction data and append it to the executed_trades_local list
                trade_dict = {"open_date": model_params["quote_date"], "option_id": option_ids[index], "dte": dte,
                              "stock_price": model_params["stock_price"], "option_price_open": market_values[index],
                              "simulation_price": option_values[index], "relative_difference": rel_diff,
                              "expected_profit": abs(expected_profits[index]), "action": "buy" if rel_diff > 0 else "sell",
                              "closed": False, "close_date": None, "close_price": None, "realized_profit": 0.0}
                executed_trades_local.append(trade_dict)
                active_trades_local.add(option_ids[index])

    # These are results for a given dte (from one parallel worker); they are later joined with the other results
    return n_calls + n_puts, executed_trades_local, active_trades_local

def track_trades(active_trades, executed_trades, option_price_lookup, quote_date, take_win, stop_loss):
    """
    Function used for closing the appropriate trades each day (only for stat-arb).

    Parameters:
        active_trades (set[str]): The set of all currently active option ids.
        executed_trades (list[dict]): The list containing the data for all executed trades as dictionaries.
        option_price_lookup (pd.DataFrame): A reindexed version of the complete option chain to find option prices more quickly.
        quote_date (pd.Timestamp): The current day on which we are tracking the previously executed trades.
        take_win (float): Close profitable positions when the profit is greater than take_win percent above initial market price.
        stop_loss (float): Close losing positions when the loss is greater than stop_loss percent of the initial market price.

    Returns:
        set[str]: The set of active option ids from which the closed positions have been removed.
        list[dict]: The list of dictionaries for the executed trades, in which we update the closed positions.
    """
    # Safer to work with copies of these objects because of the way this function is called in backtest.py
    active_trades_copy = active_trades.copy()
    executed_trades_copy = executed_trades.copy()

    # Go through all currently active trades to identify those which should be closed
    for option_id in active_trades_copy.copy():
        option_type, expiration_date, strike_price = option_id.split("_")
        expiration_date = pd.to_datetime(expiration_date)
        strike_price = float(strike_price)

        try:
            current_market_price = option_price_lookup.loc[(quote_date, expiration_date, strike_price),
                                                           "c_last" if option_type == "call" else "p_last"]

            # In the executed_trades list of dictionaries, find the trade with the current option_id which is still active
            for trade in executed_trades_copy:
                if trade["option_id"] == option_id and trade["closed"] == False:
                    action = trade["action"]
                    initial_market_price = trade["option_price_open"]

                    current_profit = current_market_price - initial_market_price if action == "buy" else initial_market_price - current_market_price

                    # Check whether to close the current position;
                    # This is done in 3 cases: the option comes to expiry, or the option increases or decreases significantly in value
                    if (quote_date == expiration_date or
                        current_profit > take_win * initial_market_price or
                        current_profit < -stop_loss * initial_market_price):

                        trade["closed"] = True
                        trade["close_date"] = quote_date
                        trade["close_price"] = current_market_price
                        trade["realized_profit"] = current_profit
                        active_trades_copy.remove(option_id)

                    break

        # Catch error if the current market price is missing from the data (extremely rare in our current data sets)
        except KeyError:
            if quote_date == expiration_date:
                active_trades_copy.remove(option_id)
                print(f"Missing data on expiration for option_id: {option_id}! This will lead to an unclosed trade!")
            else:
                print(f"Missing data on an intermediary date for option_id: {option_id}! Not that big of a deal.")

    return active_trades_copy, executed_trades_copy

def hedge(daily_options, test_params, model_params, stock_closing, option_price_lookup, dS_percent=0.02):
    """
    This function implements the delta-hedging strategy. We select a set of options to trade and evaluate their deltas through a discrete derivative.
    We then trade each option together with -delta shares of underlying stock. The difference in value of the option-stock position from one day to the
    next is then calculated (ideally, this should be as close to 0 as possible). The results are stored in a list of dictionaries, one for each hedge trade.

    Parameters:
        daily_options (pd.DataFrame): Subset of the complete option chain, containing data for a given trading day.
        test_params (dict): A dictionary of backtest parameters; see backtest.py for its components.
        model_params (dict): A dictionary of model parameters; see the function get_model_params() above.
        stock_closing (pd.DataFrame): A DataFrame containing daily closing prices of the underlying stock.
        option_price_lookup (pd.DataFrame): A reindexed version of the complete option chain to find option prices more quickly.
        dS_percent (float): Percentage representing the difference in underlying stock price used in the delta calculation.

    Returns:
        list[dict]: A list containing the dictionaries corresponding to each hedge trade.
    """
    # For delta-hedging, we choose to trade at-the-money options with a minimum number of days to expiry remaining
    min_dte_options = daily_options[daily_options["dte"] >= test_params["min_dte"]]
    long_options = min_dte_options.loc[min_dte_options.groupby("dte")["distance_atm"].idxmin()]

    # Create new dictionary of model parameters in which the underlying stock price is increased by dS for the delta calculation
    stock_price_open = model_params["stock_price"]
    dS = dS_percent * stock_price_open
    model_params_bumped = model_params.copy()
    model_params_bumped["stock_price"] = stock_price_open + dS

    # We close our positions one day after opening them; find the date and stock price at closing (knowing them ahead doesn't affect the delta calculation)
    quote_date = daily_options["quote_date"].iloc[0]
    quote_date_idx = stock_closing[stock_closing["quote_date"] == quote_date].index[0]
    next_date = stock_closing.loc[quote_date_idx + 1, "quote_date"]
    stock_price_close = stock_closing.loc[quote_date_idx + 1, "underlying_last"]

    def parallel_run(row):
        """
        Function for parallel execution of the delta-hedging strategy over each of the selected rows of the daily option chain.

        Parameters:
            row (pd.DataFrame): An individual row of the daily option chain. Each row corresponds to a pair
                                of call/put options with the same expiration date and strike price.

        Returns:
            tuple[dict]: A pair of dictionaries representing the call trade and the put trade for a given row.
        """
        # Extract some parameters from each row of option chain data
        dte = row["dte"]
        strike = row["strike"]
        expiration_date = row["expire_date"]
        expiration_date_string = str(expiration_date.date())
        call_id = "call_" + expiration_date_string + "_" + str(strike)
        put_id = "put_" + expiration_date_string + "_" + str(strike)

        # Strike prices need to be in an array for the batch evaluation function
        strike_prices = np.array([strike, strike])

        # Market option prices at which we open our positions
        call_open, put_open = row["c_last"], row["p_last"]

        # Evaluate options at the two different underlying prices and then calculate delta
        predicted_prices = op.stochastic_option(test_params, model_params, dte, strike_prices, n_calls=1, n_puts=1)
        predicted_prices_bumped = op.stochastic_option(test_params, model_params_bumped, dte, strike_prices, n_calls=1, n_puts=1)

        call_delta, put_delta = (predicted_prices_bumped - predicted_prices) / dS

        try:
            # Find the price of the option the next day (at which we close the trade) and then calculate the change in our option-stock position;
            # Create a dictionary to keep track of the trade
            call_close = option_price_lookup.loc[(next_date, expiration_date, strike), "c_last"]
            call_hedged_change = call_close - call_open - call_delta * (stock_price_close - stock_price_open)
            call_dict = {"open_date": quote_date, "option_id": call_id, "dte": dte,
                         "stock_price_open": stock_price_open, "option_price_open": call_open,
                         "stock_price_close": stock_price_close, "option_price_close": call_close,
                         "delta_sim": call_delta, "delta_market": (call_close - call_open) / (stock_price_close - stock_price_open),
                         "close_date": next_date, "realized_profit": call_hedged_change, "closed": True}

            put_close = option_price_lookup.loc[(next_date, expiration_date, strike), "p_last"]
            put_hedged_change = put_close - put_open - put_delta * (stock_price_close - stock_price_open)
            put_dict = {"open_date": quote_date, "option_id": put_id, "dte": dte,
                        "stock_price_open": stock_price_open, "option_price_open": put_open,
                        "stock_price_close": stock_price_close, "option_price_close": put_close,
                        "delta_sim": put_delta, "delta_market": (put_close - put_open) / (stock_price_close - stock_price_open),
                        "close_date": next_date, "realized_profit": put_hedged_change, "closed": True}
            
            return (call_dict, put_dict)

        except KeyError:
            print("No option price available to close hedging trade!")
            return (None, None)

    # For the current daily iteration where the hedge function is called, run a parallel hedging simulation for all options in the selected long_options data;
    # The results are collected in a list of trade dictionaries
    results = Parallel(n_jobs=-1)(delayed(parallel_run)(row) for _, row in long_options.iterrows())
    hedge_trades = []
    for call_dict, put_dict in results:
        hedge_trades.append(call_dict)
        hedge_trades.append(put_dict)

    return hedge_trades

def analyze_results(executed_trades_df, test_params, total_options_evaluated):
    """
    Function that saves the executed trades as a .csv file, prints some test results to
    the terminal, and saves a plot of the cumulative profit and percentage return of the test.

    Parameters:
        executed_trades_df (pd.DataFrame): The DataFrame of all executed trades in the test interval.
        test_params (dict): Parameters of the test used in the names of the saved files for identification.
        total_options_evaluated (int): Number of options evaluated over the entire test (not necessarily executed trades).
    """
    ticker, test_type, stock_model = test_params["ticker"], test_params["test_type"], test_params["stock_model"]
    start_date, end_date = test_params["start_date"], test_params["end_date"]

    # Save executed trades to a .csv file
    test_id_string = f"{test_type}_{ticker}_{stock_model}_{start_date.date()}_{end_date.date()}"
    executed_trades_df.to_csv(os.path.join(results_save_path, test_id_string) + ".csv", index=False)

    profit_data = executed_trades_df.groupby("close_date", as_index=False)["realized_profit"].sum()
    profit_data = profit_data.sort_values("close_date").reset_index(drop=True)
    profit_data["cumulative_profit"] = profit_data["realized_profit"].cumsum()

    total_profit = profit_data["cumulative_profit"].iloc[-1]
    total_capital_required = executed_trades_df["option_price_open"].sum()
    total_percentage_return = total_profit / total_capital_required

    print(f"Number of options evaluated: {total_options_evaluated}")
    print(f"Number of trades executed: {len(executed_trades_df)}")
    print(f"Percentage of evaluated options which were traded: {(len(executed_trades_df) / total_options_evaluated):.2f}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"Total capital required: {total_capital_required:.2f}")
    print(f"Total percentage return: {total_percentage_return:.4f}")

    # Plot the results of the backtest
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

    ax1.plot(profit_data["close_date"], profit_data["cumulative_profit"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative profit")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title("Realized cumulative profit")

    ax2.plot(profit_data["close_date"], profit_data["cumulative_profit"] / total_capital_required)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Percentage return")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_title("Realized percentage return")

    plt.tight_layout()
    plt.savefig(os.path.join(results_save_path, test_id_string) + ".png", bbox_inches="tight")
