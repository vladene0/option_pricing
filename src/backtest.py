"""
This is a script for testing the option pricing models in the option_pricing.py module
on historical market data, using statistical arbitrage and delta-hedging strategies.

The goal of statistical arbitrage is to obtain a profit from trading options directly,
while the goal of delta-hedging is to maintain the realized profit/loss as close to 0
as possible (relevant for market making).

The general steps of the statistical arbitrage strategy are the following: predict option price -> compare to market price ->
                                                                           if option is mispriced open long or short position ->
                                                                           track option value daily, close position to take win or stop loss (or at expiry).

The general steps of the delta-hedging strategy are the following: compute an option's delta -> trade it together with the corresponding
                                                                   amount of stock -> track realized change in value of the option-stock position.

In the end, we generate a .csv file of the executed trades and a plot of the profit evolution.
"""
import helpers.data_utils as data_utils
import helpers.strategies as strategies
import os
import json5
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Load parameters of the backtest
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.json5")
with open(config_path, "r") as f:
    test_params = json5.load(f)

start_date = pd.to_datetime(test_params["start_date"])
end_date = pd.to_datetime(test_params["end_date"])
test_type = test_params["test_type"]

# ======= Data reading and preparation =======

# Loading the data from our .csv files
option_chain_full, option_chain_selection, unique_dates = data_utils.load_data(test_params)

# Some basic data cleaning and processing
option_chain_cleaned = data_utils.clean_data(option_chain_selection)

# Compute historical model parameters from past market data
historical_params = data_utils.historical_params(option_chain_full, unique_dates, test_params)

# Merge calculated historical parameters back into the option chain
option_chain = option_chain_cleaned.merge(historical_params[["quote_date", "volatility_30d", "variance_30d", "vol_of_variance_30d",
                                                             "jumps_per_year", "log_jump_avg", "log_jump_std"]], on="quote_date", how="left")

# Get closing stock prices for each day
daily_closing_stock = option_chain.groupby("quote_date")["underlying_last"].first().reset_index()
daily_closing_stock["quote_date"] = pd.to_datetime(daily_closing_stock["quote_date"])

# Create DataFrame for quickly finding current market prices when backtesting
option_price_lookup = option_chain.set_index(["quote_date", "expire_date", "strike"])

# ======= Trade execution and tracking =======

# A numeric variable, a list, and a set to store test information
total_options_evaluated, executed_trades, active_trades = 0, [], set()

# Begin iterating through all quote dates in order to apply the stat-arb/delta-hedging strategies (separately, depending on test_type)
for quote_date in tqdm(option_chain["quote_date"].unique(), ncols=100):

    # For stat-arb, the first step is opening new positions up until the end_date; for delta-hedging, just call the function
    # Also keep in mind that option_chain contains data one day before start for calibration
    if start_date <= quote_date <= end_date:

        # Select options with the same quote date into a separate DataFrame
        daily_options = option_chain[option_chain["quote_date"] == quote_date]

        # Retrieve historical model parameters for the current day
        model_params = data_utils.get_model_params(option_chain, daily_options, unique_dates, test_params)

        if test_type == "arb":
            # Run trade execution in parallel over the different dte values; the parallel operations are independent
            results = Parallel(n_jobs=-1)(delayed(strategies.arbitrage)(daily_options[daily_options["dte"] == dte],
                                                                        active_trades, dte, test_params, model_params)
                                          for dte in daily_options["dte"].unique())

            # Each day, gather the separate parallel results into single objects
            for n_options_parallel, executed_trades_parallel, active_trades_parallel in results:
                total_options_evaluated += n_options_parallel
                executed_trades += executed_trades_parallel
                active_trades.update(active_trades_parallel)

        if test_type == "hedge":
            hedge_trades = strategies.hedge(daily_options, option_price_lookup, daily_closing_stock, test_params, model_params)
            total_options_evaluated += len(hedge_trades)
            executed_trades += hedge_trades

    # For stat-arb, the second step is tracking the daily values of the active options in order to close trades which meet the closing criteria
    if test_type == "arb":
        active_trades, executed_trades = strategies.track_trades(active_trades, executed_trades, option_price_lookup, quote_date, test_params)

executed_trades_df = pd.DataFrame(executed_trades)

# Keep only trades which were closed; rare cases of unclosed trades might appear when there is missing data on the expiration date of the option
executed_trades_df = executed_trades_df[executed_trades_df["closed"] == True]

# Result analysis and plotting
strategies.analyze_results(executed_trades_df, total_options_evaluated, test_params)
