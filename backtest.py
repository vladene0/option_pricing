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
import backtest_helpers as bh
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# ======= Parameters of the backtest =======

ticker = "SPY"                              # Underlying stock to backtest on; currently supported "SPY", "AAPL"
test_type = "arb"                           # Type of the backtest; "arb" for statistical arbitrage, "hedge" for delta-hedging
stock_model = "gbm"                         # Model used for the stochastic simulation of the stock price; currently supported "gbm", "heston", "jd"
start_date = pd.to_datetime("2021-07-22")   # Start date for backtesting
end_date = pd.to_datetime("2021-07-29")     # End date for backtesting
min_dte = 14                                # Minimum number of days to expiration in order to consider trading an option
n_below_atm = 1                             # Number of strike prices below atm to evaluate for each dte, for each day
n_above_atm = 1                             # Number of strike prices above atm to evaluate for each dte, for each day
n_paths = 500                               # Number of stock evolution paths in the stochastic simulation
n_simulations = 500                         # Number of simulations over which we average to find an option's most likely price
min_rel_diff = 0.2                          # Minimum relative difference between market price and simulated price in order to trade an option
max_rel_diff = 0.3                          # Maximum relative difference between market price and simulated price in order to trade an option
take_win = 0.1                              # Close profitable positions when the profit is greater than take_win percent above initial market price
stop_loss = 0.1                             # Close losing positions when the loss is greater than stop_loss percent of the initial market price
window_vol = 30                             # Size of window for volatility calculation (in days)
n_days_calibration = 1                      # Number of days before current pricing date from which we take data for Heston calibration
n_dte_calibration = 5                       # Number of dte values per day for Heston calibration; total data points per day = (1call+1put)*n_dte_calibration
n_years_jump_data = 5                       # Number of years before current date from which we extract jump data for the Jump-Diffusion model
min_jump_size = 0.04                        # Minimum difference in stock price between consecutive days to count as a jump (percentage of initial price)

# Bundle some of the parameters in a dictionary which is sent to the trade execution function
test_params_trade = {"min_dte": min_dte, "n_below_atm": n_below_atm, "n_above_atm": n_above_atm,
                     "n_paths": n_paths, "n_simulations": n_simulations, "stock_model": stock_model,
                     "min_rel_diff": min_rel_diff, "max_rel_diff": max_rel_diff}

# Backtest parameters necessary for computing model parameters from historical data; for convenience, we also use
# some of the parameters of this dictionary when loading the historical market data and when saving the results
test_params_compute = {"ticker": ticker, "test_type": test_type, "stock_model": stock_model,
                       "start_date": start_date, "end_date": end_date,
                       "n_days_calibration": n_days_calibration, "window_vol": window_vol,
                       "n_years_jump_data": n_years_jump_data, "min_jump_size": min_jump_size}

# ======= Data reading and preparation =======

# Loading the data from our .csv files
option_chain_full, option_chain_selection, unique_dates = bh.load_data(test_params_compute)

# Some basic data cleaning and processing
option_chain_cleaned = bh.clean_data(option_chain_selection)

# Compute historical model parameters from past market data
historical_params = bh.historical_params(test_params_compute, option_chain_full, unique_dates)

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
        model_params = bh.get_model_params(option_chain, daily_options, unique_dates, stock_model,
                                           n_days_calibration, n_dte_calibration)

        if test_type == "arb":
            # Run trade execution in parallel over the different dte values; the parallel operations are independent
            results = Parallel(n_jobs=-1)(delayed(bh.arbitrage)(test_params_trade, model_params, active_trades,
                                                                daily_options[daily_options["dte"] == dte], dte)
                                          for dte in daily_options["dte"].unique())

            # Each day, gather the separate parallel results into single objects
            for n_options_parallel, executed_trades_parallel, active_trades_parallel in results:
                total_options_evaluated += n_options_parallel
                executed_trades += executed_trades_parallel
                active_trades.update(active_trades_parallel)

        if test_type == "hedge":
            hedge_trades = bh.hedge(daily_options, test_params_trade, model_params, daily_closing_stock, option_price_lookup)
            total_options_evaluated += len(hedge_trades)
            executed_trades += hedge_trades

    # For stat-arb, the second step is tracking the daily values of the active options in order to close trades which meet the closing criteria
    if test_type == "arb":
        active_trades, executed_trades = bh.track_trades(active_trades, executed_trades, option_price_lookup, quote_date, take_win, stop_loss)

executed_trades_df = pd.DataFrame(executed_trades)

# Keep only trades which were closed; rare cases of unclosed trades might appear when there is missing data on the expiration date of the option
executed_trades_df = executed_trades_df[executed_trades_df["closed"] == True]

# Result analysis and plotting
bh.analyze_results(executed_trades_df, test_params_compute, total_options_evaluated)
