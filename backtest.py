import option_pricing as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# Work in progress; needs more commenting!

# Time step of one day, expressed in years
dt = 1/365

# ======= Parameters of the backtest =======

ticker = "SPY"                              # Underlying stock to backtest on; currently supported "SPY", "AAPL"
stock_model = "gbm"                         # Model used for the stochastic simulation of the stock price; currently supported "gbm", "heston"
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

# ======= Data reading and preparation =======

project_root_path = r"D:\Scoala\Master-sem3\Practica"

option_chain_paths = {"SPY": project_root_path + r"\spy_2020_2022_30dte.csv",
                      "AAPL": project_root_path + r"\aapl_2021_2023_30dte.csv"}

interest_rates_path = project_root_path + r"\par-yield-curve-rates-2020-2022.csv"

results_save_path = project_root_path + r"\Results"

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
unique_dates_full = option_chain_full["quote_date"].drop_duplicates().sort_values().reset_index(drop=True)
if start_date not in unique_dates_full.values:
    raise ValueError("Invalid backtesting start date! Please choose a working day of the week!")
if end_date not in unique_dates_full.values:
    raise ValueError("Invalid backtesting end date! Please choose a working day of the week!")

# From the original option chain, select data starting from start_date, up to end_date;
# Find the last expiry date for options on the end_date, in order to have enough data to track up to expiry options that have dte remaining on the end_date
mask = (option_chain_full["quote_date"] >= start_date) & (option_chain_full["quote_date"] <= end_date)
option_chain_first_selection = option_chain_full[mask]
max_expire = option_chain_first_selection["expire_date"].max()
if max_expire not in unique_dates_full.values:
    raise ValueError("Please choose an earlier end date; not enough data to track all options to expiration!")

# Obtain the final option chain data, starting from start_date, and up to max_expire
mask = (option_chain_full["quote_date"] >= start_date) & (option_chain_full["quote_date"] <= max_expire)
option_chain = option_chain_full[mask]

# Merge risk-free interest rate data into the option chain
option_chain = option_chain.merge(interest_rates[["quote_date", "rfr"]], on="quote_date", how="left")
option_chain["rfr"] = option_chain["rfr"].ffill()

# Fix some missing price data
option_chain.loc[option_chain["c_last"] == 0, "c_last"] = 0.01
option_chain.loc[option_chain["p_last"] == 0, "p_last"] = 0.01

# Create DataFrame for quickly finding current market prices when backtesting
option_price_lookup = option_chain.set_index(["quote_date", "expire_date", "strike"])

# Find the right date before start_date to have enough data for computing volatility and vol of variance
window = 30
lookback_days = 1 + 2 * (window - 1)
start_idx = unique_dates_full[unique_dates_full == start_date].index[0]
if start_idx < lookback_days:
    raise ValueError("Please choose a later start date; not enough data to compute volatility or vol of variance!")
lookback_idx = start_idx - lookback_days
lookback_start = unique_dates_full.iloc[lookback_idx]

# Create separate dataframe to store daily data for closing prices and volatility
daily_data = option_chain_full.groupby("quote_date")["underlying_last"].first().reset_index()
daily_data["quote_date"] = pd.to_datetime(daily_data["quote_date"])
daily_data = daily_data[(daily_data["quote_date"] >= lookback_start) & (daily_data["quote_date"] <= end_date)]

# Calculate log returns, 30-day volatility, 30-day variance, and 30-day volatility of variance
daily_data["log_returns"] = np.log(daily_data["underlying_last"] / daily_data["underlying_last"].shift(1))
daily_data["volatility_30d"] = daily_data["log_returns"].rolling(window=window).std() * np.sqrt(1/dt)
daily_data["variance_30d"] = daily_data["volatility_30d"]**2
daily_data["vol_of_variance_30d"] = daily_data["variance_30d"].rolling(window=window).std() * np.sqrt(1/dt)  # Not sure about the sqrt here! Look it up later.
daily_data.dropna(subset=["vol_of_variance_30d"], inplace=True)

# Merge volatility data back into the option chain; we will only have volatility values from start_date to end_date
option_chain = option_chain.merge(daily_data[["quote_date", "volatility_30d", "variance_30d", "vol_of_variance_30d"]], on="quote_date", how="left")

# ======= Define function for parallel trade execution =======

# This function iterates through the set of options with the same quote_date and dte, the only variable being the strike price;
# It bundles together all call and put options which are not currently active and does a batch evaluation;
def check_execute(daily_dte_options, quote_date, dte, stock_price, rfr, volatility, mean_variance, vol_of_variance):

    n_calls, n_puts = 0, 0                              # Number of call and put options to evaluate
    call_strikes, put_strikes = [], []                  # Strike prices at which to evaluate the call and put options
    call_market_values, put_market_values = [], []      # The market values of the options to evaluate
    call_ids, put_ids = [], []                          # The identification strings of the options to evaluate
    executed_trades_local = []                          # A local subset of the trades executed by a given parallel worker
    active_trades_local = set()                         # A local subset of the trades currently active from a parallel worker

    # The expiration date is fixed for a given function call, since the dte and quote_date are fixed
    expiration_date = str(daily_dte_options["expire_date"].iloc[0].date())

    # We only choose to trade options with a minimum number of days to expiration remaining
    if dte > min_dte:

        # For each option calculate distance from the "at the money" option
        daily_dte_options["distance_atm"] = abs(daily_dte_options["strike"] - stock_price)
        atm_index = daily_dte_options["distance_atm"].idxmin()

        # We evaluate only the atm option, n_below_atm, and n_above_atm options
        options_to_evaluate = daily_dte_options.loc[atm_index - n_below_atm : atm_index + n_above_atm + 1]

        # Iterate over each row, which contains data for both calls and puts at a given strike price
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
        option_values = op.stochastic_option(n_paths=n_paths, n_days=dte,
                                             initial_price=stock_price, drift=rfr,
                                             volatility=volatility,
                                             mean_variance=mean_variance, vol_of_variance=vol_of_variance,
                                             strike_prices=strike_prices, rfr=rfr, n_calls=n_calls, n_puts=n_puts,
                                             stock_model=stock_model, n_simulations=n_simulations)

        expected_profits = option_values - market_values
        rel_diffs = expected_profits / market_values

        for index, rel_diff in enumerate(rel_diffs):

            # For each of the evaluated options, check the relative difference between the market price and our model's price
            if min_rel_diff < abs(rel_diff) < max_rel_diff:

                # If a trade meets the criteria to be executed, build a dictionary with the transaction data and append it to the executed_trades_local list
                trade_dict = {"open_date": quote_date, "expiration_date": expiration_date, "option_type": option_ids[index].split("_")[0], "dte": dte,
                              "stock_price": stock_price, "strike_price": strike_prices[index],
                              "market_price": market_values[index], "simulation_price": option_values[index], "relative_difference": rel_diff,
                              "expected_profit": abs(expected_profits[index]), "action": "buy" if rel_diff > 0 else "sell", "option_id": option_ids[index],
                              "closed": False, "close_date": None, "close_price": None, "realized_profit": 0.0}
                executed_trades_local.append(trade_dict)
                active_trades_local.add(option_ids[index])

    # These are results for a given dte (from one parallel worker); they are later joined with the other results
    return n_calls + n_puts, executed_trades_local, active_trades_local

# ======= Trade execution and tracking =======

# Begin iterating through all quote dates in order to evaluate options to identify potential trades;
# Also track the values of the already executed trades in order to close profitable or losing positions

# A numeric variable, a list, and a set to store test information
total_options_evaluated, executed_trades, active_trades = 0, [], set()

for quote_date in tqdm(option_chain["quote_date"].unique(), ncols=100):

    # First part, opening new positions only up until the end_date
    if quote_date <= end_date:

        # Select options with the same quote date into a separate Data Frame
        daily_options = option_chain[option_chain["quote_date"] == quote_date].copy()

        # The stock price, the risk-free rate, the volatility, the variance, and the vol of variance are constant for a given quote date
        stock_price = daily_options["underlying_last"].iloc[0]
        rfr = daily_options["rfr"].iloc[0]
        volatility = daily_options["volatility_30d"].iloc[0]
        mean_variance = daily_options["variance_30d"].iloc[0]
        vol_of_variance = daily_options["vol_of_variance_30d"].iloc[0]

        # Run trade execution in parallel over the different dte values; the parallel operations are independent
        results = Parallel(n_jobs=-1)(delayed(check_execute)(daily_options[daily_options["dte"] == dte].copy(), quote_date,
                                                             int(np.round(dte)), stock_price, rfr, volatility, mean_variance, vol_of_variance)
                                      for dte in daily_options["dte"].unique())

        # Each day, gather the separate parallel results into single objects
        for n_options_parallel, executed_trades_parallel, active_trades_parallel in results:
            total_options_evaluated += n_options_parallel
            executed_trades += executed_trades_parallel
            active_trades.update(active_trades_parallel)

    # Second part, tracking the daily values of the active options in order to close trades which meet the closing criteria
    for option_id in active_trades.copy():
        option_type, expiration_date, strike_price = option_id.split("_")
        expiration_date = pd.to_datetime(expiration_date)
        strike_price = float(strike_price)

        try:
            current_market_price = option_price_lookup.loc[(quote_date, expiration_date, strike_price),
                                                           "c_last" if option_type == "call" else "p_last"]

            # In the executed_trades list of dictionaries, find the trade with the current option_id which is still active
            for trade in executed_trades:
                if trade["option_id"] == option_id and trade["closed"] == False:
                    action = trade["action"]
                    initial_market_price = trade["market_price"]

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
                        active_trades.remove(option_id)

                    break

        # Throw error if the current market price is missing from the data (extremely rare in our current data sets)
        except KeyError:
            if quote_date == expiration_date:
                active_trades.remove(option_id)
                print(f"Missing data on expiration for option_id: {option_id}! This will lead to an unclosed trade!")
            else:
                print(f"Missing data on an intermediary date for option_id: {option_id}! Not that big of a deal.")

# ======= Result analysis and plotting =======

executed_trades_df = pd.DataFrame(executed_trades)

# Keep only trades which were closed; rare cases of unclosed trades might appear when there is missing data on the expiration date of the option
executed_trades_df = executed_trades_df[executed_trades_df["closed"] == True]

# Save executed trades to a .csv file
test_id_string = ticker + "_" + str(start_date.date()) + "_" + str(end_date.date())
executed_trades_df.to_csv(results_save_path + r"\executed_trades_" + test_id_string + ".csv", index=False)

print(f"Number of options evaluated: {total_options_evaluated}")
print(f"Number of trades executed: {len(executed_trades_df)}")
print(f"Percentage of evaluated options which were traded: {(len(executed_trades_df) / total_options_evaluated):.2f}")

profit_data = executed_trades_df.groupby("close_date", as_index=False)["realized_profit"].sum()
profit_data = profit_data.sort_values("close_date").reset_index(drop=True)
profit_data["cumulative_profit"] = profit_data["realized_profit"].cumsum()

total_profit = profit_data["cumulative_profit"].iloc[-1]
total_capital_required = executed_trades_df["market_price"].sum()
total_percentage_return = total_profit / total_capital_required

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
plt.savefig(results_save_path + r"\profit_" + test_id_string + ".png", bbox_inches="tight")
