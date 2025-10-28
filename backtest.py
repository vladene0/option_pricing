import option_pricing as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Time step of one day, expressed in years
dt = 1/365

# Look for data on historical risk free rates later
rfr = 0

# List of columns to be read as dates
parse_dates = ["quote_date", "expire_date"]
option_chain_full = pd.read_csv(r"D:\Scoala\Master-sem3\Practica\spy_2020_2022_30dte.csv", parse_dates=parse_dates)

# Optionally, plot the stock price evolution
plot_underlying = False
if plot_underlying:
    plt.figure(figsize=(8, 6))
    plt.plot(option_chain_full["quote_date"], option_chain_full["underlying_last"])
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("SPY 2020-2022")
    plt.show()

# Choose start date and end date for backtesting; check we have data on these dates (they might be weekend days)
start_date = pd.to_datetime("2021-07-28")
end_date = pd.to_datetime("2021-07-29")
unique_dates_full = option_chain_full["quote_date"].drop_duplicates().sort_values().reset_index(drop=True)
if start_date not in unique_dates_full.values:
    raise ValueError("Invalid backtesting start date! Please choose a working day of the week!")
if end_date not in unique_dates_full.values:
    raise ValueError("Invalid backtesting end date! Please choose a working day of the week!")

# From the original option chain, select data starting from start_date, up to end_date;
# Find the last expiry date for options on the end_date, in order to have enough data to track up to expiry options that might have dte remaining on the end_date
mask = (option_chain_full["quote_date"] >= start_date) & (option_chain_full["quote_date"] <= end_date)
option_chain_first_selection = option_chain_full[mask]
max_expire = option_chain_first_selection["expire_date"].max()
if max_expire not in unique_dates_full.values:
    raise ValueError("Please choose an earlier end date; not enough data to track all options to expiration!")

# Obtain the final option chain data, starting from start_date, and up to max_expire
mask = (option_chain_full["quote_date"] >= start_date) & (option_chain_full["quote_date"] <= max_expire)
option_chain = option_chain_full[mask]

# Find the date 30 trading days before start_date to have enough data for computing volatility
lookback_days = 30
start_idx = unique_dates_full[unique_dates_full == start_date].index[0]
if start_idx < 30:
    raise ValueError("Please choose a later start date; not enough data to compute volatility!")
lookback_idx = start_idx - lookback_days
lookback_start = unique_dates_full.iloc[lookback_idx]

# Create separate dataframe to store daily data for closing prices and volatility
daily_data = option_chain_full.groupby("quote_date")["underlying_last"].first().reset_index()
daily_data["quote_date"] = pd.to_datetime(daily_data["quote_date"])
daily_data = daily_data[(daily_data["quote_date"] >= lookback_start) & (daily_data["quote_date"] <= end_date)]

# Calculate log returns and 30-day volatility
daily_data["log_returns"] = np.log(daily_data["underlying_last"] / daily_data["underlying_last"].shift(1))
daily_data["volatility_30d"] = daily_data["log_returns"].rolling(window=lookback_days).std() * np.sqrt(1/dt)
daily_data.dropna(subset=["volatility_30d"], inplace=True)

# Merge volatility data back into the option chain; we will only have volatility values from start_date to end_date
option_chain = option_chain.merge(daily_data[["quote_date", "volatility_30d"]], on="quote_date", how="left")

# Set some parameters of the backtest
potential_trades = []
min_dte = 5
n_below_atm = 1
n_above_atm = 1
n_paths = 50
n_simulations = 50
min_rel_diff = 0.2

# Begin evaluating options from the option chain for each quote date from start_date to end_date
total_options_evaluated = 0
for quote_date in daily_data["quote_date"]:

    # Select options with the same quote date into a separate Data Frame
    daily_options = option_chain[option_chain["quote_date"] == quote_date].copy()

    # Stock price and volatility are constant for a given quote date
    stock_price = daily_options["underlying_last"].iloc[0]
    volatility = daily_options["volatility_30d"].iloc[0]

    # Loop over the existent dte values
    for dte in daily_options["dte"].unique():

        # Evaluate options with at least min_dte remaining
        if dte > min_dte:

            # Select separately options with the same dte for the current quote date
            daily_dte_options = daily_options[daily_options["dte"] == dte].copy()
            dte = int(np.round(dte))

            # For each option calculate distance from the "at the money" option
            daily_dte_options["distance_atm"] = abs(daily_dte_options["strike"] - stock_price)
            atm_index = daily_dte_options["distance_atm"].idxmin()

            # We evaluate only the atm option, n_below_atm, and n_above_atm options
            options_to_evaluate = daily_dte_options.loc[atm_index - n_below_atm : atm_index + n_above_atm]

            for index, row in options_to_evaluate.iterrows():
                strike_price = row["strike"]
                total_options_evaluated += 2

                call_value = op.stochastic_option(n_paths=n_paths, n_simulations=n_simulations,
                                                  n_days=dte, initial_price=stock_price, drift=rfr, volatility=volatility,
                                                  strike_price=strike_price, rfr=rfr, option_type="call")

                put_value = op.stochastic_option(n_paths=n_paths, n_simulations=n_simulations,
                                                 n_days=dte, initial_price=stock_price, drift=rfr, volatility=volatility,
                                                 strike_price=strike_price, rfr=rfr, option_type="put")

                # For each evaluated call and put option, calculate relative difference from the market price
                call_rel_diff = (call_value - row["c_last"]) / row["c_last"]
                put_rel_diff = (put_value - row["p_last"]) / row["p_last"]

                # If the relative price differences are large enough, keep track of the potential trade
                if abs(call_rel_diff) > min_rel_diff:
                    call_trade_dict = {"quote_date": quote_date, "option_type": "call", "dte": dte, "stock_price": stock_price, "strike_price": strike_price,
                                       "market_price": row["c_last"], "simulation_price": call_value, "relative_difference": call_rel_diff}
                    potential_trades.append(call_trade_dict)

                if abs(put_rel_diff) > min_rel_diff:
                    put_trade_dict = {"quote_date": quote_date, "option_type": "put", "dte": dte, "stock_price": stock_price, "strike_price": strike_price,
                                      "market_price": row["p_last"], "simulation_price": put_value, "relative_difference": put_rel_diff}
                    potential_trades.append(put_trade_dict)

potential_trades_df = pd.DataFrame(potential_trades)
print(potential_trades_df)
print(f"Number of options evaluated: {total_options_evaluated}")
