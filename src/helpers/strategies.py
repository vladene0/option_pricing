"""
This module contains helper functions for the backtest.py script, responsible with
implementing the stat-arb and delta-hedging strategies, and analyzing the results.
"""
from .option_pricing import stochastic_option
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

def arbitrage(daily_dte_options, active_trades, dte, test_params, model_params):
    """
    This function implements the statistical arbitrage strategy. It iterates through the set of options with the same quote date and dte,
    the only variable being the strike price. The function bundles together all call and put options which are not currently active and
    does a batch evaluation. Trades are executed where appropriate, and the execution data is returned as a list of dictionaries.

    Parameters:
        daily_dte_options (pd.DataFrame): DataFrame representing option chain data for fixed quote date and dte.
        active_trades (set[str]): The set of currently active option ids.
        dte (int): Number of days to expiration of the options. This value is the same for all options in the daily_dte_options DataFrame.
        test_params (dict): Dictionary of backtest parameters. See config.json5.
        model_params (dict): A dictionary of model parameters; see data_utils.get_model_params().

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
        option_values = stochastic_option(test_params, model_params, dte, strike_prices, n_calls, n_puts)

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

def track_trades(active_trades, executed_trades, option_price_lookup, quote_date, test_params):
    """
    Function used for closing the appropriate trades each day (only for stat-arb).

    Parameters:
        active_trades (set[str]): The set of all currently active option ids.
        executed_trades (list[dict]): The list containing the data for all executed trades as dictionaries.
        option_price_lookup (pd.DataFrame): A reindexed version of the complete option chain to find option prices more quickly.
        quote_date (pd.Timestamp): The current day on which we are tracking the previously executed trades.
        test_params (dict): Dictionary of backtest parameters. See config.json5.

    Returns:
        set[str]: The set of active option ids from which the closed positions have been removed.
        list[dict]: The list of dictionaries for the executed trades, in which we update the closed positions.
    """
    take_win, stop_loss = test_params["take_win"], test_params["stop_loss"]

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

def hedge(daily_options, option_price_lookup, stock_closing, test_params, model_params, dS_percent=0.02):
    """
    This function implements the delta-hedging strategy. We select a set of options to trade and evaluate their deltas through a discrete derivative.
    We then trade each option together with -delta shares of underlying stock. The difference in value of the option-stock position from one day to the
    next is then calculated (ideally, this should be as close to 0 as possible). The results are stored in a list of dictionaries, one for each hedge trade.

    Parameters:
        daily_options (pd.DataFrame): Subset of the complete option chain, containing data for a given trading day.
        option_price_lookup (pd.DataFrame): A reindexed version of the complete option chain to find option prices more quickly.
        stock_closing (pd.DataFrame): A DataFrame containing daily closing prices of the underlying stock.
        test_params (dict): Dictionary of backtest parameters. See config.json5.
        model_params (dict): A dictionary of model parameters; see the function data_utils.get_model_params().
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
        predicted_prices = stochastic_option(test_params, model_params, dte, strike_prices, n_calls=1, n_puts=1)
        predicted_prices_bumped = stochastic_option(test_params, model_params_bumped, dte, strike_prices, n_calls=1, n_puts=1)

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

def analyze_results(executed_trades_df, total_options_evaluated, test_params, option_multiplier=100):
    """
    Function that saves the executed trades as a .csv file, prints some test results to
    the terminal, and saves a plot of the cumulative profit and percentage return of the test.

    Parameters:
        executed_trades_df (pd.DataFrame): The DataFrame of all executed trades in the test interval.
        total_options_evaluated (int): Number of options evaluated over the entire test (not necessarily executed trades).
        test_params (dict): Dictionary of backtest parameters. See config.json5.
        option_multiplier (int): Number of shares traded under one option contract (100 for stocks).
                                 The option prices calculated by us and found in option chains correspond to only one share.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    results_save_path = os.path.join(current_dir, "..", "..", "results")

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    test_type, ticker, stock_model = test_params["test_type"], test_params["ticker"], test_params["stock_model"]
    start_date, end_date = test_params["start_date"], test_params["end_date"]

    # Save executed trades to a .csv file
    test_id_string = f"{test_type}_{ticker}_{stock_model}_{start_date}_{end_date}"
    executed_trades_df.to_csv(os.path.join(results_save_path, test_id_string) + ".csv", index=False)

    profit_data = executed_trades_df.groupby("close_date", as_index=False)["realized_profit"].sum()
    profit_data = profit_data.sort_values("close_date").reset_index(drop=True)
    profit_data["realized_profit"] *= option_multiplier
    profit_data["cumulative_profit"] = profit_data["realized_profit"].cumsum()

    total_profit = profit_data["cumulative_profit"].iloc[-1]
    total_capital_required = executed_trades_df["option_price_open"].sum() * option_multiplier
    return_scale = 100 / total_capital_required

    print(f"Number of options evaluated: {total_options_evaluated}")
    print(f"Number of trades executed: {len(executed_trades_df)}")
    print(f"Percentage of evaluated options which were traded: {(len(executed_trades_df) / total_options_evaluated * 100):.2f}%")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Total capital required: ${total_capital_required:.2f}")
    print(f"Total percentage return: {(total_profit * return_scale):.2f}%")

    # Plot the results of the backtest
    fig, ax1 = plt.subplots()

    ax1.plot(profit_data["close_date"], profit_data["cumulative_profit"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative profit ($)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(ymin * return_scale, ymax * return_scale)
    ax2.set_ylabel("Percentage return (%)")

    plt.title(f"Profit results: {test_type} {ticker} {stock_model}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_save_path, test_id_string) + ".png", bbox_inches="tight")
