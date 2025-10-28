import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from joblib import Parallel, delayed
import warnings

# Time step corresponding to 1 day, expressed in years
dt = 1/365

# Class used to simulate stock price evolution using Geometric Brownian Motion
class StockGBM:
    def __init__(self, n_paths, n_days, initial_price, drift, volatility):
        self.n_paths = n_paths
        self.n_days = n_days
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility

    def simulate(self):
        eps = np.random.normal(size = (self.n_paths, self.n_days - 1)) # Independent random samples from the normal distribution used to build stock price path
        log_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * eps * np.sqrt(dt)

        # This 2-d array contains "n_paths" simulated paths for the stock price evolution, each with "n_days" time steps
        self.prices = np.zeros((self.n_paths, self.n_days))
        self.prices[:, 0] = self.initial_price
        self.prices[:, 1:] = self.initial_price * np.exp(np.cumsum(log_returns, axis = 1))
        return self.prices

    def plot_prices(self):
        for i in range(self.n_paths):
            plt.plot(range(self.n_days), self.prices[i, :])
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.show()

# Class used for American-style option pricing with the Least Squares Monte Carlo method
class OptionLSMC:
    def __init__(self, n_paths, n_days, strike_price, rfr, option_type):
        self.n_paths = n_paths
        self.n_days = n_days # Number of days to expiry for the option
        self.strike_price = strike_price
        self.rfr = rfr # Risk free interest rate, used for present-day discounting

        # Create numerical variable p to keep track of option type
        if option_type == "call":
            self.p = 0
        elif option_type == "put":
            self.p = 1
        else:
            print("Option type must be call or put!")

    def quadratic_fit(self, x, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            a, b, c = np.polyfit(x, y, 2)
        return a * x**2 + b * x + c

    def evaluate(self, stock_prices):
        cash_flow = np.zeros((self.n_paths, self.n_days))

        # Last column of the cash_flow array is initialised as the payoff at expiration
        cash_flow[:, -1] = np.maximum((-1)**self.p * (stock_prices[:, -1] - self.strike_price), 0)

        # Starting from the second to last column of the simulated stock prices array, iterate backwards over the time steps
        for j in range(self.n_days - 2, -1, -1):
            payoffs = np.maximum((-1)**self.p * (stock_prices[:, j] - self.strike_price), 0)
            itm_indices = np.where(payoffs > 0)[0] # At time step j, select the indices for paths which are "in the money" (positive payoff)

            if len(itm_indices) > 0:
                # Array to store the predicted values of continuing holding the option (in order to exercise later)
                continuation_values = np.zeros(self.n_paths)

                # For each ITM simulation path, get index of time step (column) where the cash flow is maximum
                max_cash_flow_column_indices = np.argmax(cash_flow[itm_indices, :], axis = 1)

                # Calculate array of cash flows discounted to the current time step
                discounted_cash_flow = cash_flow[itm_indices, max_cash_flow_column_indices] * np.exp(-self.rfr * (max_cash_flow_column_indices - j) * dt)

                # Predict the value of continuing holding the option with a quadratic fit
                continuation_values[itm_indices] = self.quadratic_fit(stock_prices[itm_indices, j], discounted_cash_flow)

                # Find the paths for which exercising the option is the best choice at current moment; store payoffs in the cash flow matrix
                exercise_indices = np.where(payoffs > continuation_values)[0]
                cash_flow[exercise_indices, :] = 0
                cash_flow[exercise_indices, j] = payoffs[exercise_indices]

        # Exponential factors used for present-day discounting of the final cash flow matrix
        discount_factors = np.exp(-self.rfr * np.arange(self.n_days) * dt)

        option_value = np.sum(cash_flow * discount_factors) / self.n_paths
        return option_value

# Function for evaluating the price of an option using a stochastic model
def stochastic_option(n_paths=250, n_days=30,
                      initial_price=100, drift=0.05, volatility=0.1,
                      strike_price=100, rfr=0.05, option_type="call",
                      stock_model="gbm", option_model="lsmc", n_simulations=100, plot_hist=False, print_details=True):

    # Define an inner function to run simulations in order to parallelize; this function needs a dummy argument
    def run_simulation(_):
        if stock_model == "gbm":
            s_model = StockGBM(n_paths, n_days, initial_price, drift, volatility)
        else:
            raise TypeError("Invalid stock model!")

        if option_model == "lsmc":
            o_model = OptionLSMC(n_paths, n_days, strike_price, rfr, option_type)
        else:
            raise TypeError("Invalid option model!")

        simulated_stock_prices = s_model.simulate()
        return o_model.evaluate(simulated_stock_prices)

    # Run the simulations on all available CPU cores in parallel
    price_results = Parallel(n_jobs=-1)(delayed(run_simulation)(i) for i in range(n_simulations))
    average_price = np.mean(price_results)
    
    if print_details == True:
        print(f"Stochastic model price: {average_price:.2f}")

    if plot_hist == True:
        plt.hist(price_results, edgecolor = "white")
        plt.xlabel("Calculated Price")
        plt.ylabel("Frequency")
        plt.show()

    return average_price

# Function used for calculating the price of an European option with the Black-Scholes model
# The European option price is used as a rough approximation to the American option price for validation of the stochastic model
def black_scholes_option(S=100, K=100, sigma=0.1, rfr=0.05, t=29/365, option_type="call"):
    d1 = (np.log(S/K) + (rfr + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-rfr*t) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-rfr*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        print("Invalid option type!")
    print(f"Black-Scholes model price: {price:.2f}")
    return price
