import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, n_paths, n_days, strike_price, rfr, option_type, stock_prices):
        self.n_paths = n_paths
        self.n_days = n_days # Number of days to expiry for the option
        self.strike_price = strike_price
        self.rfr = rfr # Risk free interest rate, used for present-day discounting
        self.stock_prices = stock_prices # Array of prices obtained from the Monte Carlo simulation

        # Create numerical variable p to keep track of option type
        if option_type == "call":
            self.p = 0
        elif option_type == "put":
            self.p = 1
        else:
            print("Option type must be call or put!")

    def quadratic_fit(self, x, y):
        a, b, c = np.polyfit(x, y, 2)
        return a * x**2 + b * x + c

    def evaluate_option_price(self):
        cash_flow = np.zeros((self.n_paths, self.n_days))

        # Last column of the cash_flow array is initialised as the payoff at expiration
        cash_flow[:, -1] = np.maximum((-1)**self.p * (self.stock_prices[:, -1] - self.strike_price), 0)

        # Starting from the second to last column of the simulated stock prices array, iterate backwards over the time steps
        for j in range(self.n_days - 2, -1, -1):
            payoffs = np.maximum((-1)**self.p * (self.stock_prices[:, j] - self.strike_price), 0)
            itm_indices = np.where(payoffs > 0)[0] # At time step j, select the indices for paths which are "in the money" (positive payoff)

            if len(itm_indices) > 0:
                # Array to store the predicted values of continuing holding the option (in order to exercise later)
                continuation_values = np.zeros(self.n_paths)

                # For each ITM simulation path, get index of time step (column) where the cash flow is maximum
                max_cash_flow_column_indices = np.argmax(cash_flow[itm_indices, :], axis = 1)

                # Calculate array of cash flows discounted to the current time step
                discounted_cash_flow = cash_flow[itm_indices, max_cash_flow_column_indices] * np.exp(-self.rfr * (max_cash_flow_column_indices - j) * dt)

                # Predict the value of continuing holding the option with a quadratic fit
                continuation_values[itm_indices] = self.quadratic_fit(self.stock_prices[itm_indices, j], discounted_cash_flow)

                # Find the paths for which exercising the option is the best choice at current moment; store payoffs in the cash flow matrix
                exercise_indices = np.where(payoffs > continuation_values)[0]
                cash_flow[exercise_indices, :] = 0
                cash_flow[exercise_indices, j] = payoffs[exercise_indices]

        # Exponential factors used for present-day discounting of the final cash flow matrix
        discount_factors = np.exp(-self.rfr * np.arange(self.n_days) * dt)

        option_value = np.sum(cash_flow * discount_factors) / self.n_paths
        return option_value

# Testing the model
n_paths = 1000
n_days = 30
initial_price = 100
drift = 0.3
volatility = 0.2
strike_price = 100
rfr = 0.05
option_type = "call"

price_results = []
for n_simulations in range(100):
    gbm = StockGBM(n_paths, n_days, initial_price, drift, volatility)
    simulated_prices = gbm.simulate()

    lsmc = OptionLSMC(n_paths, n_days, strike_price, rfr, option_type, simulated_prices)
    option_price = lsmc.evaluate_option_price()
    price_results.append(option_price)

plt.hist(price_results, edgecolor = "white")
plt.xlabel("Calculated Price")
plt.ylabel("Frequency")
plt.show()

"""
gbm = StockGBM(n_paths, n_days, initial_price, drift, volatility)
simulated_prices = gbm.simulate()
gbm.plot_prices()
"""