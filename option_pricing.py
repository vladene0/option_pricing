import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import njit

# Work in progress; needs more commenting!

# Time step corresponding to 1 day, expressed in years
dt = 1/365

# Class used to simulate stock price evolution using Geometric Brownian Motion
class StockGBM:
    def __init__(self, n_paths, n_days, initial_price, drift, volatility):
        self.n_paths = n_paths              # We simulate stock price evolution on "n_paths" paths for stochastic pricing of options
        self.n_days = n_days                # Number of days to expiry for the option
        self.initial_price = initial_price  # Initial stock price
        self.drift = drift                  # Drift of the stock
        self.volatility = volatility        # Volatility of the stock

    def simulate(self):
        # Independent random samples from the normal distribution used to build stock price path
        eps = np.random.normal(size=(self.n_paths, self.n_days))
        log_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * eps * np.sqrt(dt)

        # This 2-d array contains "n_paths" simulated paths for the stock price evolution, each with "n_days" time steps, plus the initial step
        prices = np.zeros((self.n_paths, self.n_days + 1))
        prices[:, 0] = self.initial_price
        prices[:, 1:] = self.initial_price * np.exp(np.cumsum(log_returns, axis=1))

        return prices

# Class used to simulate stock price evolution using the Heston Model (Stochastic Volatility)
class StockHeston:
    def __init__(self, n_paths, n_days, initial_price, drift, mean_variance, var_return_rate, vol_of_variance, correlation):
        self.n_paths = n_paths
        self.n_days = n_days
        self.initial_price = initial_price
        self.drift = drift
        self.mean_variance = mean_variance          # Mean value to which the variance (square of volatility) returns during the simulation
        self.var_return_rate = var_return_rate      # The rate at which the variance returns to the mean
        self.vol_of_variance = vol_of_variance      # The volatility of the variance
        self.correlation = correlation              # The correlation between the random processes of the stock and the variance

    def simulate(self):
        # Generate random samples from the normal distribution for the stock and variance and then correlate them
        eps1 = np.random.normal(size=(self.n_paths, self.n_days))
        eps2 = np.random.normal(size=(self.n_paths, self.n_days))
        eps1_corr, eps2_corr = eps1, self.correlation * eps1 + np.sqrt(1 - self.correlation**2) * eps2

        # Create 2-d numpy array to store the values of the variance across the time steps, for each of the simulation paths
        variances = np.zeros((self.n_paths, self.n_days + 1))
        variances[:, 0] = self.mean_variance

        # Calculate variances independently
        for j in range(1, self.n_days + 1):
            variances[:, j] = (variances[:, j-1] + self.var_return_rate * (self.mean_variance - variances[:, j-1]) * dt +
                               self.vol_of_variance * np.sqrt(variances[:, j-1]) * eps2_corr[:, j-1] * np.sqrt(dt))
            
            # Make sure the variance is never negative (since it is the square of the volatility, it has to be positive)
            variances[:, j] = np.maximum(variances[:, j], 0)

        # Array to store prices for each simulation path
        prices = np.zeros((self.n_paths, self.n_days + 1))
        prices[:, 0] = self.initial_price

        # Compute price evolution in exponential form
        log_returns = (self.drift - 0.5 * variances[:, :-1]) * dt + np.sqrt(variances[:, :-1]) * eps1_corr * np.sqrt(dt)
        prices[:, 1:] = self.initial_price * np.exp(np.cumsum(log_returns, axis=1))

        return prices

@njit
def quadratic_fit(x, y):
    """
    Quadratic regression function for the Least Squares Monte Carlo algorithm, based on the matrix formulation of the minimization problem.

    Parameters:
        x (1-d numpy array): x-coordinates of points we are fitting.
        y (1-d numpy array): y-coordinates of points we are fitting.

    Returns:
        1-d numpy array: Values of the fitted function evaluated at points x, according to LSMC.
    """
    # Number of points to fit
    n = len(x)

    # Prepare sums of various products of x and y values
    Sx, Sx2, Sx3, Sx4, Sy, Sxy, Sx2y = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute sums initialized above
    for i in range(n):
        xi = x[i]
        yi = y[i]
        x2 = xi * xi
        Sx += xi
        Sx2 += x2
        Sx3 += x2 * xi
        Sx4 += x2 * x2
        Sy += yi
        Sxy += xi * yi
        Sx2y += x2 * yi

    # The minimization problem can be expressed as a system of linear equations, A*u=v, where A is a 3x3 matrix containing the sums above,
    # u is a 3x1 vector containing the unknown fit parameters, and v is another 3x1 vector built from the sums above
    detA = Sx4*(Sx2*n - Sx*Sx) - Sx3*(Sx3*n - Sx2*Sx) + Sx2*(Sx3*Sx - Sx2*Sx2)
    
    # If the system of equations can't be solved, just return the original y values
    if abs(detA) < 1e-6:
        return y

    # Solve system of equations using Cramer's rule
    a = (Sx2y*(Sx2*n - Sx*Sx) - Sxy*(Sx3*n - Sx2*Sx) + Sy*(Sx3*Sx - Sx2*Sx2)) / detA
    b = (Sx4*(Sxy*n - Sx*Sy) - Sx3*(Sx2y*n - Sx2*Sy) + Sx2*(Sx2y*Sx - Sx2*Sxy)) / detA
    c = (Sx4*(Sx2*Sy - Sxy*Sx) - Sx3*(Sx3*Sy - Sx2y*Sx) + Sx2*(Sx3*Sxy - Sx2y*Sx2)) / detA

    # Return values of fitted quadratic function evaluated at coordinates x
    return a * x**2 + b * x + c

# Class used for American-style option pricing with the Least Squares Monte Carlo method
class OptionLSMC:
    def __init__(self, n_paths, n_days, strike_prices, rfr, n_calls, n_puts):
        self.n_paths = n_paths              # Number of simulation paths for the stock price evolution
        self.n_days = n_days                # Number of days to expiry for the option
        self.strike_prices = strike_prices  # 1-d numpy array of strike prices to evaluate options at
        self.rfr = rfr                      # Risk free interest rate, used for present-day discounting
        self.n_calls = n_calls              # Number of call options to evaluate; first n_calls values in strike_prices are strikes for the call options
        self.n_puts = n_puts                # Number of put options to evaluate; the last n_puts values in strike_prices are strikes for the put options

    # Function to evaluate option prices for a batch of "strike_prices" (1-d array) and option types (n_calls, n_puts are integers);
    # The first "n_calls" of the "strike_prices" are call strike prices, the last "n_puts" are put strike prices;
    # The "stock_prices" matrix comes from the simulation in the Stock classes;
    # This method is defined as static in order to be compatible with Numba, for faster run time
    @staticmethod
    @njit
    def lsmc_batch(stock_prices, strike_prices, rfr, n_calls, n_puts, dt):
        n_paths, n_days_full = stock_prices.shape
        n_days = n_days_full - 1
        n_strikes = n_calls + n_puts

        # Create "cash_flow" array used for computing option values in LSMC; last column of the "cash_flow" array is initialised as the payoff at expiration
        # The last axis corresponds to the different strike prices we evaluate as a batch
        cash_flow = np.zeros((n_paths, n_days_full, n_strikes))
        cash_flow[:, -1, :n_calls] = np.maximum(stock_prices[:, -1, None] - strike_prices[None, :n_calls], 0)   # For call options
        cash_flow[:, -1, n_calls:] = np.maximum(strike_prices[None, n_calls:] - stock_prices[:, -1, None], 0)   # For put options

        # Starting from the second to last column of the simulated stock prices array, iterate backwards over the time steps
        for j in range(n_days - 1, -1, -1):

            # Iterate over the strike prices (last axis in "cash_flow")
            for k in range(n_strikes):

                # At each time step and for each strike price, compute the 1-d array of payoffs for the simulation paths; differentiate between calls and puts
                payoffs = np.maximum(stock_prices[:, j] - strike_prices[k], 0) if k < n_calls else np.maximum(strike_prices[k] - stock_prices[:, j], 0)

                # Select the indices for the paths which are "in the money" (positive payoff)
                itm_indices = np.where(payoffs > 0)[0]
                itm_count = len(itm_indices)

                # If there are no itm paths, skip to the next iteration
                if itm_count == 0:
                    continue

                # For the itm paths, calculate the 1-d array of expected future cash flows discounted to the current time step
                # Use a for loop as Numba doesn't allow more complicated indexing of arrays
                discounted_cash_flow = np.zeros(itm_count)
                for index, i in enumerate(itm_indices):
                    max_col_index = np.argmax(cash_flow[i, :, k])
                    discounted_cash_flow[index] = cash_flow[i, max_col_index, k] * np.exp(-rfr * (max_col_index - j) * dt)

                # Predict the value of continuing holding the option with a quadratic fit (assuming we have at least 3 data points to fit)
                continuation_values = np.zeros(n_paths)
                continuation_values[itm_indices] = quadratic_fit(stock_prices[itm_indices, j], discounted_cash_flow) if itm_count >= 3 else discounted_cash_flow

                # Find the paths for which exercising the option is the best choice at the current moment; store payoffs in the cash flow matrix
                # Use a for loop as Numba doesn't allow more complicated indexing of arrays
                for i in itm_indices:
                    if payoffs[i] > continuation_values[i]:
                        cash_flow[i, :, k] = 0.0
                        cash_flow[i, j, k] = payoffs[i]

        # Exponential factors used for present-day discounting of the final cash flow matrix
        discount_factors = np.exp(-rfr * np.arange(n_days_full) * dt)

        # For each of the strike prices, calculate the value of the corresponding option by taking
        # the present-day discounted final cash flow, averaged over the simulation paths;
        # Use for loops as Numba doesn't allow summing over multiple axes in a Numpy array
        option_values = np.zeros(n_strikes)
        for k in range(n_strikes):
            for i in range(n_paths):
                for j in range(n_days_full):
                    option_values[k] += cash_flow[i, j, k] * discount_factors[j]

        option_values /= n_paths

        # Return 1-d array of option values for each of the input strike prices and option types
        return option_values

    # Small helper to simplify the function call for option evaluation, sending the class variables to the Numba evaluation function
    def evaluate(self, stock_prices):
        return OptionLSMC.lsmc_batch(stock_prices, self.strike_prices, self.rfr, self.n_calls, self.n_puts, dt)

def stochastic_option(n_paths=250, n_days=30,
                      initial_price=100, drift=0.05,
                      volatility=0.1,
                      mean_variance=0.2**2, var_return_rate=0.4, vol_of_variance=0.2, correlation=0.0,
                      strike_prices=np.array([100]), rfr=0.05, n_calls=1, n_puts=0,
                      stock_model="gbm", option_model="lsmc", n_simulations=100):
    """
    Function for batch evaluation of the prices of options at different strikes (same quote day and dte)
    with the LSMC algorithm, using a stochastic model for the stock price evolution.

    Parameters:
        strike_prices (1-d numpy array): Array of batch strike prices at which to evaluate options.
                                         First n_calls elements are call strikes, the last n_puts are put strikes.
        n_calls (int): Number of call options to evaluate.
        n_puts (int): Number of put options to evaluate.
        stock_model (string): The model used for stock price simulation. Currently supported are "gbm", "heston".

    Returns:
        1-d numpy array: Predicted values of each option for the corresponding strike prices in the batch strike_prices array.
    """
    if stock_model == "gbm":
        s_model = StockGBM(n_paths, n_days, initial_price, drift, volatility)
    elif stock_model == "heston":
        s_model = StockHeston(n_paths, n_days, initial_price, drift, mean_variance, var_return_rate, vol_of_variance, correlation)
    else:
        raise TypeError("Invalid stock model!")

    if option_model == "lsmc":
        o_model = OptionLSMC(n_paths, n_days, strike_prices, rfr, n_calls, n_puts)
    else:
        raise TypeError("Invalid option model!")

    # Repeat stock price simulation and option evaluation "n_simulations" times in order to use the average option value as the "true" value
    price_results = []
    for _ in range(n_simulations):
        simulated_stock_prices = s_model.simulate()
        option_values = o_model.evaluate(simulated_stock_prices)

        # This is a list which contains "n_simulations" lists of calculated option values at each strike price
        price_results.append(option_values)

    # Obtain 1-d array with length "n_calls + n_puts"; first "n_calls" elements are average call prices, the others are put prices, each at their respective strikes
    average_prices = np.mean(price_results, axis=0)

    return average_prices

# Function used for calculating the price of an European call or put option with the Black-Scholes model
# The European option price is used as a rough approximation to the American option price for validation of the stochastic model
def black_scholes_option(S=100, K=100, sigma=0.1, rfr=0.05, t=30/365, option_type="call"):
    d1 = (np.log(S/K) + (rfr + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-rfr*t) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-rfr*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        print("Invalid option type!")

    return price
