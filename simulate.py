import numpy as np
from scipy import stats


class BlackScholes(object):
    def __init__(self, stock_price, strike_price, expiry, risk_free_rate, sigma):
        self.s = stock_price
        self.e = strike_price
        self.t = expiry  # year
        self.r = risk_free_rate
        self.sigma = sigma  # volatility
        self.d1 = (
            np.log(self.s / self.e) + (self.r + self.sigma**2 / 2) * self.t
        ) / (self.sigma * self.t ** (1 / 2))
        self.d2 = (
            np.log(self.s / self.e) + (self.r - self.sigma**2 / 2) * self.t
        ) / (self.sigma * self.t ** (1 / 2))

    def call_price(self):
        call_price = self.s * stats.norm.cdf(self.d1) - self.e * np.exp(
            -self.r * self.t
        ) * stats.norm.cdf(self.d2)
        return call_price

    def put_price(self):
        put_price = -self.s * stats.norm.cdf(-self.d1) + self.e * np.exp(
            -self.r * self.t
        ) * stats.norm.cdf(-self.d2)
        return put_price

    def monte_carlo_call(self, iterations):
        np.random.seed(seed=0)
        option_data = np.zeros([iterations, 2])
        rands = np.random.normal(0, 1, [1, iterations])
        stock_price = self.s * np.exp(
            self.t * (self.r - 0.5 * self.sigma**2)
            + self.sigma * self.t ** (1 / 2) * rands
        )
        option_data[:, 1] = stock_price - self.e
        average = np.sum(np.amax(option_data, axis=1)) / iterations
        return average * np.exp(-1 * self.r * self.t)

    def monte_carlo_put(self, iterations):
        np.random.seed(seed=0)
        option_data = np.zeros([iterations, 2])
        rands = np.random.normal(0, 1, [1, iterations])
        stock_price = self.s * np.exp(
            self.t * (self.r - 0.5 * self.sigma**2)
            + self.sigma * self.t ** (1 / 2) * rands
        )
        option_data[:, 1] = self.e - stock_price
        average = np.sum(np.amax(option_data, axis=1)) / iterations
        return average * np.exp(-1 * self.r * self.t)


if __name__ == "__main__":
    stock_price = 2000
    strike_price = 3000
    expiry = 1
    risk_free_rate = 0.2
    sigma = 5

    black_scholes = BlackScholes(
        stock_price, strike_price, expiry, risk_free_rate, sigma
    )
    # print("----理論値----\n")
    # print("call options value is :", black_scholes.call_price())
    # print("put options value is :", black_scholes.put_price(), "\n")
    # print("----理論値----\n\n")
    # 1783.4595785128108

    iteratoins = 10000000
    print("----シミュレーション----\n")
    print("call options value is :", black_scholes.monte_carlo_call(iteratoins))
    # 1386.4309824340962
    print("put options value is :", black_scholes.monte_carlo_put(iteratoins), "\n")
    print("----シミュレーション----")
