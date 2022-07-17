from unittest import result
import numpy as np
from math import exp, sqrt
import random
import matplotlib.pyplot as plt
import statistics
import os
from concurrent import futures


def calc_price(S, sigma, rfr, delta, T, N, eps):
    # S: 原資産
    # sigma: ボラティリティ
    # rfr: 安全利回り
    # delta: 配当利回り
    # t: 期間
    # n: 分割数
    # eps: N(0,1)に従う乱数
    # 参考: https://www.op-labo.com/simulation/simulation.html
    price = S * exp(
        (rfr - delta - (sigma**2 / 2)) * T / N + sigma * sqrt(T / N) * eps
    )
    return price


def monte_carlo_with_prallel(split_count, steps=100000):
    S = 135
    rfr = 0.235  # (日本国債10年の年利回り)
    delta = 0.2142
    T = 1
    N = 365
    result_price = []
    for test in range(0, split_count):
        steps = 100000
        monte_calro_price = 0
        print(test)
        with futures.ThreadPoolExecutor(max_workers=500 * os.cpu_count()) as executor:
            for n in range(0, steps):
                sigma = random.normalvariate(0, 10)  # 変動率の標準偏差を年率にしたもの。正規分布に従うらしい
                eps = random.normalvariate(0, 1)
                monte_calro_price += (
                    executor.submit(calc_price, S, sigma, rfr, delta, T, N, eps)
                ).result()
        # mean = statistics.mean([f.result() for f in future_list])
        mean = monte_calro_price / steps
        result_price.append(mean)
        S = mean

    plt.plot(result_price)
    plt.show()


def monte_carlo_with_single(split_count, steps=100000):
    S = 334
    rfr = 0.235  # (日本国債10年の年利回り)
    delta = 0.2142
    T = 1
    N = split_count
    result_price = []
    result_price.append(S)
    for test in range(0, split_count):
        monte_calro_price = 0
        for n in range(0, steps):
            sigma = random.normalvariate(0, 10)  # 変動率の標準偏差を年率にしたもの。正規分布に従うらしい
            eps = random.normalvariate(0, 1)
            monte_calro_price += calc_price(S, sigma, rfr, delta, T, N, eps)

        # mean = statistics.mean([f.result() for f in future_list])
        mean = monte_calro_price / steps
        result_price.append(mean)
        print(str(S) + " -> " + str(mean))
        S = mean
    plt.plot(result_price)
    plt.show()


monte_carlo_with_single(180)
