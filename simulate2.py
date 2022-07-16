import numpy as np
from math import exp, sqrt
import random
import matplotlib.pyplot as plt
import statistics


def price(S, sigma, rfr, delta, T, N, eps):
    # S: 原資産
    # sigma: ボラティリティ
    # rfr: 安全利回り
    # delta: 配当利回り
    # t: 期間
    # n: 分割数
    # eps: N(0,1)に従う乱数
    gBM = S * exp((rfr - delta - (sigma**2 / 2)) * T / N + sigma * sqrt(T / N) * eps)
    return gBM


S = 0.1
rfr = 0.235  # (日本国際10年の年利回り)
delta = 0.01
T = 1
N = 365
aaa = []

for test in range(1, 365):
    process = np.zeros(100000)  # 10,000ステップ
    process[0] = S
    for n in range(1, len(process)):
        sigma = random.normalvariate(0, 10)  # 変動率の標準偏差を年率にしたもの。正規分布に従うらしい
        eps = random.normalvariate(0, 1)
        process[n] = price(S, sigma, rfr, delta, T, N, eps)
    mean = statistics.mean(process)
    aaa.append(mean)
    S = mean

plt.plot(aaa)
plt.show()
