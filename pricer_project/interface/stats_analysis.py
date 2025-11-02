import numpy as np
import matplotlib.pyplot as plt
from pricer_project.models.black_scholes import OptionParams, black_scholes_price
from pricer_project.models.monte_carlo import monte_carlo_price


def compare_mc_vs_bs(p: OptionParams, kind: str, n_sims_list: list[int]):
    bs_price = black_scholes_price(p, kind)
    mc_prices, mc_errors = [], []

    for n in n_sims_list:
        price, err = monte_carlo_price(p, kind, n_sims=n)
        mc_prices.append(price)
        mc_errors.append(err)

    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)
    biases = mc_prices - bs_price

    return bs_price, mc_prices, mc_errors, biases


def plot_convergence(p: OptionParams, kind: str, n_sims_list: list[int]):
    bs_price, mc_prices, mc_errors, biases = compare_mc_vs_bs(p, kind, n_sims_list)

    plt.figure(figsize=(8, 5))
    plt.axhline(bs_price, color="red", linestyle="--", label="Black–Scholes analytique")
    plt.errorbar(n_sims_list, mc_prices, yerr=mc_errors, fmt="o-", label="Monte Carlo")
    plt.xscale("log")
    plt.xlabel("Nombre de simulations (log)")
    plt.ylabel("Prix estimé")
    plt.title(f"Convergence Monte Carlo ({kind.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    p = OptionParams(S=100, K=100, r=0.05, q=0.02, sigma=0.2, T=1)
    n_list = [100, 500, 2000, 10000, 50000, 100000]
    plot_convergence(p, "call", n_list)
