from dataclasses import dataclass
import numpy as np
from math import exp, sqrt
from .black_scholes import OptionParams

# ---- Monte Carlo parameters ----
@dataclass(frozen=True)
class MCParams:
    n_sims: int = 100_000   # nombre de simulations
    seed: int = 42          # graine aléatoire pour reproductibilité

# ---- Simulation du sous-jacent ----
def simulate_paths(p: OptionParams, n_sims: int, seed: int = 42):
    """Simule les valeurs finales S_T du sous-jacent sous la mesure risque-neutre."""
    np.random.seed(seed)
    Z = np.random.randn(n_sims)
    drift = (p.r - p.q - 0.5 * p.sigma**2) * p.T
    diffusion = p.sigma * sqrt(p.T) * Z
    ST = p.S * np.exp(drift + diffusion)
    return ST

# ---- Monte Carlo pricing ----
def monte_carlo_price(p: OptionParams, kind: str, n_sims: int = 100_000, seed: int = 42):
    """Prix d'une option européenne (call/put) par Monte Carlo."""
    ST = simulate_paths(p, n_sims, seed)
    if kind == "call":
        payoffs = np.maximum(ST - p.K, 0)
    else:
        payoffs = np.maximum(p.K - ST, 0)
    discounted_payoffs = exp(-p.r * p.T) * payoffs
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs) / sqrt(n_sims)
    return price, stderr

# ---- Test rapide ----
if __name__ == "__main__":
    p = OptionParams(S=100, K=100, r=0.05, q=0.02, sigma=0.2, T=1)

    call_price, call_err = monte_carlo_price(p, "call", n_sims=100_000)
    put_price, put_err = monte_carlo_price(p, "put", n_sims=100_000)

    print("---- Monte Carlo Results ----")
    print(f"Call Price: {call_price:.4f} ± {call_err:.4f}")
    print(f"Put  Price: {put_price:.4f} ± {put_err:.4f}")
