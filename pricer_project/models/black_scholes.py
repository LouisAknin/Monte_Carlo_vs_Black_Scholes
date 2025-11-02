from dataclasses import dataclass
from math import exp, log, sqrt, erf, pi

# ---- Option parameters ----
@dataclass(frozen=True)
class OptionParams:
    S: float      # spot price
    K: float      # strike
    r: float      # risk-free rate
    q: float      # dividend yield
    sigma: float  # volatility
    T: float      # maturity

# ---- Normal distribution ----
def std_norm_cdf(x: float):
    """CDF de la loi normale standard."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def std_norm_pdf(x: float):
    """PDF de la loi normale standard."""
    return (1 / sqrt(2 * pi)) * exp(-0.5 * x**2)

# ---- Core formulas ----
def d1(p: OptionParams) -> float:
    return (log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * sqrt(p.T))

def d2(p: OptionParams, d1_val: float = None):
    if d1_val is None:
        d1_val = d1(p)
    return d1_val - p.sigma * sqrt(p.T)

def disc(rate: float, T: float):
    return exp(-rate * T)

# ---- Black-Scholes prices ----
def call_price(p: OptionParams):
    d1v = d1(p)
    d2v = d2(p, d1v)
    return p.S * exp(-p.q * p.T) * std_norm_cdf(d1v) - p.K * exp(-p.r * p.T) * std_norm_cdf(d2v)

def put_price(p: OptionParams):
    d1v = d1(p)
    d2v = d2(p, d1v)
    return p.K * exp(-p.r * p.T) * std_norm_cdf(-d2v) - p.S * exp(-p.q * p.T) * std_norm_cdf(-d1v)

def black_scholes_price(p: OptionParams, kind: str):
    return call_price(p) if kind == "call" else put_price(p)

# ---- Greeks ----
def delta(p: OptionParams, kind: str):
    d1v = d1(p)
    if kind == "call":
        return exp(-p.q * p.T) * std_norm_cdf(d1v)
    else:
        return -exp(-p.q * p.T) * std_norm_cdf(-d1v)

def gamma(p: OptionParams):
    d1v = d1(p)
    return exp(-p.q * p.T) * std_norm_pdf(d1v) / (p.S * p.sigma * sqrt(p.T))

def vega(p: OptionParams):
    d1v = d1(p)
    return p.S * exp(-p.q * p.T) * std_norm_pdf(d1v) * sqrt(p.T)

def theta(p: OptionParams, kind: str):
    d1v = d1(p)
    d2v = d2(p, d1v)
    part1 = - (p.S * exp(-p.q * p.T) * std_norm_pdf(d1v) * p.sigma) / (2 * sqrt(p.T))
    if kind == "call":
        part2 = -p.r * p.K * exp(-p.r * p.T) * std_norm_cdf(d2v)
        part3 = p.q * p.S * exp(-p.q * p.T) * std_norm_cdf(d1v)
        return part1 + part2 + part3
    else:
        part2 = p.r * p.K * exp(-p.r * p.T) * std_norm_cdf(-d2v)
        part3 = -p.q * p.S * exp(-p.q * p.T) * std_norm_cdf(-d1v)
        return part1 + part2 + part3

def rho(p: OptionParams, kind: str):
    d2v = d2(p)
    if kind == "call":
        return p.K * p.T * exp(-p.r * p.T) * std_norm_cdf(d2v)
    else:
        return -p.K * p.T * exp(-p.r * p.T) * std_norm_cdf(-d2v)

# ---- Test ----
if __name__ == "__main__":
    p = OptionParams(S=100, K=100, r=0.05, q=0.02, sigma=0.2, T=1)

    print("---- Prices ----")
    print("Call:", call_price(p))
    print("Put :", put_price(p))

    print("---- Greeks (Call) ----")
    print("Delta:", delta(p, "call"))
    print("Gamma:", gamma(p))
    print("Vega :", vega(p))
    print("Theta:", theta(p, "call"))
    print("Rho  :", rho(p, "call"))