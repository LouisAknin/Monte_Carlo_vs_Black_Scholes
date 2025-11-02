import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pricer_project.models.black_scholes import OptionParams, black_scholes_price, delta, gamma, vega, theta, rho
from pricer_project.models.monte_carlo import monte_carlo_price


st.set_page_config(page_title="Option Pricer", page_icon="📈", layout="centered")
st.title("Option Pricer — Black–Scholes & Monte Carlo")

# --- Sidebar inputs ---
st.sidebar.header("Parameters")
S = st.sidebar.number_input("Spot S", min_value=0.0, value=100.0, step=1.0, format="%.4f")
K = st.sidebar.number_input("Strike K", min_value=0.0, value=100.0, step=1.0, format="%.4f")
r = st.sidebar.number_input("Risk-free rate r (cont.)", value=0.05, step=0.005, format="%.4f")
q = st.sidebar.number_input("Dividend yield q (cont.)", value=0.02, step=0.005, format="%.4f")
sigma = st.sidebar.number_input("Volatility σ (annual)", min_value=0.0, value=0.2, step=0.01, format="%.4f")
T = st.sidebar.number_input("Maturity T (years)", min_value=0.0, value=1.0, step=0.25, format="%.4f")
kind = st.sidebar.selectbox("Option type", options=["call", "put"])

st.sidebar.header("Monte Carlo")
n_sims = st.sidebar.number_input("Number of simulations", min_value=100, value=100_000, step=1_000)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

params = OptionParams(S=S, K=K, r=r, q=q, sigma=sigma, T=T)

# --- Analytical price ---
st.subheader("Analytical — Black–Scholes")
bs_price = black_scholes_price(params, kind)
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"Price ({kind})", value=f"{bs_price:.6f}")
with col2:
    st.caption("Model: GBM, r–q constant, European payoff, risk-neutral measure.")

# --- Greeks ---
st.subheader("Greeks (Analytical)")
g_delta = delta(params, kind)
g_gamma = gamma(params)
g_vega  = vega(params)
g_theta = theta(params, kind)
g_rho   = rho(params, kind)

st.write(
    f"**Delta**: {g_delta:.6f}  |  "
    f"**Gamma**: {g_gamma:.6f}  |  "
    f"**Vega**: {g_vega:.6f}  |  "
    f"**Theta**: {g_theta:.6f}  |  "
    f"**Rho**: {g_rho:.6f}"
)

# --- Monte Carlo price ---
st.subheader("Monte Carlo")
mc_price, mc_err = monte_carlo_price(params, kind, n_sims=int(n_sims), seed=int(seed))
st.write(f"**MC Price**: {mc_price:.6f}  ±  {mc_err:.6f} (1σ)")

diff = mc_price - bs_price
st.write(f"**Difference (MC − BS)**: {diff:.6f}")

# --- Convergence demo (optional plot) ---
with st.expander("Show convergence plot (Monte Carlo → Black–Scholes)"):
    max_power = st.slider("Max log10(n_sims)", min_value=2, max_value=6, value=5, step=1)
    grid = np.unique((np.logspace(2, max_power, num=max_power*5)).astype(int))
    prices = []
    errs = []
    for n in grid:
        p, e = monte_carlo_price(params, kind, n_sims=int(n), seed=int(seed))
        prices.append(p)
        errs.append(e)
    prices = np.array(prices)
    errs = np.array(errs)

    fig = plt.figure(figsize=(7, 4))
    plt.axhline(bs_price, linestyle="--", label="Black–Scholes", linewidth=1.5)
    plt.errorbar(grid, prices, yerr=errs, fmt="o-", markersize=3, linewidth=1, label="Monte Carlo")
    plt.xscale("log")
    plt.xlabel("Number of simulations (log scale)")
    plt.ylabel("Estimated price")
    plt.title(f"Convergence ({kind.upper()})")
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(fig)