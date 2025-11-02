"""from pricer_project.utils.stats_analysis import plot_convergence
from pricer_project.models.black_scholes import OptionParams

if __name__ == "__main__":
    p = OptionParams(S=100, K=100, r=0.05, q=0.02, sigma=0.2, T=1)
    plot_convergence(p, "call", [100, 500, 2000, 10000, 50000, 100000])

"""

# main.py
import os
import subprocess

def run_streamlit_app():
    """Lance l'application Streamlit du projet."""
    # Chemin vers ton app Streamlit
    app_path = os.path.join("pricer_project", "interface", "app.py")

    # Lancer Streamlit avec le même interpréteur Python
    subprocess.run(["streamlit", "run", app_path], check=True)

if __name__ == "__main__":
    run_streamlit_app()
