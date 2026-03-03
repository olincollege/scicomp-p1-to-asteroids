# Constants from Hirayama (1918)
# config/constants.py

# Weights for the Zappalà distance metric (standardized in 1994)
K1 = 1.25  # semi-major axis weight
K2 = 2.0  # eccentricity weight
K3 = 2.0  # inclination weight

# Jupiter's semi-major axis (a') from Hirayama (1918)
JUPITER_A = 5.20336  # [cite: 174]

# Historical Family Benchmarks (Hirayama 1918)
# Used to verify if your engine finds the original core members
HIRAYAMA_BENCHMARKS = {
    "Koronis": {"seed": 158, "n_range": (720, 740), "i_max": 4.0},  # [cite: 43, 65]
    "Eos": {"seed": 221, "n_range": (671, 682), "i_range": (8.6, 11.3)},  # [cite: 163]
    "Themis": {"seed": 24, "n_range": (622, 653), "i_range": (0.3, 2.7)},  # [cite: 165]
}

# Velocity cutoff (m/s) - Initial starting point to reach 95% completeness
V_CUTOFF = 50.0
