import numpy as np
from config.constants import JUPITER_A, K1, K2, K3


def get_secular_series(a):
    """Calculates S1 and S2 series expansions from Hirayama (1918)"""
    alpha = a / JUPITER_A
    # S1 series expansion [cite: 172]
    s1 = 1 + (1.875 * alpha**2) + (2.734 * alpha**4)
    # S2 series expansion [cite: 198]
    s2 = 1 + (1.75 * alpha**2) + (2.461 * alpha**4)
    return s1, s2


def calculate_dv(ast1, ast2):
    """
    Implements the Zappalà distance metric.
    Converts orbital differences into a velocity (m/s).
    """
    da = ast1["a"] - ast2["a"]
    de = ast1["e"] - ast2["e"]
    di = ast1["sini"] - ast2["sini"]

    a_bar = (ast1["a"] + ast2["a"]) / 2
    # Mean motion n calculation (simplified gravitational units)
    n = np.sqrt(1 / a_bar**3)

    # Distance formula components
    term1 = K1 * (da / a_bar) ** 2
    term2 = K2 * (de) ** 2
    term3 = K3 * (di) ** 2

    return n * a_bar * np.sqrt(term1 + term2 + term3)
