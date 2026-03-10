"""
constants.py
-----------------------------
File for the Physical and algorithmic constants for the Asteroid Families
identification project, based on Zappalà and Hirayama Papers.

Source References
----------
Zappalà, V., Cellino, A., Farinella, P., & Knezevic, Z. (1990).
    Asteroid Families. I. Identification by Hierarchical Clustering and
    Reliability Assessment. Astronomical Journal, 100, 2030.
    DOI: 10.1086/115658

Hirayama, K. (1919).
    Groups of Asteroids Probably of Common Origin.
    Astronomical Journal, 31, 185.
"""

# PHYSICAL CONSTANTS
GM_SUN = 1.32712440018e20  # Gravitational parameter of the Sun [m^3 / s^2]

GM_SUN_AU_DAY = (
    2.9591220828559093e-04  # Gravitational parameter of the Sun [AU^3 / day^2]
)

AU_TO_M = 1.495978707e11  # Astronomical Unit [m]

# AU to km
AU_TO_KM = 1.495978707e8  # AU to km

DAY_TO_SEC = 86400.0  # Days to seconds

# -----------------------------------------------------------------------------------------

# ZAPPALÀ METRIC WEIGHTS
# we will use the equation that is noted below
# d = n*a * sqrt(k_a*(da/a)^2 + k_e*(de)^2 + k_i*(d_sini)^2)

K_A = 5.0 / 4.0  # Weight on semi-major axis term which should = 1.25

K_E = 2.0  # Weight on eccentricity term

K_I = 2.0  # Weight on sine of inclination term

# -----------------------------------------------------------------------------------------

# MAIN BELT BOUNDARIES
# Filters the dataset to the main asteroid belt region [AU]
# Inner boundary: near Mars-crossing / nu6 secular resonance
# Outer boundary: near Jupiter 2:1 mean-motion resonance (Hecuba gap)

A_MIN = 2.0  # [AU] inner edge of main belt
A_MAX = 3.5  # [AU] outer edge of main belt

# Sub-belt boundaries (for local background estimation, Zappalà et al. 1990)
# Inner main belt
A_INNER_MIN = 2.0
A_INNER_MAX = 2.5

# Middle main belt
A_MIDDLE_MIN = 2.5
A_MIDDLE_MAX = 2.82

# Outer main belt
A_OUTER_MIN = 2.82
A_OUTER_MAX = 3.5

# Kirkwood gap positions [AU] (approximate, for reference / filtering)
KIRKWOOD_GAPS = {
    "3:1": 2.501,  # Jupiter 3:1 mean-motion resonance
    "5:2": 2.824,  # Jupiter 5:2 mean-motion resonance
    "7:3": 2.956,  # Jupiter 7:3 mean-motion resonance
    "2:1": 3.278,  # Jupiter 2:1 mean-motion resonance (Hecuba gap)
}

# Width to exclude around each Kirkwood gap [AU] (±)
KIRKWOOD_GAP_WIDTH = 0.02

# -----------------------------------------------------------------------------------------

# HCM CLUSTERING PARAMETERS

# Velocity cutoff sweep range [m/s]
# Zappalà et al. (1990) explore ~50–200 m/s
V_CUTOFF_MIN = 50.0  # [m/s]
V_CUTOFF_MAX = 200.0  # [m/s]
V_CUTOFF_STEP = 10.0  # [m/s] step size for sweep

# Default / recommended cutoff for main belt (Zappalà et al. 1990)
V_CUTOFF_DEFAULT = 100.0  # [m/s]

# Minimum number of members for a cluster to be considered a candidate family
MIN_FAMILY_SIZE = 5

# Linkage method for scipy hierarchical clustering
# Zappalà et al. (1990) use single-linkage (nearest-neighbor chaining)
LINKAGE_METHOD = "single"

# QUASI-RANDOM LEVEL (QRL) SIGNIFICANCE TEST
# Zappalà et al. (1990), Section 3
# Monte Carlo background estimation: shuffle proper elements, re-cluster,
# repeat N times to build background cluster-size distribution.

# Number of Monte Carlo iterations for QRL background estimation
N_MONTE_CARLO = 100

# Significance threshold: a real family must exceed this percentile
# of the background cluster-size distribution
QRL_SIGNIFICANCE_PERCENTILE = 99.0  # corresponds to ~99th percentile (2.3 sigma)

# -----------------------------------------------------------------------------------------

# ROBUSTNESS / PERTURBATION TEST
# Zappalà et al. (1990), Section 4
# Perturb proper elements within their uncertainties and re-cluster.
# Stable families retain most of their members across perturbations.

# Number of perturbation trials
N_PERTURBATION_TRIALS = 50

# Fraction of members that must survive perturbation to be called "robust"
ROBUSTNESS_THRESHOLD = 0.70  # 70% membership stability → robust family

# Typical proper element uncertainties (used if per-asteroid errors unavailable)
# From Knezevic & Milani (2003)
SIGMA_A_DEFAULT = 1e-4  # [AU]  typical uncertainty in proper a
SIGMA_E_DEFAULT = 1e-4  # [-]   typical uncertainty in proper e
SIGMA_SINI_DEFAULT = 1e-4  # [-]   typical uncertainty in proper sin(i)

# -----------------------------------------------------------------------------------------

# TARGET ASTEROID FAMILIES (benchmark, AstDys)
# Used for completeness / contamination evaluation.
# Names match AstDys family catalog identifiers.
# Approximate proper semi-major axis centroids [AU] listed for reference.

TARGET_FAMILIES = {
    "Flora": {"a_center": 2.20, "astdys_id": 402},
    "Vesta": {"a_center": 2.36, "astdys_id": 4},
    "Maria": {"a_center": 2.55, "astdys_id": 170},
    "Eunomia": {"a_center": 2.64, "astdys_id": 15},
    "Koronis": {"a_center": 2.87, "astdys_id": 158},
    "Eos": {"a_center": 3.01, "astdys_id": 221},
    "Themis": {"a_center": 3.13, "astdys_id": 24},
    "Hygiea": {"a_center": 3.14, "astdys_id": 10},
}

# Minimum completeness required for benchmark success (project requirement)
COMPLETENESS_THRESHOLD = 0.95

# -----------------------------------------------------------------------------------------

# VISUALIZATION

FIG_DPI = 150  # Figure DPI

SCATTER_SIZE = 0.5  # Point size in scatter plots

BACKGROUND_ALPHA = 0.15  # Alpha (transparency) for background asteroids

FAMILY_ALPHA = 0.7  # Alpha for family members

FAMILY_CMAP = "tab20"  # Colormap for families

# CLUSTERING COLUMNS
CLUSTERING_COLS = ["D_deg", "e_p", "sin_i_p"]

# TROJAN PREPROCESSING FILTERS
D_DEG_MIN = 0.5  # minimum libration amplitude [deg]
D_DEG_MAX = 34.0  # maximum libration amplitude [deg]
LYAPUNOV_MAX = 5  # max Lyapunov number — above this = chaotic

TARGET_FAMILIES = {
    "Eurybates": {"astdys_id": 3548},
    "Arkesilaos": {"astdys_id": 11351},
    "1996_RJ": {"astdys_id": 4709},
    "Ennomos": {"astdys_id": 4035},
    "Hektor": {"astdys_id": 624},
    "Mele": {"astdys_id": 9430},
    "Panthoos": {"astdys_id": 4060},
    "Cloanthus": {"astdys_id": 9799},
}
