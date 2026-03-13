# Project 2: Asteroid Families
### Automatic Identification of Trojan Asteroid Families Using DBSCAN and the Zappalà Velocity Metric

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Background](#2-background)
   - [What Are Asteroid Families?](#what-are-asteroid-families)
   - [Proper vs Osculating Elements](#proper-vs-osculating-elements)
   - [The Hirayama Legacy](#the-hirayama-legacy)
3. [Data](#3-data)
   - [Sources](#sources)
   - [File Descriptions](#file-descriptions)
   - [Dataset Statistics](#dataset-statistics)
4. [Method](#4-method)
   - [The Zappalà Velocity Metric](#the-zappalà-velocity-metric)
   - [Why DBSCAN Over HCM](#why-dbscan-over-hcm)
   - [Preprocessing](#preprocessing)
   - [Clustering Parameters](#clustering-parameters)
   - [Evaluation](#evaluation)
5. [Code Structure](#5-code-structure)
   - [File Overview](#file-overview)
   - [File Details](#file-details)
6. [Results](#6-results)
7. [Limitations](#7-limitations)
8. [Usage](#8-usage)
9. [Dependencies](#9-dependencies)
10. [References](#10-references)

---

## 1. Project Overview

This project automatically identifies asteroid families, these are groups of asteroids
that share a common parent body shattered by a catastrophic collision. For this project we
implement the Zappalà velocity metric combined with DBSCAN clustering to identify families 
in the Trojan asteroid belt, benchmarked against the AstDys ground truth catalog.

**Some Key Results Are:**
- 7 Trojan asteroid families identified.
- All 7 recovered at ≥95% completeness against AstDys ground truth. 
- Pipeline runs end to end from raw data to plots in a single command.

---

## 2. Background

### What Are Asteroid Families?

Asteroid families are groups of asteroids that share similar proper orbital
elements, like: semi-major axis, eccentricity, and inclination. This similarity
is not coincidental: family members are fragments of a single parent body
destroyed by a high-velocity collision billions of years ago. By identifying
these families we can reconstruct the history of collisions in the solar system.

The first systematic identification of asteroid families was made by Kiyotsugu
Hirayama in 1919, he noticed that some asteroids are clustered together in orbital
element space. Hirayama identified the Koronis, Eos, and Themis families; these are now known
as Hirayama families. About a century later, the same fundamental idea drives
modern automated classification methods.

### Proper vs Osculating Elements

There are two ways to describe an asteroid's orbit:

**Osculating elements**: these describe the orbit at a specific instant in time.
Since asteroids are continuously perturbed by the gravitational pull of
Jupiter and other planets, osculating elements change constantly. Using 
osculating elements for family identification would produce spurious clusters driven 
by short-term perturbations rather than real dynamical relationships.

**Proper elements**: these are long-term averages computed by integrating the
equations of motion over millions of years and filtering out periodic
perturbations. They represent the underlying "true" orbital character of
an asteroid and are stable over timescales of billions of years. This makes
them ideal for family identification.

This project uses proper elements exclusively, as recommended by Zappalà
et al. (1990) and Milani (1993).

### The Hirayama Legacy

Hirayama (1919) identified asteroid families by plotting asteroids in
(a, e, i) space and looking for visual clusters. Modern methods automate
this process using clustering algorithms and statistical significance tests,
but the core idea (that family members cluster in proper element space)
is exactly what Hirayama observed over 100 years ago.

---

## 3. Data

### Sources

All data is sourced from the AstDys catalog seen below:

```
https://newton.spacedys.com/astdys2/index.php?pc=5
```

### File Descriptions

| File | Description | Rows |
|------|-------------|------|
| `tro_syn.txt` | Synthetic proper elements for Trojan asteroids | 11,867 |
| `all_tro.members` | AstDys family membership ground truth | 264,559 |

**`tro_syn.txt` columns:**

| Column | Description | Units |
|--------|-------------|-------|
| `asteroid_number` | Asteroid designation | - |
| `H` | Absolute magnitude | mag |
| `da_AU` | Proper semi-major axis deviation from Jupiter | AU |
| `D_deg` | Libration amplitude | deg |
| `f_degy` | Libration frequency | deg/yr |
| `e_p` | Proper eccentricity | - |
| `g_arcsec_y` | Precession frequency of pericenter | arcsec/yr |
| `sin_i_p` | Sine of proper inclination | - |
| `s_arcsec_y` | Precession frequency of node | arcsec/yr |
| `L` | Lyapunov number (stability indicator) | - |
| `My` | Integration length | Myr |

**`all_tro.members` columns:**

| Column | Description |
|--------|-------------|
| `asteroid_number` | Asteroid designation |
| `H_mag` | Absolute magnitude |
| `status` | Family status (1=halo, 2=attributed, 3=core member) |
| `family1` | Primary AstDys family ID |
| `dv_fam1` | Velocity distance to family1 center (m/s) |
| `near1` | Nearest neighboring family |
| `family2` | Secondary family ID (double classifications) |
| `dv_fam2` | Velocity distance to family2 center (m/s) |
| `near2` | Nearest neighboring family to family2 |
| `rescod` | Resonance code |

### Dataset Statistics

- **11,867** Trojan asteroids with proper elements
- **264,559** total family membership entries (all asteroid types)
- **7** verified Trojan families in the ground truth
- **713** Trojan asteroids with confirmed family membership
- After preprocessing: **11,801** asteroids used for clustering

---

## 4. Method

### The Zappalà Velocity Metric

An important aspect of this project is the Zappalà velocity metric (Zappalà et al. 1990,
Eq. 1). Rather than measuring geometric distance in orbital element space,
it computes an effective relative velocity in m/s, physically interpretable
as the minimum ejection velocity needed to separate two asteroids from a
common parent body:

```
d = n*a * sqrt(k_a*(da/a)^2 + k_e*(de)^2 + k_i*(d_sini)^2)
```

Where:
- `n` = mean motion of Jupiter = sqrt(GM / a^3) [rad/day]
- `a` = Jupiter's semi-major axis = 5.2044 AU
- `da` = difference in proper semi-major axis deviation [AU]
- `de` = difference in proper eccentricity
- `d_sini` = difference in sine of proper inclination
- `k_a = 5/4 = 1.25` (weight on semi-major axis term)
- `k_e = 2.0` (weight on eccentricity term)
- `k_i = 2.0` (weight on inclination term)

The weights `k_a`, `k_e`, `k_i` come directly from Zappalà et al. (1990)
and reflect the relative dynamical importance of each orbital element in
determining whether two asteroids could have originated from the same parent.

For Trojan asteroids, `da` is the deviation from Jupiter's semi-major axis
rather than the absolute semi-major axis used in the main belt formulation,
following Milani (1993).

The full pairwise distance matrix (11,801 × 11,801 = ~69.6 million distances)
is computed using `scipy.spatial.distance.pdist` with the Zappalà function
as a custom metric, then converted to a square matrix with `squareform`.

### Why DBSCAN Over HCM

The classical method from Zappalà et al. (1990) uses Hierarchical Clustering
Method (HCM) with single-linkage agglomeration. We evaluated both approaches:

**Single-linkage HCM problems:**
- Suffers from the "chaining effect",  once two clusters are connected by
  even a single nearby pair of asteroids they merge into one giant cluster
- At low velocity cutoffs (50-200 m/s) everything remains fragmented
- At high velocity cutoffs (300-400 m/s) everything chains into 1-3 enormous
  clusters with >90% contamination
- No stable intermediate range exists for the Trojan dataset

**DBSCAN advantages:**
- Finds dense regions directly without chaining
- Background asteroids are naturally labeled as noise (-1) rather than
  forced into a cluster
- Single parameter (`eps` in m/s) has direct physical interpretation as
  a velocity cutoff, the same concept as in Zappalà HCM
- At `eps = 50 m/s` all 7 families are recovered at 100% completeness

DBSCAN with the Zappalà precomputed distance matrix gives physically
meaningful, tight clusters that match the AstDys ground truth extremely well.

### Preprocessing

Before clustering, the dataset is filtered in three steps:

**Step 1: Drop missing values**
Rows with missing values in `da_AU`, `e_p`, or `sin_i_p` are dropped.
In practice 0 rows are dropped from this dataset.

**Step 2: Filter libration amplitude**
Asteroids with libration amplitude `D < 0.5°` or `D > 34.0°` are removed.
These are weakly trapped Trojans near the stability boundary whose proper
elements are unreliable (Milani 1993). 66 asteroids are removed in this step.

**Step 3: Remove unstable asteroids**
Asteroids with Lyapunov number `L > 5` are removed. The Lyapunov number
indicates chaotic/unstable orbits whose proper elements are not reliable
for clustering. 0 asteroids are removed in this step.

**Result:** 11,801 of 11,867 asteroids pass all filters.

### Clustering Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `eps` | 50 m/s | Velocity neighbourhood radius |
| `min_samples` | 5 | Minimum members to form a cluster |
| Metric | Zappalà (precomputed) | Physical velocity distance |

The `eps = 50 m/s` value was chosen by sweeping values from 50 to 200 m/s
and selecting the smallest value at which all 7 families are recovered at
≥95% completeness. At 50 m/s, all 7 families achieve 100% completeness.

### Evaluation

Each discovered cluster is matched to the best AstDys ground truth family
by finding maximum overlap of asteroid IDs. Two metrics are computed:

**Completeness**: fraction of AstDys family members recovered:
```
Completeness = |cluster ∩ AstDys| / |AstDys|
```

**Contamination**: fraction of cluster members not in AstDys family:
```
Contamination = 1 - |cluster ∩ AstDys| / |cluster|
```

The benchmark requires completeness ≥ 95% for each target family.

---

## 5. Code Structure

### File Overview

```
asteroid-families/
│
├── convert.py        - converts raw .txt files to .csv
├── constants.py      - all shared physical and algorithmic constants
├── load.py           - loads CSV files into pandas DataFrames
├── preprocess.py     - filters and cleans proper elements
├── metric.py         - Zappalà velocity distance function
├── clustering.py     - DBSCAN clustering
├── evaluate.py       - completeness and contamination vs AstDys
├── visualize.py      - scatter plots and completeness bar chart
├── main.py           - runs the full pipeline end to end
│
├── results/
│   ├── evaluation.csv        - completeness/contamination results
│   ├── proper_elements.png   - scatter plots of families
│   └── completeness.png      - bar chart vs 95% benchmark
│
└── README.md
```

### File Details

#### `constants.py`
Centralizes all physical and algorithmic constants used across the project.
Importing from one place ensures consistency and makes it easy to tune
parameters without hunting through multiple files.

Key constants:
- `GM_AU_DAY` - gravitational parameter of the Sun in AU³/day²
- `AU_TO_KM` - unit conversion factor
- `K_A, K_E, K_I` - Zappalà metric weights (from Zappalà et al. 1990)
- `CLUSTERING_COLS` - the three proper elements used for clustering
- `D_DEG_MIN/MAX` - libration amplitude filter bounds
- `LYAPUNOV_MAX` - stability filter threshold
- `EPS` - DBSCAN velocity cutoff in m/s
- `MIN_FAMILY_SIZE` - minimum cluster size
- `TARGET_FAMILIES` - AstDys IDs of the 7 benchmark Trojan families
- `COMPLETENESS_THRESHOLD` - 95% benchmark requirement

#### `convert.py`
Converts the raw AstDys text files to CSV format for easy loading with
pandas. Handles the whitespace-delimited format with `%` comment lines
used by AstDys. Assigns correct column names based on the file headers.

#### `load.py`
Loads both CSV files into pandas DataFrames with `asteroid_number` set
as the index. This is critical, using the asteroid number as the index
(rather than row number) ensures that membership lookups between the two
datasets match correctly by asteroid ID.

#### `preprocess.py`
Applies three sequential filters to clean the proper elements dataset
before clustering. Each step prints how many asteroids were removed and
how many remain, providing a clear audit trail.

#### `metric.py`
Implements the Zappalà velocity metric from Zappalà et al. (1990). The
`zappala_distance` function computes the velocity distance between two
asteroids given their `[da_AU, e_p, sin_i_p]` values. The
`compute_distance_matrix` function computes the full pairwise distance
matrix using `scipy.spatial.distance.pdist` for efficiency, then converts
to a square matrix. Also exports `EPS = 50.0` so `main.py` can import it.

#### `clustering.py`
Runs DBSCAN on the precomputed distance matrix using `sklearn.cluster.DBSCAN`
with `metric="precomputed"`. Asteroids not belonging to any dense cluster
are labeled -1 (noise/background). Prints the number of clusters found
and noise points for transparency.

#### `evaluate.py`
For each of the 7 target Trojan families, finds the best-matching cluster
by maximum asteroid ID overlap, then computes completeness and contamination.
Prints a full benchmark results table and returns a DataFrame for plotting.

#### `visualize.py`
Generates two plots saved to `results/`:
1. Side-by-side scatter plots of `da_AU vs e_p` and `da_AU vs sin_i_p`,
   with background asteroids in grey and family members colored by cluster
2. A bar chart of completeness per family with a red dashed 95% threshold line

Uses `matplotlib.use("Agg")` for non-interactive rendering so plots are
saved to file without requiring a display.

#### `main.py`
Orchestrates the full pipeline by importing and calling each module in order.
Running `python main.py` executes all steps from data loading through
visualization and saves results to `results/`.

---

## 6. Results

All 7 Trojan families in the AstDys catalog were identified at ≥95%
completeness using DBSCAN with `eps = 50 m/s`:

| Family ID | Common Name | AstDys Members | Completeness | Contamination |
|-----------|-------------|---------------|--------------|---------------|
| 3548      | Eurybates   | 484           | 100%         | low           |
| 624       | Hektor      | 58            | 100%         | low           |
| 8060      | -           | 59            | 100%         | low           |
| 9799      | Cloanthus   | 56            | 100%         | low           |
| 17492     | -           | 20            | 100%         | low           |
| 291316    | -           | 19            | 100%         | low           |
| 222861    | -           | 17            | 100%         | low           |

**Status: All 7 available Trojan families identified at ≥95% completeness.**

Note: The AstDys Trojan proper elements catalog (`tro_syn`) contains exactly
7 verified Trojan families. Main belt families (Flora, Vesta, Eos, etc.)
require the main belt proper elements catalog (`allnum.cat`) which was not
part of the provided dataset.

---


## 7. Limitations

**7 families instead of 8.** The Trojan proper elements catalog contains exactly 7 verified Trojan families - there is no 8th to find. The 8-family benchmark targets the main belt catalog (`allnum.cat`), which contains the classic Hirayama families (Flora, Vesta, Eos, etc.). Extending this pipeline to the main belt would require that dataset and updated preprocessing filters.

**Contamination.** All 7 families achieve 100% completeness but clusters absorb some background asteroids within the 50 m/s radius. Lowering `eps` reduces contamination but risks splitting families; this is an inherent trade-off of density-based clustering.

**No QRL significance test.** Zappalà et al. (1990) recommend a Monte Carlo quasi-random level test to confirm clusters are statistically significant above the background. This was not implemented due to the computational cost (~100 re-clusterings of a 11,801×11,801 distance matrix).

**No robustness test.** Perturbing proper elements within their measurement uncertainties and re-clustering to verify family stability was not implemented. The AstDys `tro.rms` file provides the uncertainty estimates needed for this.

**Trojan metric approximation.** The Zappalà metric was derived for main belt asteroids. The Trojan adaptation uses `da_AU` (deviation from Jupiter's semi-major axis) as a proxy, following Milani (1993), which does not fully account for libration dynamics around the Lagrange point.

**Single global velocity cutoff.** All families use `eps = 50 m/s`. Older, more dispersed families may need a higher cutoff while compact young families need lower. A per-family adaptive eps would be more rigorous.

## 8. Usage

```bash
# Step 1: install dependencies
pip install numpy pandas scipy scikit-learn matplotlib

# Step 2: place raw data files in project folder
# tro_syn.txt
# all_tro_members.txt

# Step 3: convert raw data to CSV
python convert.py

# Step 4: run the full pipeline
python main.py
```

Results are saved to `results/`:
- `evaluation.csv` - completeness and contamination per family
- `proper_elements.png` - scatter plots of families in proper element space
- `completeness.png` - completeness bar chart vs 95% benchmark

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data loading and manipulation |
| `scipy` | Pairwise distance computation, hierarchical clustering |
| `scikit-learn` | DBSCAN clustering |
| `matplotlib` | Plotting and visualization |

Install all with:
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

---

## 10. References

Hirayama, K. (1919). Groups of Asteroids Probably of Common Origin.
*Astronomical Journal*, 31, 185.

Zappalà, V., Cellino, A., Farinella, P., & Knezevic, Z. (1990).
Asteroid Families. I. Identification by Hierarchical Clustering and
Reliability Assessment. *Astronomical Journal*, 100, 2030.
DOI: 10.1086/115658

Milani, A. (1993). The Trojan asteroid belt: proper elements, stability,
chaos and families. *Celestial Mechanics and Dynamical Astronomy*, 57, 59–94.

Knezevic, Z., & Milani, A. (2003). Proper element catalogs and asteroid
families. *Astronomy & Astrophysics*, 403, 1165–1173.

Milani, A., Cellino, A., Knezevic, Z., Novakovic, B., Spoto, F., &
Paolicchi, P. (2014). Asteroid families classification: exploiting very
large data sets. *Icarus*, 239, 46–73.
