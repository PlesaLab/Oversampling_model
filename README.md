# Synthetic Library Oversampling Model

This repository provides a Python implementation of a quantitative model for estimating the sampling requirements of synthetic gene libraries with non‑uniform representation and imperfect fidelity. The model is conceptually inspired by the probabilistic framework used in Guido NJ, Handerson S, Joseph EM, Leake D, Kung LA (2016) [Determination of a Screening Metric for High Diversity DNA Libraries](https://doi.org/10.1371/journal.pone.0167088). PLoS ONE 11(12): e0167088.

---

## Overview

Given a library with:

- **Diversity (N)** – number of unique intended variants,
- **Fidelity (f)** – fraction of *perfect* molecules (e.g. 0.8 for 80%),
- **Uniformity (Gini)** – representation unevenness (0 = perfectly uniform, 1 = extremely skewed),

the model estimates either:

1. **Required sample size** `t` to reach a target coverage with a given probability, or  
2. **Guaranteed coverage** achievable for a given `t` and probability.

The script implements a coupon‑collector–style model with Poissonization and lognormal‑derived frequency distributions whose Gini coefficient matches the input value.

---

## Installation

Requires Python ≥3.8 with no non‑standard dependencies (only `numpy`).

```bash
git clone https://github.com/PlesaLab/Oversampling_model.git
cd Oversampling_model
pip install numpy
```

---

## Usage

### Compute number of samples needed for desired coverage

```bash
python oversampling.py --N 1536 --fidelity 0.30 --gini 0.40 t_for_coverage --coverage 0.95 --prob 0.95
```

**Output example:**

```
[t_for_coverage]
N=1,536  fidelity=0.3000  gini=0.4000  K=400
Target: coverage >= 0.9500 with probability >= 0.950
Required samples t  = 36,068.091
Effective perfect draws f*t = 10,820.427
At t, E[coverage]=0.957693  Var=2.187476e-05
```

### Compute coverage guaranteed for a given sample size

```bash
python oversampling.py --N 1536 --fidelity 0.30 --gini 0.40 coverage_for_t --t 1E5 --prob 0.95
```

**Output example:**

```
[coverage_for_t]
N=1,536  fidelity=0.3000  gini=0.4000  K=400
At t=100,000.000, guaranteed coverage (p>=0.950) = 0.994161
E[coverage]=0.996523  Var=2.062565e-06
```

---

## Model Details

- **Fidelity (`f`)** acts as a multiplicative penalty: only a fraction `f` of draws are usable (so effective sampling = `f × t`).
- **Non‑uniformity (Gini)** is achieved via a lognormal representation tuned by bisection to match the specified Gini value.
- **Coverage distribution** is approximated via a central‑limit‑theorem (CLT) form: 

$P(C_t \ge c) \approx 1 - \Phi\!\left(\frac{c - \mathbb{E}[C_t]}{\sqrt{\mathrm{Var}[C_t]}}\right)$

where $𝐶_𝑡$ is the coverage fraction observed after sampling `t` molecules.


- Solving for `t` gives the minimal sampling depth meeting your desired coverage and probability.

---

## 📊 Example Parameters

| Parameter | Meaning | Typical Range |
|------------|----------|----------------|
| `N` | Number of unique genes | 384–786,432 |
| `fidelity` | Fraction perfect | 0.05–0.6 |
| `gini` | Library unevenness | 0.15–0.9 |
| `coverage` | Desired completeness | 0.8–0.99 |
| `prob` | Probability guarantee | 0.9–0.99 |

---

## Notes

- The algorithm uses 400 aggregate bins by default (`--K`) for speed; increase this for very high `N` or extreme `Gini` values.


---

## License

MIT License © 2025 Plesa Lab @ University of Oregon

---


