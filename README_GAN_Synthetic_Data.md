
# WGAN-GP for Tabular Synthetic Data (PyTorch)

The project implements a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for **tabular** data, augmented with **feature matching** and **correlation matching** regularizers for higher-fidelity samples.

---

## Repository layout
.
├── GAN-Synthetic-Data-generation.pdf
├── GAN-Synthetic-Data-generation.ipynb     # main notebook (WGAN-GP training + export)
└── synthetic\_wgan\_gp.csv            # example output (synthetic samples)

````

---

## Quick start (environment)

```bash
pip install --upgrade pip
pip install torch numpy pandas scikit-learn seaborn matplotlib scipy tqdm
````

Open the notebook **`GAN-Synthetic-Data-generation.ipynb`** and run top-to-bottom. Adjust hyperparameters in the **Training Loop Setup** cell as needed.

---

## Data

**Expected input**: `data.xlsx` at the repository root. The notebook performs preprocessing:

* Load to DataFrame: `pd.read_excel("data.xlsx")`
* Remove outliers via **z-score** on selected columns:

  * `sal_pur_rat` with threshold **3**
  * `lib_igst_itc_rat` with threshold **0.1**
* Select **numeric** columns only and **standardize** using `StandardScaler()`
* Wrap in `TensorDataset` + `DataLoader(batch_size=64, shuffle=True)`

**Feature dimensionality**: the model learns over the numeric feature count in your data (example run: **256 features**). Generator outputs samples in this standardized feature space; use the saved scaler to invert if you want original units.

---

## Model

### Generator (MLP)

* Input noise **z ∈ ℝ^32**
* Layers:
  `Linear(32→128) → ReLU → BatchNorm1d(128) → Linear(128→256) → ReLU → BatchNorm1d(256) → Linear(256→256)`
* Output: **256-D** synthetic vector (standardized space)

### Critic (Wasserstein)

* Input: **256-D** feature vector
* Layers:
  `Linear(256→256) → LeakyReLU(0.2) → Linear(256→128) → LeakyReLU(0.2) → Linear(128→1)`
* Can return penultimate features to support **feature matching** loss.

---

## Objectives

* **WGAN-GP critic loss**

  $$
  \mathcal{L}_D \;=\; -\Big(\mathbb{E}[D(x_{\text{real}})] - \mathbb{E}[D(G(z))]\Big) \;+\; \lambda_{\text{GP}}\cdot \text{GP}
  $$

  with **gradient penalty** computed on random interpolations between real and generated samples.

* **Generator loss**

  $$
  \mathcal{L}_G \;=\; -\mathbb{E}[D(G(z))] \;+\; \lambda_{\text{FM}} \,\big\|\mu_f^{\text{fake}} - \mu_f^{\text{real}}\big\|_2^2
  \;+\; \lambda_{\text{corr}} \,\big\|\mathrm{Corr}(G(z)) - \mathrm{Corr}(x_{\text{real}})\big\|_2^2
  $$

**Coefficients (defaults from the notebook)**

* Gradient penalty: **λ\_GP = 10**
* Feature matching: **λ\_FM = 10.0** (MSE between critic feature means)
* Correlation matching: **λ\_corr = 5.0** (MSE between mini-batch Pearson correlation matrices)

---

## Training configuration (defaults)

| Setting                     | Value                    |
| --------------------------- | ------------------------ |
| Noise dimension             | 32                       |
| Features (input/output dim) | 256                      |
| Batch size                  | 64                       |
| Epochs                      | 1500                     |
| Optimizer                   | Adam                     |
| Learning rate               | 1e-4                     |
| Betas                       | (0.5, 0.9)               |
| Critic steps per G          | 15                       |
| Device                      | Auto (CUDA if available) |

Edit these in the hyperparameters cell.

---

## How to run

1. **Place data**: add your Excel file as **`data.xlsx`** (or update the path in the first cell).
2. **Train & generate**: run all cells. The notebook logs losses/diagnostics and writes:

   * **`synthetic_wgan_gp.csv`** — synthetic dataset with the **same schema** (column order/names) as your input DataFrame.
3. **(Optional) Invert scaling** to get back to original units using the fitted `StandardScaler`.

---

## Evaluation (built-in)

* **Pearson correlation**: real vs. synthetic heatmaps; **MAE** between correlation matrices.
* **Kolmogorov–Smirnov**: per-feature two-sample tests.
* **Distribution plots**: histogram/density overlays for key columns.

---



