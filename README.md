# Physics-Enforced Neural Network (PENN) for Polymer Melt Viscosity

## Comprehensive Technical Documentation

---

### Table of Contents

1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Dataset Overview & Molecular Featurization](#2-dataset-overview--molecular-featurization)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Generalization & Extrapolation Splitting Strategy](#4-generalization--extrapolation-splitting-strategy)
5. [PyTorch Data Pipeline & Tensor Structure](#5-pytorch-data-pipeline--tensor-structure)
6. [PENN Model Architecture & Physics Enforcement](#6-penn-model-architecture--physics-enforcement)
7. [Training Dynamics, Loss Formulation & Early Stopping](#7-training-dynamics-loss-formulation--early-stopping)
8. [Weights & Biases (W&B) MLOps & Experiment Tracking](#8-weights--biases-mlops--experiment-tracking)
9. [Empirical Results on Extrapolation Benchmarks](#9-empirical-results-on-extrapolation-benchmarks)

---

### 1. Executive Summary & System Architecture

The **Physics-Enforced Neural Network (PENN)** is a hybrid machine learning and domain-knowledge framework designed to predict the **non-Newtonian melt viscosity** ($\eta$) of polymers across diverse chemical structures and processing conditions (temperature $T$, shear rate $\dot{\gamma}$, molecular weight $M_w$, and polydispersity index $\text{PDI}$).

Unlike pure black-box neural networks (which violate fundamental physical constraints and fail on out-of-distribution extrapolation), PENN splits the prediction pipeline into two synergistic layers:

1. **Chemical/Structural Mapping**: An MLP maps the polymer's molecular fingerprint (Morgan circular fingerprints) and $\text{PDI}$ to **latent, polymer-specific rheological constants** ($\alpha_1, \alpha_2, M_{\text{cr}}, C_1, C_2, T_r, n, \dot{\gamma}_{\text{cr}}$, etc.).
2. **Differentiable Physics Layers**: The estimated physical parameters parameterize analytical rheology equations (Williams-Landel-Ferry/Arrhenius temperature dependence, critical entanglement molecular weight scaling, and power-law shear thinning).

```
                      [ Polymer SMILES ] + [ PDI ]
                                  │
                                  ▼
               [ RDKit Morgan Fingerprint (1024-d) ]
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   MLP_PENN (Latent NN)   │
                     └────────────┬─────────────┘
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                           ▼                           ▼
[ α1, α2, Mcr, k1, βM ]     [ C1, C2, Tr ]          [ n, γ̇_cr, β_shear ]
(Entanglement Scaling)    (Temperature Shift)        (Shear Thinning)
      │                           │                           │
      │                  [ Temperature (T) ]                  │
      │                           │                           │
      ▼                           ▼                           │
 ┌──────────────────────────────────────┐                     │
 │      Temperature Shift (a_T)         │                     │
 └──────────────────┬───────────────────┘                     │
                    │                                         │
       [ Molecular Weight (Mw) ]                              │
                    │                                         │
                    ▼                                         │
 ┌──────────────────────────────────────┐                     │
 │   Zero-Shear Viscosity Layer (η_0)   │                     │
 └──────────────────┬───────────────────┘                     │
                    │                                         │
                    └─────────────────┐   [ Shear Rate (γ̇) ]  │
                                      │           │           │
                                      ▼           ▼           ▼
                                ┌───────────────────────────────┐
                                │   Shear Thinning Layer (η)    │
                                └───────────────┬───────────────┘
                                                │
                                                ▼
                                    [ Predicted Melt Viscosity ]
```

---

### 2. Dataset Overview & Molecular Featurization

- **Source File**: `data/raw/melt_viscosity_dataset.csv`
- **Cleaned Dataset Size**: 1,921 valid polymer records across homopolymers, copolymers, and polymer blends.
- **Core Variables**:
  - **Chemical Identity**: Monomer/Polymer SMILES string representation, Polymer Type (`homopolymer`, `copolymer`, `blend`), Component weights (`Weight 1`, `Weight 2`).
  - **Physical Input Conditions**:
    - Molecular Weight ($M_w$) [g/mol]
    - Temperature ($T$) [°C / K]
    - Shear Rate ($\dot{\gamma}$) [$\text{s}^{-1}$]
    - Polydispersity Index ($\text{PDI} = M_w / M_n$)
  - **Target**: Melt Viscosity ($\eta$) [$\text{Pa}\cdot\text{s}$ or $\text{Poise}$].

#### Molecular Fingerprint Extraction

Using RDKit's `rdFingerprintGenerator.GetMorganGenerator`, 1024-bit Morgan circular fingerprints (radius = 2, equivalent to ECFP4) are generated for each chemical component:

```python
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
mol = Chem.MolFromSmiles(smiles)
fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
```

---

### 3. Data Preprocessing Pipeline

Implemented in `src/data/preprocessing.py`:

#### A. Multi-Component Fingerprint Aggregation

For copolymers and polymer blends, component fingerprints and physical properties are aggregated based on chemistry rules:

- **Homopolymer**: Single component fingerprint:
  $$\text{fp}_{\text{agg}} = \text{fp}_1$$

- **Copolymer**: Linear composition-weighted average:
  $$\text{fp}_{\text{agg}} = \sum_i w_i \cdot \text{fp}_i$$
  $$M_{w,\text{agg}} = \sum_i w_i \cdot M_{w,i}, \qquad \sum_i w_i = 1$$

- **Polymer Blend**: Element-wise composition-weighted harmonic mean:

  $$
  \frac{1}{\text{fp}_{\text{blend},j}}
  =
  \sum_i \frac{w_i}{\text{fp}_{i,j} + \epsilon}
  $$

  Therefore:

  $$
  \text{fp}_{\text{blend},j}
  =
  \left(
  \sum_i \frac{w_i}{\text{fp}_{i,j} + \epsilon}
  \right)^{-1}
  $$

  where \(j\) denotes the fingerprint dimension and \(\epsilon > 0\) is a small numerical-stability constant.

#### B. PDI Missing Value Imputation

Missing $\text{PDI}$ values are imputed using the global dataset median ($\text{PDI} = 2.06$):

```python
df['PDI'] = df['PDI'].fillna(2.06)
```

#### C. PENNScaler (Unified Physical Dimension Scaling)

Rheological parameters are bounded within $[-1, 1]$ using scikit-learn's `MinMaxScaler`:

- **Fingerprints**: Normalized bit vectors to $[-1, 1]$.
- **PDI & Temperature**: Standard min-max scaling to $[-1, 1]$.
- **Shared Physical Scaler**:  
  To enforce dimensionally consistent transitions in the physics equations, `log10(M_w)`, `log10(γ̇ + 10⁻⁵)`, and `log10(η)` are fit together under a **single global shared scaler**:

  $$
  \begin{bmatrix}
  \log_{10}(M_w) \\
  \log_{10}(\dot{\gamma} + 10^{-5}) \\
  \log_{10}(\eta)
  \end{bmatrix}
  \rightarrow
  \text{MinMaxScaler}
  \rightarrow
  [-1,1]^3
  $$

  The same fitted scaler is then used to transform and inverse-transform all three physical quantities consistently.

---

### 4. Generalization & Extrapolation Splitting Strategy

Implemented in `src/data/splitting.py`:

To evaluate whether the physics-informed model extrapolates beyond the training regime rather than simply interpolating:

1. **Chemistry-Grouped Partitioning**:
   A `GroupShuffleSplit` on `monomer_id` isolates 90% of polymer chemistries for base training and 10% of chemistries for evaluation.
2. **Physical Variable Extrapolation Probe**:
   For the 10% held-out polymer families, samples are divided at the median of a specific physical variable (e.g., $M_w$, $\dot{\gamma}$, or $T$):
   - One half is placed in the training set (e.g., lower $M_w$ range).
   - The remaining half is placed in the test set (e.g., upper $M_w$ range).
     This forces the model to predict into unseen physical regimes for unseen chemistries.

---

### 5. PyTorch Data Pipeline & Tensor Structure

Implemented in `src/data/dataloader.py`:

`PENNDataset` converts DataFrames into PyTorch batches yielding the 6-tuple `(XX, M, S, T, P, visc)`:

| Tensor | Shape                | Type            | Description                                |
| :----- | :------------------- | :-------------- | :----------------------------------------- |
| `XX`   | `[batch_size, 1024]` | `torch.float32` | Polymer Morgan fingerprint vector          |
| `M`    | `[batch_size, 1]`    | `torch.float32` | Scaled $\log_{10}(M_w)$                    |
| `S`    | `[batch_size, 1]`    | `torch.float32` | Scaled $\log_{10}(\dot{\gamma} + 10^{-5})$ |
| `T`    | `[batch_size, 1]`    | `torch.float32` | Scaled Temperature                         |
| `P`    | `[batch_size, 1]`    | `torch.float32` | Scaled $\text{PDI}$                        |
| `visc` | `[batch_size, 1]`    | `torch.float32` | Target scaled $\log_{10}(\eta)$            |

---

### 6. PENN Model Architecture & Physics Enforcement

Implemented in `src/models/penn.py`:

#### A. Latent MLP Parameter Estimator (`MLP_PENN`)

- **Input Layer**: `1024 (Fingerprint) + 1 (PDI) = 1025 dimensions`.
- **Architecture**: Linear(1025, 120) $\rightarrow$ Dropout(0.2) $\rightarrow$ ReLU $\rightarrow$ Linear(120, 120) $\rightarrow$ Dropout(0.2) $\rightarrow$ ReLU $\rightarrow$ Linear(120, 11).
- **Physical Parameter Transformations**:
  - $\alpha_1 = 3.0 \cdot \sigma(\theta_0)$ _(Unentangled power-law slope, expected $\approx 1.0$)_
  - $\alpha_2 = 6.0 \cdot \sigma(\theta_1)$ _(Entangled reptation slope, expected $\approx 3.4$)_
  - $k_1 = 2.0 \cdot \tanh(\theta_2) - 1.0$ _(Zero-shear baseline offset)_
  - $\beta_M = 30 + 30 \cdot \sigma(\theta_3)$ _(Molecular weight regime transition sharpness)_
  - $M_{\text{cr}} = 0.5 \cdot \sigma(\theta_4) - 0.5$ _(Critical entanglement molecular weight)_
  - $C_1 = 2.0 \cdot \sigma(\theta_5), \quad C_2 = 2.0 \cdot \sigma(\theta_6), \quad T_r = \tanh(\theta_7) - 1.0$ _(WLF / Arrhenius shift constants)_
  - $n = \sigma(\theta_8)$ _(Shear-thinning power-law index)_
  - $\dot{\gamma}_{\text{cr}} = \sigma(5.0 \cdot \theta_9) - 1.0$ _(Critical shear rate threshold)_
  - $\beta_{\text{shear}} = 10 + 30 \cdot \sigma(\theta_{10})$ _(Shear thinning transition sharpness)_

#### B. Differentiable Rheological Layers

##### 1. Temperature Shift ($a_T$)

Using the WLF equation formulation:
$$a_T = \frac{-C_1 (T - T_r)}{C_2 + (T - T_r)}$$

##### 2. Molecular Weight Dependent Zero-Shear Viscosity (`MolWeight`)

Implements the transition between the unentangled Rouse regime (η₀ ∝ Mᵥ¹) and entangled Reptation regime (η₀ ∝ Mᵥ³·⁴) via a smooth sigmoid-approximated Heaviside step function:

- **low_mw** = (k₁ + aₜ) + α₁M

- **k₂** = (k₁ + aₜ) + (α₁ − α₂)M_cr

- **high_mw** = k₂ + α₂M

- **η₀** = low_mw × σ(βₘ(M_cr − M)) + high_mw × σ(βₘ(M − M_cr))

##### 3. Shear-Thinning Viscosity (`ShearRate`)

Models the transition from the Newtonian zero-shear plateau to pseudoplastic shear thinning:

- **low_shear** = η₀

- **high_shear** = η₀ − n(S − γ̇_cr)

- **η** = low_shear × σ(βₛ(γ̇_cr − S)) + high_shear × σ(βₛ(S − γ̇_cr))

---

### 7. Training Dynamics, Loss Formulation & Early Stopping

#### Loss Function with Physics Enforcing Penalties

The total objective function combines empirical prediction error with physics constraint losses on the asymptotic slopes α₁ and α₂:

**L_total** = L_MSE(η̂, η) + λ_phys × ‖α₁ − 1.0‖² + λ_phys × ‖α₂ − 3.4‖²

- **λ_phys** = `config["a_weight"]` = 0.1

#### Training Configuration:

- **Optimizer**: Adam ($\text{lr} = 10^{-3}$)
- **Criterion**: MSE Loss
- **Batch Size**: 32
- **Max Epochs**: 200
- **Early Stopping**: Monitored on Validation Loss with `patience = 30` and `delta = 1e-4`.

---

### 8. Weights & Biases (W&B) MLOps & Experiment Tracking

The training pipeline in `main.py` integrates full-stack W&B logging:

- **Gradient & Topology Tracking**: `wandb.watch(model, log="all", log_freq=10)`.
- **Real-Time Physics Parameter Monitoring**: Logs the distribution (`mean`, `std`, `min`, `max`) of all 11 latent constants at every epoch.
- **Validation Prediction Table**: Generates a `wandb.Table` containing per-sample metadata, true scaled vs. predicted viscosity, residuals, and absolute errors.
- **Scatter Plot**: Automatically logs an interactive actual vs. predicted plot (`wandb.plot.scatter`).
- **Model Checkpoint Artifacts**: Automatically saves the best model state dictionary as a versioned `wandb.Artifact` (`penn_best_model`).

---

### 9. Empirical Results on Extrapolation Benchmarks

On the $M_w$ extrapolation split (training on 1,821 records, testing on 100 out-of-distribution samples):

```
========================================================================
PENN Extrapolation Split (Variable: Mw) - Best Validation Performance
========================================================================
• Best Epoch:            27
• Best Validation Loss:  0.0145
• R² Score:              0.7232
• RMSE:                  0.1205
• MAE:                   0.1014
• Early Stopping:        Triggered at epoch 57 (patience = 30)
• Model Artifact:        Saved to best_penn_model.pth & uploaded to W&B
========================================================================
```
