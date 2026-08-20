from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Configuration
# ============================================================

@dataclass
class PreprocessingConfig:
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048

    pdi_median: float = 2.06

    scale_min: float = -1.0
    scale_max: float = 1.0

    random_state: int = 42


# ============================================================
# SMILES utilities
# ============================================================

def parse_smiles_list(value):
    """
    Convert the SMILES column into a list of SMILES.

    Examples
    --------
    '[SMILES_A, SMILES_B]' -> [SMILES_A, SMILES_B]

    'SMILES_A' -> [SMILES_A]
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Try Python-list representation first
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple)):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except (ValueError, SyntaxError):
            pass

    # Fallback: comma-separated representation
    if "," in value:
        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return [value]


def smiles_to_mol(smiles: str):
    """
    Convert SMILES to RDKit molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol


def smiles_to_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    Generate Morgan fingerprint as a float vector.
    """

    mol = smiles_to_mol(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
    )

    arr = np.zeros(n_bits, dtype=np.float32)

    # Convert RDKit ExplicitBitVect -> numpy
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)

    return arr


# ============================================================
# Composition utilities
# ============================================================

def get_weights(row, n_components):
    """
    Extract composition weights.

    Current dataset appears to expose Weight 1 / Weight 2.
    """

    if n_components == 1:
        return np.array([1.0], dtype=np.float32)

    weights = []

    for i in range(1, n_components + 1):

        column = f"Weight {i}"

        if column in row.index:
            value = row[column]

            if pd.notna(value):
                weights.append(float(value))

    if len(weights) != n_components:
        raise ValueError(
            f"Expected {n_components} weights but found {weights}"
        )

    weights = np.asarray(weights, dtype=np.float32)

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            f"Invalid composition weights: {weights}"
        )

    # Normalize in case they don't exactly sum to 1
    weights = weights / total

    return weights


# ============================================================
# Fingerprint aggregation
# ============================================================

def weighted_arithmetic_average(
    fingerprints,
    weights,
):
    """
    Composition-weighted arithmetic average.

    Used for copolymers.
    """

    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    return np.sum(
        fingerprints * weights[:, None],
        axis=0,
    )


def weighted_harmonic_average(
    fingerprints,
    weights,
    eps=1e-8,
):
    """
    Composition-weighted harmonic average.

    H = 1 / sum(w_i / x_i)

    IMPORTANT:
    Binary fingerprints contain zeros. Therefore a direct harmonic
    mean would be undefined wherever a component fingerprint is zero.

    We therefore use eps for numerical stability.

    This should be verified against the original implementation/paper
    if their "harmonic average" refers to a different fingerprint
    aggregation operation.
    """

    fingerprints = np.asarray(
        fingerprints,
        dtype=np.float32,
    )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    safe_fp = np.maximum(fingerprints, eps)

    denominator = np.sum(
        weights[:, None] / safe_fp,
        axis=0,
    )

    return 1.0 / denominator


def aggregate_fingerprint(
    smiles_list,
    sample_type,
    weights,
    config,
):
    """
    Generate and aggregate polymer fingerprints.
    """

    fps = [
        smiles_to_fingerprint(
            smiles,
            radius=config.fingerprint_radius,
            n_bits=config.fingerprint_bits,
        )
        for smiles in smiles_list
    ]

    if len(fps) == 0:
        raise ValueError("No valid SMILES found.")

    if len(fps) == 1:
        return fps[0]

    sample_type = str(sample_type).lower()

    if "copolymer" in sample_type:
        return weighted_arithmetic_average(
            fps,
            weights,
        )

    if "blend" in sample_type:
        return weighted_harmonic_average(
            fps,
            weights,
        )

    # Safe fallback
    return weighted_arithmetic_average(
        fps,
        weights,
    )


# ============================================================
# Main preprocessor
# ============================================================

class PolymerPreprocessor:

    def __init__(self, config=None):
        self.config = config or PreprocessingConfig()

        self.fp_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.mw_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.shear_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.temperature_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.pdi_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.viscosity_scaler = MinMaxScaler(
            feature_range=(
                self.config.scale_min,
                self.config.scale_max,
            )
        )

        self.fitted = False

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    def clean_dataframe(self, df):

        df = df.copy()

        numeric_columns = [
            "Mn",
            "Mw",
            "PDI",
            "Temperature",
            "Shear_Rate",
            "Melt_Viscosity",
            "Weight 1",
            "Weight 2",
            "Aug",
        ]

        for col in numeric_columns:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

        # Remove rows without essential values
        essential = [
            "SMILES",
            "Mw",
            "Temperature",
            "Shear_Rate",
            "Melt_Viscosity",
            "Sample_Type",
        ]

        df = df.dropna(
            subset=essential
        ).reset_index(drop=True)

        # PDI median imputation
        df["PDI"] = df["PDI"].fillna(
            self.config.pdi_median
        )

        return df

    # --------------------------------------------------------
    # Fingerprints
    # --------------------------------------------------------

    def generate_fingerprints(self, df):

        fingerprints = []

        for _, row in df.iterrows():

            smiles_list = parse_smiles_list(
                row["SMILES"]
            )

            weights = get_weights(
                row,
                len(smiles_list),
            )

            fp = aggregate_fingerprint(
                smiles_list=smiles_list,
                sample_type=row["Sample_Type"],
                weights=weights,
                config=self.config,
            )

            fingerprints.append(fp)

        return np.vstack(fingerprints)

    # --------------------------------------------------------
    # Transform physical variables
    # --------------------------------------------------------

    @staticmethod
    def log_mw(df):
        return np.log10(
            df["Mw"].astype(float)
        ).to_numpy().reshape(-1, 1)

    @staticmethod
    def log_shear(df):
        return np.log10(
            df["Shear_Rate"].astype(float) + 1e-5
        ).to_numpy().reshape(-1, 1)

    @staticmethod
    def log_viscosity(df):
        return np.log10(
            df["Melt_Viscosity"].astype(float)
        ).to_numpy().reshape(-1, 1)

    # --------------------------------------------------------
    # Fit scalers
    # --------------------------------------------------------

    def fit(self, df, fingerprints):

        self.fp_scaler.fit(fingerprints)

        self.mw_scaler.fit(
            self.log_mw(df)
        )

        self.shear_scaler.fit(
            self.log_shear(df)
        )

        self.temperature_scaler.fit(
            df["Temperature"]
            .to_numpy()
            .reshape(-1, 1)
        )

        self.pdi_scaler.fit(
            df["PDI"]
            .to_numpy()
            .reshape(-1, 1)
        )

        self.viscosity_scaler.fit(
            self.log_viscosity(df)
        )

        self.fitted = True

        return self

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    def transform(self, df, fingerprints):

        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()."
            )

        result = {}

        result["fingerprint"] = (
            self.fp_scaler.transform(
                fingerprints
            ).astype(np.float32)
        )

        result["Mw"] = (
            self.mw_scaler.transform(
                self.log_mw(df)
            ).astype(np.float32)
        )

        result["Shear_Rate"] = (
            self.shear_scaler.transform(
                self.log_shear(df)
            ).astype(np.float32)
        )

        result["Temperature"] = (
            self.temperature_scaler.transform(
                df["Temperature"]
                .to_numpy()
                .reshape(-1, 1)
            ).astype(np.float32)
        )

        result["PDI"] = (
            self.pdi_scaler.transform(
                df["PDI"]
                .to_numpy()
                .reshape(-1, 1)
            ).astype(np.float32)
        )

        result["viscosity"] = (
            self.viscosity_scaler.transform(
                self.log_viscosity(df)
            ).astype(np.float32)
        )

        return result

    # --------------------------------------------------------
    # Inverse viscosity
    # --------------------------------------------------------

    def inverse_viscosity(self, values):

        values = np.asarray(values).reshape(-1, 1)

        log_eta = self.viscosity_scaler.inverse_transform(
            values
        )

        return np.power(
            10,
            log_eta,
        )