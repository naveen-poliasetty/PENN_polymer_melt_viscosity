from torch.utils.data import DataLoader
import numpy as np

from src.data.splitting import extrapolation_split
from src.data.preprocessing import PolymerViscosityDataset

def create_dataloaders(
    df,
    preprocessor,
    physical_variable="Mw",
    batch_size=64,
    random_state=42,
    num_workers=0,
):

    # --------------------------------------------------------
    # Generate fingerprints
    # --------------------------------------------------------

    fingerprints = preprocessor.generate_fingerprints(df)

    # --------------------------------------------------------
    # Extrapolation split
    # --------------------------------------------------------

    train_idx, test_idx = extrapolation_split(
        df,
        physical_variable=physical_variable,
        random_state=random_state,
    )

    train_df = df.loc[
        train_idx
    ].reset_index(drop=True)

    test_df = df.loc[
        test_idx
    ].reset_index(drop=True)

    train_fp = fingerprints[
        np.isin(
            df.index.to_numpy(),
            train_idx,
        )
    ]

    test_fp = fingerprints[
        np.isin(
            df.index.to_numpy(),
            test_idx,
        )
    ]

    # --------------------------------------------------------
    # FIT SCALERS ONLY ON TRAIN
    # --------------------------------------------------------

    preprocessor.fit(
        train_df,
        train_fp,
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    train_data = preprocessor.transform(
        train_df,
        train_fp,
    )

    test_data = preprocessor.transform(
        test_df,
        test_fp,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = PolymerViscosityDataset(
        train_data["fingerprint"],
        train_data["Mw"],
        train_data["Shear_Rate"],
        train_data["Temperature"],
        train_data["PDI"],
        train_data["viscosity"],
    )

    test_dataset = PolymerViscosityDataset(
        test_data["fingerprint"],
        test_data["Mw"],
        test_data["Shear_Rate"],
        test_data["Temperature"],
        test_data["PDI"],
        test_data["viscosity"],
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        test_loader,
        train_df,
        test_df,
        preprocessor,
    )