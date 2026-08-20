import numpy as np
import pandas as pd


def extrapolation_split(
    df,
    physical_variable,
    test_monomer_fraction=0.10,
    random_state=42,
):
    """
    Create the monomer-based extrapolation split.

    Parameters
    ----------
    df : DataFrame

    physical_variable :
        One of:
            "Mw"
            "Shear_Rate"
            "Temperature"

    Returns
    -------
    train_idx
    test_idx
    """

    if physical_variable not in {
        "Mw",
        "Shear_Rate",
        "Temperature",
    }:
        raise ValueError(
            "physical_variable must be Mw, "
            "Shear_Rate or Temperature"
        )

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # 1. Select 10% of unique polymer/monomer identities
    # --------------------------------------------------------

    monomers = df["Polymer"].dropna().unique()

    n_test_monomers = max(
        1,
        int(
            np.ceil(
                len(monomers)
                * test_monomer_fraction
            )
        ),
    )

    test_monomers = rng.choice(
        monomers,
        size=n_test_monomers,
        replace=False,
    )

    held_out_mask = df["Polymer"].isin(
        test_monomers
    )

    train_idx = df.index[
        ~held_out_mask
    ].to_numpy()

    test_candidate = df.loc[
        held_out_mask
    ].copy()

    # --------------------------------------------------------
    # 2. Median physical variable within each held-out monomer
    # --------------------------------------------------------

    test_indices = []
    train_indices_from_heldout = []

    for monomer, group in test_candidate.groupby(
        "Polymer"
    ):

        median_value = group[
            physical_variable
        ].median()

        lower = group[
            group[physical_variable]
            <= median_value
        ]

        upper = group[
            group[physical_variable]
            > median_value
        ]

        # Randomly decide which side becomes train
        if rng.random() < 0.5:

            train_group = lower
            test_group = upper

        else:

            train_group = upper
            test_group = lower

        train_indices_from_heldout.extend(
            train_group.index.tolist()
        )

        test_indices.extend(
            test_group.index.tolist()
        )

    # --------------------------------------------------------
    # 3. Add one side of held-out monomers to train
    # --------------------------------------------------------

    train_idx = np.concatenate([
        train_idx,
        np.asarray(
            train_indices_from_heldout,
            dtype=int,
        ),
    ])

    test_idx = np.asarray(
        test_indices,
        dtype=int,
    )

    return train_idx, test_idx