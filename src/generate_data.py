from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dummy_housing_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic housing data for regression training."""
    rng = np.random.default_rng(random_state)

    luas_bangunan = rng.normal(loc=120, scale=40, size=n_samples).clip(30, 400)
    luas_tanah = (luas_bangunan * rng.normal(1.4, 0.25, size=n_samples)).clip(50, 600)
    jumlah_kamar = rng.integers(1, 7, size=n_samples)
    jumlah_kamar_mandi = rng.integers(1, 5, size=n_samples)
    usia_rumah = rng.integers(0, 50, size=n_samples)
    jarak_ke_pusat_kota = rng.normal(10, 6, size=n_samples).clip(0.5, 40)

    # Non-linear price function + noise to mimic real market behavior.
    base_price = (
        (luas_bangunan * 4_200_000)
        + (luas_tanah * 1_650_000)
        + (jumlah_kamar * 95_000_000)
        + (jumlah_kamar_mandi * 70_000_000)
        - (usia_rumah * 13_500_000)
        - (jarak_ke_pusat_kota * 24_000_000)
    )

    # Interaction effect: larger homes near city center get premium.
    premium = (luas_bangunan * np.maximum(0, 12 - jarak_ke_pusat_kota) * 350_000)
    noise = rng.normal(0, 180_000_000, size=n_samples)
    harga_rumah = (base_price + premium + noise).clip(250_000_000, None)

    data = pd.DataFrame(
        {
            "luas_bangunan": luas_bangunan.round(2),
            "luas_tanah": luas_tanah.round(2),
            "jumlah_kamar": jumlah_kamar,
            "jumlah_kamar_mandi": jumlah_kamar_mandi,
            "usia_rumah": usia_rumah,
            "jarak_ke_pusat_kota": jarak_ke_pusat_kota.round(2),
            "harga_rumah": harga_rumah.round(2),
        }
    )

    return data


def main() -> None:
    output_dir = Path("artifacts/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = create_dummy_housing_data(n_samples=5000, random_state=42)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved: {train_path} | shape={train_df.shape}")
    print(f"Test data saved:  {test_path} | shape={test_df.shape}")


if __name__ == "__main__":
    main()
