from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ARTIFACTS_DIR = Path("artifacts")
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "revenue_torch_model.pt"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
TARGET_COLUMN = "Revenue"


class RevenueRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET_COLUMN, "Margin"], errors="ignore").copy()

    if "Date" in features.columns:
        date_series = pd.to_datetime(features["Date"], errors="coerce")
        features["Date"] = date_series.map(lambda d: d.toordinal() if pd.notna(d) else -1)

    return features


def prepare_test_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    encoded = pd.get_dummies(df, drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)

    x = encoded.astype(np.float32).to_numpy()
    x = (x - feature_mean) / feature_std

    return x


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Brak artifacts/revenue_torch_model.pt. Najpierw uruchom train_nn.py"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Brak artifacts/test_data.csv. Najpierw uruchom preprocess_data.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    feature_columns = checkpoint["feature_columns"]
    feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
    feature_std[feature_std == 0.0] = 1.0

    model = RevenueRegressor(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_df = pd.read_csv(TEST_DATA_PATH)
    x_test = build_feature_frame(test_df)
    x_test_matrix = prepare_test_matrix(x_test, feature_columns, feature_mean, feature_std)

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_test_matrix)
        predictions = model(x_tensor).squeeze(1).numpy()

    output_df = pd.DataFrame(
        {
            "row_id": range(len(predictions)),
            "predicted_revenue": predictions,
        }
    )
    output_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Wczytano model: {MODEL_PATH}")
    print(f"Wczytano dane testowe: {TEST_DATA_PATH}")
    print(f"Zapisano predykcje do: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
