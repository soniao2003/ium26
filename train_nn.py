from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ARTIFACTS_DIR = Path("artifacts")
TRAIN_DATA_PATH = ARTIFACTS_DIR / "train_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "revenue_torch_model.pt"
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


def prepare_training_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    encoded = pd.get_dummies(df, drop_first=False)
    feature_columns = encoded.columns.tolist()

    x = encoded.astype(np.float32).to_numpy()
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0

    x = (x - feature_mean) / feature_std

    return x, feature_mean, feature_std, feature_columns


def main() -> None:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            "Brak artifacts/train_data.csv. Najpierw uruchom preprocess_data.py"
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df = train_df.dropna(subset=[TARGET_COLUMN])

    x_train = build_feature_frame(train_df)
    y_train = train_df[TARGET_COLUMN].astype(np.float32).to_numpy().reshape(-1, 1)

    x_matrix, feature_mean, feature_std, feature_columns = prepare_training_matrix(x_train)

    x_tensor = torch.from_numpy(x_matrix)
    y_tensor = torch.from_numpy(y_train)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = RevenueRegressor(input_dim=x_matrix.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(80):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": x_matrix.shape[1],
        "feature_columns": feature_columns,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_column": TARGET_COLUMN,
    }
    torch.save(checkpoint, MODEL_PATH)

    print(f"Wczytano dane treningowe: {TRAIN_DATA_PATH}")
    print(f"Liczba próbek treningowych: {len(train_df)}")
    print(f"Model zapisano do: {MODEL_PATH}")


if __name__ == "__main__":
    main()
