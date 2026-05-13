import os

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch import nn


class RevenueRegressor(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_feature_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model_uri(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        value = f.read().strip()
    if not value:
        raise ValueError("Plik z URI modelu MLflow jest pusty.")
    return value


def main() -> None:
    test_norm_path = os.path.join("artifacts", "test_data_norm.csv")
    model_path = os.path.join("artifacts", "torch_revenue_model.pth")
    model_uri_path = os.path.join("artifacts", "torch_revenue_mlflow_model_uri.txt")
    features_path = os.path.join("artifacts", "torch_revenue_model_features.txt")

    if not os.path.exists(test_norm_path):
        raise FileNotFoundError(
            "Brak przetworzonych danych testowych. Uruchom najpierw preprocess_data.py, "
            "aby wygenerować plik artifacts/test_data_norm.csv."
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            "Brak pliku z listą cech. Upewnij się, że train_torch_model.py został "
            "uruchomiony do końca i poprawnie zapisał artifacts/torch_revenue_model_features.txt."
        )

    test_df = pd.read_csv(test_norm_path)
    feature_cols = load_feature_names(features_path)

    if "Revenue" in test_df.columns:
        features_df = test_df.drop(columns=["Revenue"])
    else:
        features_df = test_df.copy()

    features_df = pd.get_dummies(features_df, drop_first=False)
    features_df = features_df.reindex(columns=feature_cols, fill_value=0)

    x_test = features_df.to_numpy(dtype=np.float32)
    x_test_tensor = torch.from_numpy(x_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow_registry.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    model_uri = os.getenv("MLFLOW_MODEL_URI")
    if model_uri is None and os.path.exists(model_uri_path):
        model_uri = load_model_uri(model_uri_path)

    if model_uri:
        model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
        print(f"Wczytano model MLflow z URI: {model_uri}")
    elif os.path.exists(model_path):
        # Fallback dla starszych artefaktów zapisanych bez MLflow.
        model = RevenueRegressor(in_features=x_test.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Wczytano model z pliku state_dict: {model_path}")
    else:
        raise FileNotFoundError(
            "Brak modelu do inferencji. Uruchom najpierw train_torch_model.py, aby "
            "zapisać model MLflow lub artifacts/torch_revenue_model.pth."
        )

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(x_test_tensor.to(device)).cpu().numpy().squeeze()

    output_path = os.path.join("artifacts", "torch_revenue_predictions.csv")
    preds_df = pd.DataFrame({"Predicted_Revenue": preds})
    preds_df.to_csv(output_path, index=False)

    print(f"Zapisano predykcje do pliku: {output_path}")


if __name__ == "__main__":
    main()
