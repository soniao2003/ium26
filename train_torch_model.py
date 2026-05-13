import os
import tempfile
from datetime import datetime

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def prepare_tensors(
    df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features_df = df.drop(columns=[target_col])

    features_df = pd.get_dummies(features_df, drop_first=False)

    if feature_cols is not None:
        features_df = features_df.reindex(columns=feature_cols, fill_value=0)
    else:
        feature_cols = list(features_df.columns)

    x = features_df.to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    return x_tensor, y_tensor, feature_cols


def write_model_card(
    path: str,
    registered_model_name: str,
    selected_model_uri: str,
    registration_status: str,
    registered_version: str,
    train_samples: int,
    dev_samples: int,
    input_features: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    final_train_mse: float,
    final_dev_mse: float,
    final_dev_rmse: float,
    final_dev_mae: float,
) -> None:
    model_card = f"""# Model Card: Revenue Regressor v1

## Model Details
- Framework: PyTorch
- Model class: RevenueRegressor
- Registered model name: {registered_model_name}
- Selected model URI: {selected_model_uri}
- Registration status: {registration_status}
- Registered version: {registered_version or 'n/a'}
- Generated at (UTC): {datetime.utcnow().isoformat()}Z

## Intended Use
- **Primary Use**: Revenue forecasting (regression)
- **Use Cases**: Sales analysis, planning support

## Performance
- **Train MSE**: {final_train_mse:.6f}
- **Dev MSE**: {final_dev_mse:.6f}
- **Dev RMSE**: {final_dev_rmse:.6f}
- **Dev MAE**: {final_dev_mae:.6f}

## Data
- **Dataset**: Ferrero sales dataset (preprocessed)
- **Train Samples**: {train_samples}
- **Dev Samples**: {dev_samples}
- **Features**: {input_features} (after one-hot encoding)

## Training Setup
- **Epochs**: {num_epochs}
- **Batch Size**: {batch_size}
- **Learning Rate**: {learning_rate}
- **Optimizer**: Adam
- **Loss**: MSE

## Limitations
- Performance can degrade on data distribution shifts.
- Current pipeline assumes preprocessing consistency between train and inference.

## Maintenance
- **Monitoring**: Per-run metric tracking in MLflow.
- **Retraining**: Recommended when dev metrics significantly degrade.
- **Contact**: data-science@company.com
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(model_card)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    train_path = os.path.join("artifacts", "train_data_norm.csv")
    dev_path = os.path.join("artifacts", "dev_data_norm.csv")

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError(
            "Brak przetworzonych danych. Uruchom najpierw preprocess_data.py, "
            "aby wygenerować pliki artifacts/train_data_norm.csv i artifacts/dev_data_norm.csv."
        )

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    target_col = "Revenue"

    x_train, y_train, feature_cols = prepare_tensors(train_df, target_col)
    x_dev, y_dev, _ = prepare_tensors(dev_df, target_col, feature_cols)

    batch_size = 32
    learning_rate = 1e-3
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RevenueRegressor(in_features=x_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs_str = os.getenv("NUM_EPOCHS", "10")
    try:
        num_epochs = int(num_epochs_str)
    except ValueError:
        num_epochs = 10

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow_registry.db")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "revenue_torch_regressor")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "ferrero_revenue_forecasting")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="torch_revenue_regressor") as run:
        mlflow.log_params(
            {
                "model": "RevenueRegressor",
                "optimizer": "Adam",
                "criterion": "MSELoss",
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_layers": "[32,16]",
                "input_features": x_train.shape[1],
                "train_samples": len(train_df),
                "dev_samples": len(dev_df),
                "device": str(device),
            }
        )

        final_train_mse = 0.0
        final_dev_mse = 0.0
        final_dev_rmse = 0.0
        final_dev_mae = 0.0

        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                current_batch_size = batch_x.size(0)
                epoch_loss += loss.item() * current_batch_size

            train_mse = epoch_loss / len(train_dataset)

            model.eval()
            with torch.no_grad():
                dev_preds_tensor = model(x_dev.to(device))
                dev_mse = criterion(dev_preds_tensor, y_dev.to(device)).item()

            dev_preds_np = dev_preds_tensor.cpu().numpy().squeeze()
            y_dev_np = y_dev.cpu().numpy().squeeze()
            dev_rmse = float(np.sqrt(dev_mse))
            dev_mae = float(mean_absolute_error(y_dev_np, dev_preds_np))

            final_train_mse = train_mse
            final_dev_mse = dev_mse
            final_dev_rmse = dev_rmse
            final_dev_mae = dev_mae

            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "dev_mse": dev_mse,
                    "dev_rmse": dev_rmse,
                    "dev_mae": dev_mae,
                },
                step=epoch,
            )

            print(
                f"Epoka {epoch} - train MSE: {train_mse:.6f} - "
                f"dev MSE: {dev_mse:.6f} - dev RMSE: {dev_rmse:.6f} - dev MAE: {dev_mae:.6f}"
            )

        os.makedirs("artifacts", exist_ok=True)
        model_path = os.path.join("artifacts", "torch_revenue_model.pth")
        torch.save(model.state_dict(), model_path)

        meta_path = os.path.join("artifacts", "torch_revenue_model_features.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            for col in feature_cols:
                f.write(col + "\n")

        mlflow.log_artifact(meta_path, artifact_path="metadata")
        model_info = mlflow.pytorch.log_model(model, artifact_path="model")

        run_model_uri = model_info.model_uri
        selected_model_uri = run_model_uri
        registration_status = "not_registered"
        registered_version = ""
        try:
            model_version = mlflow.register_model(model_uri=run_model_uri, name=registered_model_name)
            registered_version = str(model_version.version)
            selected_model_uri = f"models:/{registered_model_name}/{registered_version}"
            registration_status = "registered"
            model_card_content = read_text_file(model_card_path)
            client = MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_version,
                description=model_card_content,
            )
            print(
                f"Zarejestrowano model w Model Registry: {registered_model_name} "
                f"(wersja {registered_version})"
            )
        except Exception as exc:
            registration_status = f"registration_failed: {type(exc).__name__}"
            print(f"Ostrzezenie: nie udalo sie zarejestrowac modelu w Registry: {exc}")

        registry_info_path = os.path.join("artifacts", "torch_revenue_model_registry_info.txt")
        with open(registry_info_path, "w", encoding="utf-8") as f:
            f.write(f"registered_model_name={registered_model_name}\n")
            f.write(f"registered_version={registered_version or 'n/a'}\n")
            f.write(f"registration_status={registration_status}\n")
            f.write(f"run_model_uri={run_model_uri}\n")
            f.write(f"selected_model_uri={selected_model_uri}\n")

        model_card_path = os.path.join("artifacts", "torch_revenue_model_card.md")
        write_model_card(
            path=model_card_path,
            registered_model_name=registered_model_name,
            selected_model_uri=selected_model_uri,
            registration_status=registration_status,
            registered_version=registered_version,
            train_samples=len(train_df),
            dev_samples=len(dev_df),
            input_features=x_train.shape[1],
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_train_mse=final_train_mse,
            final_dev_mse=final_dev_mse,
            final_dev_rmse=final_dev_rmse,
            final_dev_mae=final_dev_mae,
        )
        mlflow.log_artifact(registry_info_path, artifact_path="metadata")

        model_card_content = read_text_file(model_card_path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(model_card_content)
            temp_model_card_path = f.name
        try:
            mlflow.log_artifact(temp_model_card_path, artifact_path="documentation")
        finally:
            os.remove(temp_model_card_path)

        model_uri_path = os.path.join("artifacts", "torch_revenue_mlflow_model_uri.txt")
        with open(model_uri_path, "w", encoding="utf-8") as f:
            f.write(run_model_uri + "\n")

        print(f"Zapisano model do pliku: {model_path}")
        print(f"Zapisano listę cech do pliku: {meta_path}")
        print(f"Zapisano informacje o rejestrze modelu do pliku: {registry_info_path}")
        print(f"Zapisano model card do pliku: {model_card_path}")
        print(f"Zapisano URI modelu MLflow do pliku: {model_uri_path}")
        print(f"Model MLflow URI: {selected_model_uri}")


if __name__ == "__main__":
    main()
