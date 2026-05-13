import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    data = pd.read_csv("ferrero_rocher_sales_dataset.csv")
    print(f"Wielkość danych: {data.shape}")

    data_clean = data.copy()
    data_clean = data_clean.drop_duplicates()
    data_clean["Date"] = pd.to_datetime(
        data_clean["Date"], origin="1899-12-30", unit="D"
    )
    data_clean = data_clean.dropna(
        subset=[
            col for col in data_clean.columns if data_clean[col].name != "Promotion"
        ]
    )
    data_clean["Promotion"] = data_clean["Promotion"].fillna("No Promotion")

    train_data, temp_data = train_test_split(data_clean, test_size=0.2, random_state=42)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Wielkość zbioru treningowego: {len(train_data)}")
    print(f"Wielkość zbioru walidacyjnego: {len(dev_data)}")
    print(f"Wielkość zbioru testowego: {len(test_data)}")

    numeric_cols = train_data.select_dtypes(include=["number"]).columns
    scaler = MinMaxScaler()

    train_data_norm = train_data.copy()
    dev_data_norm = dev_data.copy()
    test_data_norm = test_data.copy()

    train_data_norm[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
    dev_data_norm[numeric_cols] = scaler.transform(dev_data[numeric_cols])
    test_data_norm[numeric_cols] = scaler.transform(test_data[numeric_cols])

    os.makedirs("artifacts", exist_ok=True)

    data_clean.to_csv("artifacts/data_clean.csv", index=False)
    train_data.to_csv("artifacts/train_data.csv", index=False)
    dev_data.to_csv("artifacts/dev_data.csv", index=False)
    test_data.to_csv("artifacts/test_data.csv", index=False)

    train_data_norm.to_csv("artifacts/train_data_norm.csv", index=False)
    dev_data_norm.to_csv("artifacts/dev_data_norm.csv", index=False)
    test_data_norm.to_csv("artifacts/test_data_norm.csv", index=False)

    print("Zapisano artefakty w katalogu 'artifacts'")


if __name__ == "__main__":
    main()
