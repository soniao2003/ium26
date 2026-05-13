import pandas as pd


def print_subset_stats(name: str, df: pd.DataFrame) -> None:
    print(f"Statystyki dla zbioru: {name}")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    print("\nStatystyki zmiennych numerycznych (mean, min, max, std, median):")
    stats = df[numeric_cols].agg(["mean", "min", "max", "std", "median"])
    print(stats)

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    if len(categorical_cols) > 0:
        print("\nRozkład zmiennych kategorycznych:")
        for col in categorical_cols:
            print(f"\n{col}:")
            counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            for value, count in counts.head(5).items():
                print(f"  {value}: {count} ({percentages[value]:.2f}%)")


def main() -> None:
    data_clean = pd.read_csv("artifacts/data_clean.csv")
    train_data = pd.read_csv("artifacts/train_data.csv")
    dev_data = pd.read_csv("artifacts/dev_data.csv")
    test_data = pd.read_csv("artifacts/test_data.csv")

    print(f"Wielkość danych (po czyszczeniu): {data_clean.shape}")
    print(f"Wielkość zbioru treningowego: {len(train_data)}")
    print(f"Wielkość zbioru walidacyjnego: {len(dev_data)}")
    print(f"Wielkość zbioru testowego: {len(test_data)}")

    print("\nLiczba wierszy: ", len(data_clean))
    print("Liczba kolumn: ", len(data_clean.columns))

    categorical_cols = data_clean.select_dtypes(include=["object", "str"]).columns
    print("\nRozkład zmiennych kategorycznych:")
    for col in categorical_cols:
        print(f"\n{col}:")
        counts = data_clean[col].value_counts()
        percentages = data_clean[col].value_counts(normalize=True) * 100
        for value, count in counts.items():
            print(f"  {value}: {count} ({percentages[value]:.2f}%)")

    numeric_cols = data_clean.select_dtypes(include=["number"]).columns
    print("\nWartości MIN i MAX dla każdej kolumny numerycznej:")
    for col in numeric_cols:
        print(f"{col}:")
        print(f"Min: {data_clean[col].min()}")
        print(f"Max: {data_clean[col].max()}")

    print("\nStatystyki zmiennych numerycznych:")
    print(data_clean[numeric_cols].describe())

    print_subset_stats("treningowy", train_data)
    print_subset_stats("walidacyjny", dev_data)
    print_subset_stats("testowy", test_data)


if __name__ == "__main__":
    main()
