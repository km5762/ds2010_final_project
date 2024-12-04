import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


def display_corr_mat() -> None:
    df = pd.read_csv(Path(__file__).parent.parent / "dataset" / "listings_dataset.csv")
    df["host_since"] = pd.to_datetime(
        df["host_since"], format="%d/%m/%y", errors="coerce"
    )
    df["host_is_superhost"] = (
        df["host_is_superhost"].dropna().apply(lambda is_superhost: is_superhost == "t")
    )
    df["instant_bookable"] = (
        df["instant_bookable"].dropna().apply(lambda is_superhost: is_superhost == "t")
    )
    df = df.drop(
        columns=[
            "id",
            "host_response_time",
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "accomodates/bathrooms ratio",
        ]
    )
    print(df["host_is_superhost"].dtype)

    correlation_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def main() -> None:
    display_corr_mat()


if __name__ == "__main__":
    main()
