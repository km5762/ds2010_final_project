import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

data_path = Path(__file__).parent / "listings.csv"
crime_path = Path(__file__).parent / "crime_house-price_by_neighbourhood.csv"


def create_df() -> pd.DataFrame:
    df = pd.read_csv(
        data_path, dtype={"id": int}, parse_dates=["host_since"], date_format="%d/%m/%y"
    )

    # If the listing does not have a price then its useless
    # If there is no superhost or instant bookable, there is a data input error
    response_mapping = {
        "within an hour": 1,
        "within a few hours": 2,
        "within a day": 3,
        "a few days or more": 4,
    }

    df["host_response_time"] = df["host_response_time"].map(response_mapping)

    df = df.dropna(
        subset=[
            "price_night",
            "host_is_superhost",
            "instant_bookable",
            "host_response_time",
        ]
    )

    df["host_is_superhost"].apply(lambda is_superhost: is_superhost == "t")
    df["instant_bookable"].apply(lambda is_superhost: is_superhost == "t")

    # Id is only in the dataset for meta analysis purposes; not useful for actual analysis.
    df = df.drop(
        columns=[
            "id",
        ]
    )

    # Inner join with crime dataset
    crime_housing_df = pd.read_csv(
        Path(__file__).parent / "crime_house-price_by_neighbourhood.csv"
    )

    df = pd.merge(
        df,
        crime_housing_df,
        how="inner",
        on="neighbourhood_cleansed",
    )

    # Meta column
    df = df.drop(columns=["neighbourhood_link"])

    return df


def plot_mat(mat):
    plt.figure(figsize=(12, 8))
    sns.heatmap(mat, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def main() -> None:
    df = create_df()


if __name__ == "__main__":
    assert data_path.exists()
    assert crime_path.exists()
    main()


# df = pd.get_dummies(df, columns=["property_type"], drop_first=True)
# df = pd.get_dummies(df, columns=["room_type"], drop_first=True)

# correlation_matrix = df.corr()

# # Mask the lower triangle to avoid duplicate correlations
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# # Get the upper triangle of the correlation matrix (excluding diagonal)
# corr_masked = correlation_matrix.where(mask)

# # Get the correlations with 'price' column (excluding 'price' itself)
# corr_with_price = corr_masked["price_night"].drop("price_night")

# # Sort the correlations in descending order
# top_n = corr_with_price.sort_values(ascending=False).head(5)

# # Sort the correlations in descending order
# # top_n = corr_values.sort_values(ascending=False).head(5)
# print(top_n)
