import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path(__file__).parent / "cleaned_listings.csv"

dtype_mapping = {
    "id": "int64",
    "host_response_time": np.int8,
    "host_response_rate": "float32",
    "host_is_superhost": "bool",
    "neighbourhood_cleansed": "string",
    "property_type": "string",
    "room_type": "string",
    "accommodates": "int16",
    "bathrooms": "float32",
    "bedrooms": "float32",
    "beds": "float32",
    "price_night": "int64",
    "minimum_nights": "int32",
    "availability_365": "int8",
    "number_of_reviews": "int64",
    "review_scores_rating": "float32",
    "instant_bookable": "bool",
    "crime_rate": "int64",
    "median_housing_cost": "int64",
}


def main() -> None:
    df = pd.read_csv(data_path, dtype=dtype_mapping, parse_dates=["host_since"])
    df.drop(columns=["id", "neighbourhood_cleansed"], inplace=True)

    create_corr_plot(df)


def create_corr_plot(df: pd.DataFrame) -> None:
    corr_matrix = df.corr()
    price_corr = corr_matrix["price_night"].drop("price_night")

    cmap = plt.colormaps["coolwarm"]

    def rescale(y):
        return (y - np.min(y)) / (np.max(y) - np.min(y))

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=price_corr.index,
        y=price_corr.values,
        palette=cmap(rescale(list(price_corr.values))).tolist(),
        hue=price_corr.index,
        legend=False,
    )
    plt.xticks(rotation=90)
    plt.ylabel("Correlation with price per night")
    plt.title("Correlation of price per night with other features")
    plt.subplots_adjust(left=0.05, top=0.95, bottom=0.4, right=0.95)
    plt.show()


if __name__ == "__main__":
    assert data_path.exists()
    main()
