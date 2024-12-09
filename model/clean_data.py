import pandas as pd

from pathlib import Path

data_path = Path(__file__).parent / "listings.csv"
crime_path = Path(__file__).parent / "crime_house-price_by_neighbourhood.csv"
output_path = Path(__file__).parent / "cleaned_listings.csv"

data_dtype_mapping = {
    "id": "int64",
    "host_response_time": "string",  # int
    "host_response_rate": "float64",
    "host_is_superhost": "object",  # bool
    "neighbourhood_cleansed": "string",
    "property_type": "string",
    "room_type": "string",
    "accommodates": "int64",
    "bathrooms": "float64",
    "bedrooms": "float64",
    "beds": "float64",
    "price_night": "object",  # int64
    "minimum_nights": "int64",
    "availability_365": "int64",
    "number_of_reviews": "int64",
    "review_scores_rating": "float64",
    "instant_bookable": "object",  # bool
    "crime_rate": "int64",
    "median_housing_cost": "int64",
}

response_time_mapping = {
    "within an hour": 1,
    "within a few hours": 2,
    "within a day": 3,
    "a few days or more": 4,
}


def create_df() -> pd.DataFrame:
    df = pd.read_csv(data_path, dtype=data_dtype_mapping)

    df["parsed_date"] = pd.to_datetime(
        df["host_since"], format="%d/%m/%y", errors="coerce"
    )
    df["parsed_date"] = df["parsed_date"].fillna(
        pd.to_datetime(df["host_since"], format="%d/%m/%Y", errors="coerce")
    )
    df.drop(columns=["host_since"], inplace=True)
    df.rename(columns={"parsed_date": "host_since"}, inplace=True)

    # Ordinal encoding for categorical variable
    df["host_response_time"] = df["host_response_time"].map(response_time_mapping)

    # If the listing does not have a price then its useless
    # If there is no superhost or instant bookable, there is a data input error
    # Rest is just to get clean data
    df = df.dropna(
        subset=[
            "price_night",
            "host_is_superhost",
            "instant_bookable",
            "host_response_time",
            "host_since",
            "property_type",
            "room_type",
        ]
    )

    df["host_response_time"] = df["host_response_time"].astype(int)

    df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})
    df["instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False})

    df = pd.get_dummies(df, columns=["property_type"])
    df = pd.get_dummies(df, columns=["room_type"])

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


def main() -> None:
    if not output_path.exists():
        print("Writing cleaned data")
        create_df().to_csv(output_path, index=False)
    else:
        print(
            f"Data is already cleaned. delete {output_path.absolute()} to regenerate."
        )


if __name__ == "__main__":
    assert data_path.exists()
    assert crime_path.exists()
    main()
