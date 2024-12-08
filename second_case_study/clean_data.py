import pandas as pd

from pathlib import Path

data_path = Path(__file__).parent / "listings.csv"
crime_path = Path(__file__).parent / "crime_house-price_by_neighbourhood.csv"
output_path = Path(__file__).parent / "cleaned_listings.csv"


def create_df() -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["host_since"], date_format="%d/%m/%y")

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
        df = create_df()
        df.to_csv(output_path, index=False)
    else:
        print(
            f"Data is already cleaned. delete {output_path.absolute()} to regenerate."
        )


if __name__ == "__main__":
    assert data_path.exists()
    assert crime_path.exists()
    main()
