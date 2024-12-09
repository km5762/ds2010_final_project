from math import sqrt
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# https://mljar.com/blog/save-load-random-forest/

pd.set_option("display.max_rows", None)

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

random_forest = RandomForestRegressor(
    random_state=42,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=15,
    n_estimators=600,
    n_jobs=-1,
    verbose=1,
)


def main() -> None:
    df = pd.read_csv(data_path, dtype=dtype_mapping, parse_dates=["host_since"])
    df.drop(columns=["id", "neighbourhood_cleansed"], inplace=True)
    df["host_since"] = pd.to_numeric(
        df["host_since"]
    )  # So the model can work with the dates

    df = df.where(df["number_of_reviews"] >= 15).dropna(
        subset=["number_of_reviews"]
    )  # Simulate being in areas where we get enough people to review
    df = df.where(
        df["review_scores_rating"] >= 4
    ).dropna(
        subset=["review_scores_rating"]
    )  # Business doesnt plan on being a bad owner, so we want to simulate our data having decent landlords

    # testing dropping host columns
    df.drop(
        columns=[
            "number_of_reviews",
            "review_scores_rating",
            # "host_since",
            # "host_response_rate",
            # "host_response_time",
            # "host_is_superhost",
        ],
        inplace=True,
    )  # we want quality listings, but we dont want the qualifying columns

    df.dropna(
        subset=[
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds",
        ],
        inplace=True,
    )

    # While we may be losing oddball properties, such as the one random villa overlooking the la basin thats 3k a night, this will do well for most houses in la county
    mean_y = df["price_night"].mean()
    std_y = df["price_night"].std()
    df["z_score"] = (df["price_night"] - mean_y) / std_y

    df = df[np.abs(df["z_score"]) <= 3.5]
    df = df.drop(columns=["z_score"])

    print(f"Num samples: {len(df)}")
    print(f"Min price: {df["price_night"].min()}")
    print(f"Max price: {df["price_night"].max()}")

    forest(df)


def forest(df: pd.DataFrame):
    X = df.drop(columns=["price_night"])
    y = df["price_night"]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    random_forest.fit(X_train, y_train)
    cv_scores = cross_val_score(random_forest, X, y, cv=5, scoring="r2")

    y_pred = random_forest.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean R² score: {np.mean(cv_scores):.4f}")
    print("-" * 30)


if __name__ == "__main__":
    assert data_path.exists()
    main()

# when getting rid of host columns
# MSE: 9037.41657138136
# RMSE: 95.06532791392117
# Cross-validation scores: [0.63984009 0.66577678 0.6060663  0.71070762 0.65461769]
# Mean R² score: 0.6554

# with host columns
# MSE: 9049.288645178214
# RMSE: 95.12774908079248
# Cross-validation scores: [0.63992165 0.6687529  0.60627109 0.71011109 0.65430674]
# Mean R² score: 0.6559
