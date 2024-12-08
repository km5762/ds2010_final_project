from math import sqrt
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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


def main() -> None:
    df = pd.read_csv(data_path, dtype=dtype_mapping, parse_dates=["host_since"])
    df.drop(columns=["id", "neighbourhood_cleansed"], inplace=True)
    df["host_since"] = pd.to_numeric(df["host_since"])

    lasso(df)


def lasso(df: pd.DataFrame):
    X = df.drop(columns=["price_night"])
    y = df["price_night"]  # Target variable

    # Might mess up reviews
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tune alpha hyper-parameter
    param_grid = {"alpha": np.linspace(start=0.6, stop=0.8, num=30)}  # best alpha seems to be 0.6482758620689655
    grid_search = GridSearchCV(
        Lasso(max_iter=9000, random_state=42), param_grid, cv=5, n_jobs=8
    )

    grid_search.fit(X_train_scaled, y_train)

    alpha = grid_search.best_params_["alpha"]
    print(f"Best alpha found was : {alpha}")

    # lasso = Lasso(
    #     alpha=alpha,
    # )
    # lasso.fit(X_train_scaled, y_train)

    y_pred = grid_search.predict(X_test_scaled)

    best_lasso = grid_search.best_estimator_

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")
    r2 = best_lasso.score(X_test_scaled, y_test)
    print(f"R-squared: {r2}")

    # Most impactful features
    coefficients = pd.Series(best_lasso.coef_, index=X.columns)
    print(
        coefficients.where(abs(coefficients) > 0.0001)
        .dropna()
        .sort_values(key=lambda value: abs(value))
    )


if __name__ == "__main__":
    assert data_path.exists()
    main()
