from math import sqrt
import pandas as pd
from pathlib import Path
import numpy as np  # noqa: E402
# from sklearnex import patch_sklearn

# patch_sklearn()
# https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html

from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import Lasso, ElasticNet  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

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

pipelines = {
    "elasticnet": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=9000, random_state=42)),
        ]
    ),
    "lasso": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=9000, random_state=42)),
        ]
    ),
    "decision_tree": Pipeline([("model", DecisionTreeRegressor(random_state=42))]),
    "svr": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
    "random_forest": Pipeline(
        [("model", RandomForestRegressor(random_state=42))]
    ),  # takes the longest with the params i have
}

param_grids = {
    "elasticnet": {  # rerun, worse with intel
        "model__alpha": np.linspace(start=0.1, stop=1, num=30),
        "model__l1_ratio": np.linspace(start=0.1, stop=1, num=30),
    },
    "lasso": {  # rerun, worse with intel
        "model__alpha": np.linspace(start=0.1, stop=1, num=30),
    },
    "svr": {
        "model__C": np.linspace(start=0.1, stop=100, num=8),
        "model__epsilon": [0.01, 0.1, 1],
        "model__kernel": ["linear", "rbf"],
    },
    "decision_tree": {  # rerun
        "model__max_depth": [None, 10, 20, 30, 40, 50],
        "model__min_samples_split": [2, 5, 10, 15, 20, 30],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 10, 20, 25],
        "model__min_samples_split": [2, 5, 10],
    },
}


def main() -> None:
    df = pd.read_csv(data_path, dtype=dtype_mapping, parse_dates=["host_since"])
    df.drop(columns=["id", "neighbourhood_cleansed"], inplace=True)
    df["host_since"] = pd.to_numeric(df["host_since"])

    pipeline(df)


def pipeline(df: pd.DataFrame):
    X = df.drop(columns=["price_night"])
    y = df["price_night"]  # Target variable

    # might mess up some columns but those should have low correlation anyway
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results: dict[str, GridSearchCV[Pipeline]] = {}

    for name, pipeline in pipelines.items():
        print(f"Running GridSearchCV for {name}...")
        grid_search = GridSearchCV(
            pipeline, param_grids[name], cv=5, n_jobs=15, verbose=2
        )
        grid_search.fit(X_train, y_train)

        results[name] = grid_search

        # for name, grid_search in results.items():
        y_pred = grid_search.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation score for {name}: {grid_search.best_score_}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"R^2: {grid_search.best_estimator_.score(X_test, y_test)}")
        print("-" * 30)


if __name__ == "__main__":
    assert data_path.exists()
    main()


# Best parameters for elasticnet: {'model__alpha': np.float64(0.1), 'model__l1_ratio': np.float64(0.41034482758620694)}
# Best cross-validation score for elasticnet: 0.38547878096355065
# MSE: 214596.13878445583
# RMSE: 463.24522532289075
# R^2: 0.3210429186804712

# Best parameters for decision_tree: {'model__max_depth': 10, 'model__min_samples_split': 10}
# Best cross-validation score for decision_tree: 0.44333948726312433
# Mean Squared Error: 204004.71641472253
# Root Mean Squared Error: 451.66881275412686

# this took like an hour
# Best parameters for random_forest: {'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 200}
# Best cross-validation score for random_forest: 0.635616362556353
# Mean Squared Error: 127602.11567634382
# Root Mean Squared Error: 357.2143833559111

# 15 or so min
# Best parameters for svr: {'model__C': np.float64(100.0), 'model__epsilon': 0.01, 'model__kernel': 'rbf'}
# Best cross-validation score for svr: 0.3743219964923488
# Mean Squared Error: 192166.48277136552
# Root Mean Squared Error: 438.36797644372416
