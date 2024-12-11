import pandas as pd

listings = pd.read_excel("second_case_study/listings.xlsx", sheet_name="listings")
augmented = pd.read_excel("second_case_study/augmented.xlsx")

augmented_listings = pd.merge(listings, augmented, "inner", "neighbourhood_cleansed")

augmented_listings = augmented_listings.dropna(subset=["crime_rate", "median_housing_cost"])

crime_rate_threshold = augmented_listings["crime_rate"].quantile(0.95)
housing_cost_threshold = augmented_listings["median_housing_cost"].quantile(0.95)

augmented_listings = augmented_listings[
    (augmented_listings["crime_rate"] <= crime_rate_threshold) & 
    (augmented_listings["median_housing_cost"] <= housing_cost_threshold)
]

augmented_listings.to_excel("second_case_study/augmented_listings.xlsx", index=False)
