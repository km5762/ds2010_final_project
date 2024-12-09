from scipy.stats import randint
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("cleaned_listings.csv")

df = df.dropna(subset=['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'price_night', 'room_type', 'property_type'])

df_encoded = pd.get_dummies(df, columns=['room_type', 'property_type'], drop_first=True)

X = df_encoded[['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates'] + [col for col in df_encoded.columns if 'room_type' in col or 'property_type' in col]]
y = df_encoded['price_night']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rfr = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 500, 800, 1000], 
    'max_depth': [None, 10, 20, 30, 40],            
    'min_samples_split': randint(2, 20),              
    'min_samples_leaf': randint(1, 20),             
    'max_features': [1.0, 'sqrt', 'log2', None],         
}

random_search = RandomizedSearchCV(
    estimator=rfr,
    param_distributions=param_dist,
    n_iter=100,  
    cv=2,
    scoring='r2',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

best_rfr = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

y_pred = best_rfr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Optimized Random Forest: MSE={mse:.4f}, R²={r2:.4f}")
