import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("second_case_study/cleaned_listings.csv")

df = df.dropna(subset=['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'price_night', 'room_type', 'property_type'])

df = df[df['review_scores_rating'] > 4]
df = df[df['number_of_reviews'] > 50]

df_encoded = pd.get_dummies(df, columns=['room_type', 'property_type'], drop_first=True)

X = df_encoded[['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates'] + [col for col in df_encoded.columns if 'room_type' in col or 'property_type' in col]]
y = df_encoded['price_night']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1),
    'SVR': SVR(kernel='linear'),
    'RandomForest': RandomForestRegressor(n_estimators=500, random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)  
    y_pred = model.predict(X_test_scaled)  
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R2': r2}

for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, R²={metrics['R2']:.4f}")
