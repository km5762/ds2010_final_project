import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("second_case_study/cleaned_listings.csv")

df = df.dropna(subset=['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'price_night'])

df = df[df['review_scores_rating'] > 4]
df = df[df['number_of_reviews'] > 50]

X = df[['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates'] + [col for col in df.columns if 'room_type' in col or 'property_type' in col]]
y = df['price_night']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train_scaled, y_train)  
y_pred = model.predict(X_test_scaled)  

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs Predicted Price Per Night', fontsize=16)
plt.xlabel('Actual Price Per Night', fontsize=14)
plt.ylabel('Predicted Price Per Night', fontsize=14)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  
plt.text(0.05, 0.95, f'R^2: {r2:.3f}\nMSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()