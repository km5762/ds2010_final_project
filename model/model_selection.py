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

df = pd.read_csv("cleaned_listings.csv")

df = df.dropna(subset=['median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'price_night'])

# df = df[df['review_scores_rating'] > 4]
# df = df[df['number_of_reviews'] > 50]

X = df[['crime_rate', 'median_housing_cost', 'bathrooms', 'bedrooms', 'beds', 'accommodates'] + [col for col in df.columns if 'room_type' in col or 'property_type' in col]]
y = df['price_night']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
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
    
model_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in model_names]
r2_values = [results[name]['R2'] for name in model_names]

x = np.arange(len(model_names)) 

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(x - 0.2, mse_values, 0.4, label='MSE', color='red')
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Squared Error', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.bar(x + 0.2, r2_values, 0.4, label='R²', color='blue')
ax2.set_ylabel('R² Score', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.set_title('Model Performance: MSE and R² for Each Model')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45)

# Show plot
fig.tight_layout()
plt.show()

