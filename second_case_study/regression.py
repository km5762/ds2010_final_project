import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("augmented_listings.xlsx")

df.dropna(subset=['price_night', 'crime_rate', 'median_housing_cost'], inplace=True)

X = df[['crime_rate', 'median_housing_cost']]  
y = df['price_night']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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
