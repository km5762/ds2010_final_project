import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = pd.read_csv('dataset/listings_dataset.csv')

#converting to dataframe
df = pd.DataFrame(dataset)

#rename column
df.rename(columns={'accomodates/bathrooms ratio' : 'ratio'}, inplace=True)
print(df['ratio'].head(20))
print(df.columns)

# Select 'ratio' and 'review_scores_rating' columns
df = df[['ratio', 'review_scores_rating']]

#converting to numeric
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')

#drop null values
df = df.dropna()

correlation = df['ratio'].corr(df['review_scores_rating'])

print(f"Correlation between 'ratio' and 'review_scores_rating': {correlation}")

#regression line
x = df['ratio']
y = df['review_scores_rating']

# Fit a regression line
coefficients = np.polyfit(x, y, deg=1)
slope, intercept = coefficients
regression_line = slope * x + intercept

# Compute predictions using the regression line
y_pred = slope * x + intercept

# Calculate R^2
r2 = r2_score(y, y_pred)

# Calculate MSE and RMSE
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Calculate MAE
mae = mean_absolute_error(y, y_pred)

# Print metrics
print(f"R^2: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plot the two columns as a line plot
df.plot.scatter(x='ratio', y='review_scores_rating')
plt.plot(x, regression_line, color='red', label=f'Regression Line (y={slope:.2f}x + {intercept:.2f})')

# Add labels and title
plt.title("Scatter Plot with Regression Line")
plt.xlabel("Ratio")
plt.ylabel("Review Scores Rating")
plt.legend()

# Show the plot
plt.show()

