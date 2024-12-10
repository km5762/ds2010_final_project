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
ratio_bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25]  # Define bin edges
ratio_bin_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-25',]  # Define labels for bins
df['ratio_bin'] = pd.cut(df['ratio'], bins=ratio_bin_edges, labels=ratio_bin_labels, include_lowest=True)
rating_bin_edges = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]  # Define bin edges
rating_bin_labels = ['0-.5', '.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']  # Define labels for bins
df['rating_bin'] = pd.cut(df['review_scores_rating'], bins=rating_bin_edges, labels=rating_bin_labels, include_lowest=True)

# Calculate average review_scores_rating for each bin
ratio_bin_analysis = df.groupby('ratio_bin')['review_scores_rating'].mean().reset_index()
rating_bin_analysis = df.groupby('rating_bin')['ratio'].mean().reset_index()
# Display the result
print(ratio_bin_analysis)
print(rating_bin_analysis)

# Create a bar chart
plt.bar(ratio_bin_analysis['ratio_bin'], ratio_bin_analysis['review_scores_rating'], color='skyblue')
plt.xlabel('Ratio Bin')
plt.ylabel('Average Review Scores Rating')
plt.title('Average Review Scores Rating by Ratio Bin')
plt.show()

plt.bar(rating_bin_analysis['rating_bin'], rating_bin_analysis['ratio'],)
plt.xlabel('Rating Bin')
plt.ylabel('Average Ratio')
plt.title('Average Ratio by Rating Bin')
plt.show()

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

