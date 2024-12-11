import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns


dataset = pd.read_csv('dataset/listings_dataset.csv')

# Converting to DataFrame
df = pd.DataFrame(dataset)

# Rename column
df.rename(columns={'accomodates/bathrooms ratio': 'ratio'}, inplace=True)

# Select relevant columns
df = df[['ratio', 'review_scores_rating', 'accommodates', 'bathrooms']]

# Converting to numeric
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')
df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

# Drop null values
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


# Calculate and print correlation between each variable and rating
# correlation_ratio = df['ratio'].corr(df['review_scores_rating'])
# correlation_accommodates = df['accommodates'].corr(df['review_scores_rating'])
# correlation_bathrooms = df['bathrooms'].corr(df['review_scores_rating'])

# print(f"Correlation between 'ratio' and 'review_scores_rating': {correlation_ratio}")
# print(f"Correlation between 'accommodates' and 'review_scores_rating': {correlation_accommodates}")
# print(f"Correlation between 'bathrooms' and 'review_scores_rating': {correlation_bathrooms}")

def plot_scatter_and_regression(x, y, x_label, y_label, title):
    # Ensure valid inputs (no NaNs)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    # Fit a regression line
    coefficients = np.polyfit(x, y, deg=1)
    slope, intercept = coefficients
    regression_line = slope * x + intercept

    # Calculate metrics
    y_pred = regression_line
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    correlation = np.corrcoef(x, y)[0, 1]

    # Print metrics
    print(f"Slope: {slope}, Intercept: {intercept}")

    print(f"{title}")
    print(f"Correlation Coefficient: {correlation:.2f}")
    print(f"R^2: {r2:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}\n")

    # Scatter plot
    plt.scatter(x, y, label='Data points')
    plt.plot(x, regression_line, color='red', label=f'Regression Line (y={slope:.2f}x + {intercept:.2f})')

    # Add labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

# Ensure 'df' has no missing values for the required columns
df = df.dropna(subset=['accommodates', 'bathrooms', 'review_scores_rating', 'ratio'])

# Plot for ratio vs review_scores_rating
plot_scatter_and_regression(
    df['ratio'].values, df['review_scores_rating'].values,
    x_label="Ratio", y_label="Review Scores Rating",
    title="Scatter Plot: Ratio vs Review Scores Rating"
)

# Plot for accommodates vs review_scores_rating
plot_scatter_and_regression(
    df['accommodates'].values, df['review_scores_rating'].values,
    x_label="Accommodates", y_label="Review Scores Rating",
    title="Scatter Plot: Accommodates vs Review Scores Rating"
)

# Plot for bathrooms vs review_scores_rating
plot_scatter_and_regression(
    df['bathrooms'].values, df['review_scores_rating'].values,
    x_label="Bathrooms", y_label="Review Scores Rating",
    title="Scatter Plot: Bathrooms vs Review Scores Rating"
)