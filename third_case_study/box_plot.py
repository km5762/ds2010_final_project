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

df['ratio'].replace("#DIV/0!", None, inplace=True)


# Converting to numeric
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')
df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')



# Drop null values
df = df.dropna()

def plot_boxplot(x, y, x_label, y_label, title):
    # Create a dataframe for easier plotting
    data = pd.DataFrame({x_label: x, y_label: y}).dropna()
    
    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_label, y=y_label, data=data, palette="Set2")

    # Add labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Ensure 'df' has no missing values for the required columns
df = df.dropna(subset=['accommodates', 'bathrooms', 'review_scores_rating', 'ratio'])


# Bin 'ratio' into categories
bins = [0, 1, 2, 3, 4, np.inf]  # Define bin edges
labels = ['0-1', '1-2', '2-3', '3-4', '4+']  # Define bin labels
df['ratio_bins'] = pd.cut(df['ratio'], bins=bins, labels=labels, right=False)

# Box plot for ratio_bins vs review_scores_rating
plot_boxplot(
    df['ratio_bins'], df['review_scores_rating'],
    x_label="Ratio (Binned)", y_label="Review Scores Rating",
    title="Box Plot: Ratio (Binned) vs Review Scores Rating"
)

# Box plot for accommodates vs review_scores_rating
plot_boxplot(
    df['accommodates'], df['review_scores_rating'],
    x_label="Accommodates", y_label="Review Scores Rating",
    title="Box Plot: Accommodates vs Review Scores Rating"
)

# Box plot for bathrooms vs review_scores_rating
plot_boxplot(
    df['bathrooms'], df['review_scores_rating'],
    x_label="Bathrooms", y_label="Review Scores Rating",
    title="Box Plot: Bathrooms vs Review Scores Rating"
)

