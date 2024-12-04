import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset/listings_dataset.csv')

#converting to dataframe
df = pd.DataFrame(dataset)

#rename column
df.rename(columns={'accomodates/bathrooms ratio' : 'ratio'}, inplace=True)

# Plot the two columns as a line plot
df.plot.scatter(x='ratio', y='review_scores_rating')
plt.show()

