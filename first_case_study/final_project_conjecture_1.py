
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

data = pd.read_csv("listings_dataset.csv")

data['review_scores_rating'] = data['review_scores_rating'].fillna(0) # where review_scores_rating is NA, put a 0

clean_data = data.dropna(subset='price_night') # drop the rows where there is no value in price_night
clean_data = clean_data.dropna(subset='host_is_superhost') # drop the rows where there is no value for host_is_superhost

conjecture_data = clean_data[['host_is_superhost', 'price_night', 'review_scores_rating']] # only includes the three columns that this conjecture is focused on

# split the data into two parts, superhosts and non-superhosts
sh = conjecture_data[conjecture_data['host_is_superhost'] == 't'] # sh meaning superhost
nsh = conjecture_data[conjecture_data['host_is_superhost'] == 'f'] # nsh meaning nonsuperhost

# print the describe() for each portion of the dataset
print(sh.describe())
print(nsh.describe())

# cleaning the data to make histograms more readable
sh_hist = sh.loc[sh['price_night'] <= 1000] # there are only 500/15000 entries above $1000/night, makes the histogram more readable
nsh_hist = nsh.loc[nsh['price_night'] <= 1000] # only 1000/20000 entries at > $1000/night, makes histogram more readable
print(sh_hist.describe())
print(nsh_hist.describe())

# histogram for superhost price per night
plt.figure(figsize = (8,6))
plt.hist(sh_hist['price_night'], bins=100)
plt.xlabel('Price per Night')
plt.ylabel('Frequency')
plt.title('Histogram for Superhost Price per Night')
plt.legend()
plt.show()

# histogram for nonsuperhost price per night
plt.figure(figsize = (8,6))
plt.hist(nsh_hist['price_night'], bins=100)
plt.xlabel('Price per Night')
plt.ylabel('Frequency')
plt.title('Histogram for Non-Superhost Price per Night')
plt.legend()
plt.show()

# histogram for superhost review score
plt.figure(figsize = (8,6))
plt.hist(sh_hist['review_scores_rating'], bins=100)
plt.xlabel('Review Score')
plt.ylabel('Frequency')
plt.title('Histogram for Superhost Review Score')
plt.legend()
plt.show()

# histogram for nonsuperhost review score
plt.figure(figsize = (8,6))
plt.hist(nsh_hist['review_scores_rating'], bins=100)
plt.xlabel('Review Score')
plt.ylabel('Frequency')
plt.title('Histogram for Non-Superhost Review Score')
plt.legend()
plt.show()

# Mann-Whitney U Test to compare the distributions of the superhosts and nonsuperhosts for both price per night and review score
print('Price per Night Mann-Whitney U Test:')
stat, p = mannwhitneyu(sh['price_night'], nsh['price_night'])
print('stat=%f, p=%f' % (stat, p))
if p > 0.05:
    print('No Significant Difference between Superhosts and Non-Superhosts')
else:
    print('Significant Difference between Superhosts and Non-Superhosts')

print('Review Score Mann-Whitney U Test:')
stat, p = mannwhitneyu(sh['review_scores_rating'], nsh['review_scores_rating'])
print('stat=%f, p=%f' % (stat, p))
if p > 0.05:
    print('No Significant Difference between Superhosts and Non-Superhosts')
else:
    print('Significant Difference between Superhosts and Non-Superhosts')