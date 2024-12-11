import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("second_case_study/augmented_listings.xlsx")
correlation = df["crime_rate"].corr(df["price_night"])
sns.regplot(x="crime_rate", y="price_night", data=df, scatter_kws={'s': 50}, line_kws={"color": "red", "lw": 2})
plt.title("Price Per Night vs. Crime Rate", fontsize=14)
plt.xlabel("Crime Rate (per 100,000 residents)", fontsize=12)
plt.ylabel("Price Per Night ($)", fontsize=12)
plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()