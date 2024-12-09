from lxml import html
import pandas as pd
import requests

df = pd.read_excel("listings.xlsx", sheet_name="crime rates")

for index, row in df.iterrows():
    link = row["neighbourhood_link"]

    if pd.isna(link) or link == "drop":
        continue

    response = requests.get(link)
    tree = html.fromstring(response.content)
    crime_rate = tree.xpath('//div[@class="container" and @id="crime-jmp"]//div[@class="facts-box-body"][1]//em')
    housing_prices = tree.xpath('//div[@class="container" and @id="housing-jmp"]//div[@class="facts-box-body"][1]//em')
    
    print(index)
    if crime_rate and crime_rate[0].text.strip().lower() != 'n/a':
        df.at[index, "crime_rate"] = float(crime_rate[0].text.strip().replace(',', ''))
    else:
        df.at[index, "crime_rate"] = None  

    df.at[index, "median_housing_cost"] = float(housing_prices[0].text.strip().replace(',', '').replace('$', '')) if housing_prices else None

df.to_excel("augmented.xlsx", index=False)
