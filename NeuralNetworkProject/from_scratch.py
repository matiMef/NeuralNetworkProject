import numpy as np
import pandas as pd
import matplotlib as plt

current_year = 2026
data = pd.read_csv('kc_house_data.csv')
data = data[["sqft_living", "sqft_lot", "lat", "long", "yr_built", "yr_renovated", "floors", "bedrooms", "bathrooms", "price"]]
price = data["price"]

data = data[data["bedrooms"] <= 15]
data["sqft_lot"] = data["sqft_lot"].clip(upper=213008)
data["sqft_living"] = data["sqft_living"].clip(upper=4978)

last_refurb_year = data["yr_renovated"].mask(data["yr_renovated"] == 0, data["yr_built"])
data["years_since_refurb"] = current_year - last_refurb_year
data["house_age"]  = current_year - data["yr_built"]

data = data.drop("yr_renovated", axis=1)
data = data.drop("yr_built", axis=1)

X = data.drop("price", axis=1)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

def dataset_testing():
    print(price)
    print(data["price"].head(50))
    mask = (X_normalized > -4) & (X_normalized < 8.5)
    ok = mask.all().all()
    avg = X_normalized.mean()
    median = X_normalized.median()
    max = X_normalized.max()
    min = X_normalized.min()
    threshold = data["sqft_lot"].quantile(0.99)
    print(f"99% danych mieści się poniżej wartości: {threshold}")
    threshold2 = data["sqft_living"].quantile(0.99)
    print(f"99% danych mieści się poniżej wartości: {threshold2}")
    print(min)
    print(max)
    print(median)
    print(avg)
    print(ok)
    print(X_normalized)
    print(data)

