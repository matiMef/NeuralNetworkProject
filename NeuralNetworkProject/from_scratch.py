import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv('kc_house_data.csv')
data = data[["sqft_living15", "sqft_lot15", "lat", "long", "yr_built", "yr_renovated", "floors", "bedrooms", "bathrooms", "price"]]
data = data[data["bedrooms"] <= 15]
data["sqft_lot15"] = data["sqft_lot15"].clip(upper=157687)
X = data.drop("price", axis=1)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

mask = (X_normalized > -5) & (X_normalized < 32)
ok = mask.all().all()
avg = X_normalized.mean()
median = X_normalized.median()
max = X_normalized.max()
min = X_normalized.min()

threshold = data["sqft_lot15"].quantile(0.99)
print(f"99% danych mieści się poniżej wartości: {threshold}")
print(min)
print(max)
print(median)
print(avg)
print(ok)
print(X_normalized)
print(data)

