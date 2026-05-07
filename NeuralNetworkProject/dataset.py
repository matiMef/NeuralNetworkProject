import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path='kc_house_data.csv', current_year=2026):
        np.random.seed(42)
        
        # 1. Wczytywanie i wstępna selekcja
        data = pd.read_csv(file_path)
        data = data[["sqft_living", "sqft_lot", "lat", "long", "yr_built", "yr_renovated", "floors", "bedrooms", "bathrooms", "price"]]

        # 2. Clipping i czyszczenie
        data = data[data["bedrooms"] <= 15]
        data["sqft_lot"] = data["sqft_lot"].clip(upper=213008)
        data["sqft_living"] = data["sqft_living"].clip(upper=4978)

        # 3. Feature Engineering
        last_refurb_year = data["yr_renovated"].mask(data["yr_renovated"] == 0, data["yr_built"])
        data["years_since_refurb"] = current_year - last_refurb_year
        data["house_age"] = current_year - data["yr_built"]
        data = data.drop(["yr_renovated", "yr_built"], axis=1)
        
        # 4. Podział na X i Y
        X = data.drop("price", axis=1)
        Y = data["price"].values.reshape(-1, 1)

        # 5. Losowanie indeksów
        indices = np.random.permutation(len(X))
        tren_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        train_idx = indices[:tren_size]
        val_idx = indices[tren_size : tren_size + val_size]
        test_idx = indices[tren_size + val_size:]

        # 6. Fizyczny podział (zanim znormalizujemy)
        self.X_tren_raw, self.Y_tren = X.iloc[train_idx], Y[train_idx]
        self.X_val_raw, self.Y_val = X.iloc[val_idx], Y[val_idx]
        self.X_test_raw, self.Y_test = X.iloc[test_idx], Y[test_idx]

        # 7. Normalizacja X (liczona tylko na treningowym!)
        self.X_mean = self.X_tren_raw.mean(axis=0)
        self.X_std = self.X_tren_raw.std(axis=0)

        self.X_tren = (self.X_tren_raw - self.X_mean) / self.X_std
        self.X_val = (self.X_val_raw - self.X_mean) / self.X_std
        self.X_test = (self.X_test_raw - self.X_mean) / self.X_std

        # Zamiana na numpy po normalizacji
        self.X_tren, self.X_val, self.X_test = self.X_tren.values, self.X_val.values, self.X_test.values

        # 8. Normalizacja Y
        self.y_mean = self.Y_tren.mean()
        self.y_std = self.Y_tren.std()
        
        self.Y_tren_norm = (self.Y_tren - self.y_mean) / self.y_std
        self.Y_val_norm = (self.Y_val - self.y_mean) / self.y_std
        self.Y_test_norm = (self.Y_test - self.y_mean) / self.y_std