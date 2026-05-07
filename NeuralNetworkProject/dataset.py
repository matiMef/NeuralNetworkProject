def prepare_dataset():
    current_year = 2026
    data = pd.read_csv('kc_house_data.csv')
    data = data[["sqft_living", "sqft_lot", "lat", "long", "yr_built", "yr_renovated", "floors", "bedrooms", "bathrooms", "price"]]

    data = data[data["bedrooms"] <= 15]
    data["sqft_lot"] = data["sqft_lot"].clip(upper=213008)
    data["sqft_living"] = data["sqft_living"].clip(upper=4978)

    last_refurb_year = data["yr_renovated"].mask(data["yr_renovated"] == 0, data["yr_built"])
    data["years_since_refurb"] = current_year - last_refurb_year
    data["house_age"] = current_year - data["yr_built"]

    data = data.drop(["yr_renovated", "yr_built"], axis=1)

    X = data.drop("price", axis=1)
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    X_np = X_normalized.values
    Y_np = data["price"].values.reshape(-1, 1) # Od razu do (m, 1)

    np.random.seed(42)
    indices = np.random.permutation(len(X_np))
    tren_size = int(0.8 * len(X_np))

    tren_size = int(0.8 * len(X_np))
    val_size = int(0.1 * len(X_np))

    tren_idx = indices[:tren_size]
    val_idx = indices[tren_size : tren_size + val_size]
    test_idx = indices[tren_size + val_size:]

    X_tren, Y_tren = X_np[tren_idx], Y_np[tren_idx]
    X_val, Y_val = X_np[val_idx], Y_np[val_idx]
    X_test, Y_test = X_np[test_idx], Y_np[test_idx]

    y_mean = Y_tren.mean()
    y_std = Y_tren.std()
    Y_tren_norm = (Y_tren - y_mean) / y_std
    Y_val_norm = (Y_val - y_mean) / y_std
    Y_test_norm = (Y_test - y_mean) / y_std