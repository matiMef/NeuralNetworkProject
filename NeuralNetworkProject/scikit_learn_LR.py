import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import MSE_MAE_val, MSE_MAE_test

def show_features_importance(model) -> None:
    features = ["sqft_living", "sqft_lot", "lat", "long", "floors", 
                "bedrooms", "bathrooms", "years_since_refurb", "house_age"]
    
    coefs = pd.Series(model.coef_, index=features)
    
    print("\nISTOTNOŚĆ CECH (REGRESJA LINIOWA)")
    print(coefs.sort_values(ascending=False))

def linear_regression(ds) -> list:
    line_regr = LinearRegression()
    line_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    mse_val, mae_val = MSE_MAE_val(ds, line_regr)
    mse_test, mae_test = MSE_MAE_test(ds, line_regr)
    show_features_importance(line_regr)
    return mse_test, mae_test 