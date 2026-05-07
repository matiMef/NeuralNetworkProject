import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def MSE_MAE_summary(ds, model) -> float:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mae_test

def show_features_importance(model) -> None:
    features = ["sqft_living", "sqft_lot", "lat", "long", "floors", 
                "bedrooms", "bathrooms", "years_since_refurb", "house_age"]
    
    coefs = pd.Series(model.coef_, index=features)
    
    print("\nISTOTNOŚĆ CECH (REGRESJA LINIOWA)")
    print(coefs.sort_values(ascending=False))

def linear_regression(ds) -> None:
    line_regr = LinearRegression()
    line_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    MSE_MAE_summary(ds, line_regr)
    show_features_importance(line_regr)

    