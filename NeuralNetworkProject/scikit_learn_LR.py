import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import MSE_MAE_test, eval_chart

import pandas as pd
import matplotlib.pyplot as plt

def show_features_importance(model) -> None:
    features = ["sqft_living", "sqft_lot", "lat", "long", "floors", 
                "bedrooms", "bathrooms", "years_since_refurb", "house_age"]
    
    # Tworzymy Serię dla łatwiejszego sortowania
    coefs = pd.Series(model.coef_, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Używamy posortowanych danych do wykresu
    colors = ['skyblue' if c > 0 else 'salmon' for c in coefs]
    plt.bar(coefs.index, coefs.values, color=colors)
    
    # Dodatki estetyczne
    plt.axhline(0, color='black', linewidth=0.8) # Linia zero
    plt.title("Istotność cech - Regresja Liniowa (Współczynniki)")
    plt.ylabel("Wartość współczynnika")
    plt.xlabel("Cechy")
    
    # Obracamy etykiety, żeby się nie nakładały
    plt.xticks(rotation=45)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\nISTOTNOŚĆ CECH (REGRESJA LINIOWA) - wartości:")
    print(coefs)

def linear_regression(ds) -> list:
    line_regr = LinearRegression()
    line_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    mse_test, mae_test = MSE_MAE_test(ds, line_regr)
    show_features_importance(line_regr)
    
    y_pred_mlp_norm = line_regr.predict(ds.X_test)
    y_pred_mlp_dollars = y_pred_mlp_norm * ds.y_std + ds.y_mean
    eval_chart(ds, y_pred_mlp_dollars)

    return mse_test, mae_test 