import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import MSE_MAE_test, eval_chart

def show_features_importance(model) -> None:
    features = ["sqft_living", "sqft_lot", "lat", "long", "floors", 
                "bedrooms", "bathrooms", "years_since_refurb", "house_age"]
    
    # Tworzymy Serię dla łatwiejszego sortowania
    coefs = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
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

def random_forest(ds, _n_estimators, _max_depth) -> list:
    rf_regr = RandomForestRegressor(
        n_estimators=_n_estimators, 
        max_depth=_max_depth, 
        random_state=42,
        verbose=1 
    )

    rf_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())

    mse_test, mae_test = MSE_MAE_test(ds, rf_regr)
    show_features_importance(rf_regr)

    y_pred_mlp_norm = rf_regr.predict(ds.X_test)
    y_pred_mlp_dollars = y_pred_mlp_norm * ds.y_std + ds.y_mean
    eval_chart(ds, y_pred_mlp_dollars)

    return mse_test, mae_test 